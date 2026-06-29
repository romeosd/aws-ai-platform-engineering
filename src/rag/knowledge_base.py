"""
Amazon Bedrock Knowledge Bases — RAG pipeline.

Covers:
- Retrieve-only mode (fetch relevant chunks)
- Retrieve-and-Generate mode (full RAG with citation)
- Hybrid search (semantic + keyword)
- Source attribution and citation extraction
- Async batch retrieval
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata."""

    content: str
    score: float
    source_uri: str = ""
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def short_source(self) -> str:
        """Return a human-readable short source name."""
        if self.source_uri:
            return self.source_uri.split("/")[-1]
        return "unknown"


@dataclass
class RAGResponse:
    """Full Retrieve-and-Generate response with citations."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    session_id: str = ""

    @property
    def source_uris(self) -> list[str]:
        """Extract all unique source URIs from citations."""
        uris: list[str] = []
        for citation in self.citations:
            for ref in citation.get("retrievedReferences", []):
                loc = ref.get("location", {})
                uri = loc.get("s3Location", {}).get("uri", "")
                if uri and uri not in uris:
                    uris.append(uri)
        return uris

    def format_with_citations(self) -> str:
        """Return the answer formatted with inline source references."""
        sources = self.source_uris
        if not sources:
            return self.answer

        source_block = "\n\n**Sources:**\n" + "\n".join(
            f"[{i+1}] {uri}" for i, uri in enumerate(sources)
        )
        return self.answer + source_block


class BedrockKnowledgeBaseClient:
    """
    Production-grade Amazon Bedrock Knowledge Bases client.

    Supports both retrieve-only and retrieve-and-generate patterns.
    Configures directly from aws_config.yaml.

    Example:
        kb = BedrockKnowledgeBaseClient()

        # Retrieve relevant chunks
        chunks = kb.retrieve("What is the data retention policy?")
        for chunk in chunks:
            print(f"[{chunk.score:.3f}] {chunk.short_source}: {chunk.content[:120]}")

        # Full RAG with grounded answer + citations
        response = kb.retrieve_and_generate("Summarise our security controls")
        print(response.format_with_citations())
    """

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        region: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        cfg = get_config()
        kb_cfg = cfg.bedrock_knowledge_bases

        self.knowledge_base_id = knowledge_base_id or kb_cfg.knowledge_base_id
        self.region = region or cfg.bedrock.region
        self._retrieval_cfg = kb_cfg.retrieval

        self._session = session or boto3.Session()
        self._bedrock_agent_runtime = self._session.client(
            "bedrock-agent-runtime", region_name=self.region
        )

        self._model_arn = (
            f"arn:aws:bedrock:{self.region}::foundation-model/"
            f"{cfg.get_model_id('claude_sonnet')}"
        )

        logger.info(
            "BedrockKnowledgeBaseClient initialised",
            knowledge_base_id=self.knowledge_base_id,
            region=self.region,
        )

    def retrieve(
        self,
        query: str,
        max_results: int | None = None,
        search_type: str | None = None,
        relevance_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: The search query.
            max_results: Number of results to return (overrides config).
            search_type: SEMANTIC | HYBRID | KEYWORD (overrides config).
            relevance_threshold: Minimum relevance score filter.
            metadata_filter: Optional Bedrock metadata filter expression.

        Returns:
            List of RetrievedChunk sorted by relevance score descending.
        """
        n_results = max_results or self._retrieval_cfg.get("max_results", 10)
        s_type = search_type or self._retrieval_cfg.get("search_type", "HYBRID")
        threshold = relevance_threshold or self._retrieval_cfg.get("relevance_threshold", 0.0)

        retrieval_query: dict[str, Any] = {
            "vectorSearchConfiguration": {
                "numberOfResults": n_results,
                "overrideSearchType": s_type,
            }
        }

        if metadata_filter:
            retrieval_query["vectorSearchConfiguration"]["filter"] = metadata_filter

        try:
            response = self._bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_query,
            )
        except ClientError as exc:
            logger.error("Knowledge base retrieval failed", error=str(exc), query=query[:100])
            raise

        chunks: list[RetrievedChunk] = []
        for result in response.get("retrievalResults", []):
            score = result.get("score", 0.0)
            if score < threshold:
                continue

            location = result.get("location", {})
            source_uri = location.get("s3Location", {}).get("uri", "")

            chunks.append(
                RetrievedChunk(
                    content=result["content"]["text"],
                    score=score,
                    source_uri=source_uri,
                    source_type=location.get("type", ""),
                    metadata=result.get("metadata", {}),
                )
            )

        chunks.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            "Knowledge base retrieval complete",
            query_snippet=query[:60],
            chunks_returned=len(chunks),
            search_type=s_type,
        )

        return chunks

    def retrieve_and_generate(
        self,
        query: str,
        session_id: str | None = None,
        max_results: int | None = None,
        prompt_template: str | None = None,
    ) -> RAGResponse:
        """
        Retrieve context and generate a grounded answer using Bedrock RAG.

        Args:
            query: The user question.
            session_id: Optional session ID for multi-turn conversations.
            max_results: Number of context chunks to retrieve.
            prompt_template: Custom prompt template (uses $search_results$ placeholder).

        Returns:
            RAGResponse with grounded answer and full citation metadata.
        """
        n_results = max_results or self._retrieval_cfg.get("max_results", 10)

        knowledge_base_config: dict[str, Any] = {
            "knowledgeBaseId": self.knowledge_base_id,
            "modelArn": self._model_arn,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": n_results,
                    "overrideSearchType": self._retrieval_cfg.get("search_type", "HYBRID"),
                }
            },
        }

        if prompt_template:
            knowledge_base_config["generationConfiguration"] = {
                "promptTemplate": {"textPromptTemplate": prompt_template}
            }

        request: dict[str, Any] = {
            "input": {"text": query},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": knowledge_base_config,
            },
        }

        if session_id:
            request["sessionId"] = session_id

        try:
            response = self._bedrock_agent_runtime.retrieve_and_generate(**request)
        except ClientError as exc:
            logger.error("Retrieve-and-generate failed", error=str(exc))
            raise

        answer = response.get("output", {}).get("text", "")
        citations = response.get("citations", [])
        new_session_id = response.get("sessionId", "")

        logger.info(
            "Retrieve-and-generate complete",
            query_snippet=query[:60],
            answer_length=len(answer),
            citation_count=len(citations),
            session_id=new_session_id,
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            session_id=new_session_id,
        )

    async def retrieve_batch_async(
        self,
        queries: list[str],
        max_results: int = 5,
    ) -> list[list[RetrievedChunk]]:
        """
        Retrieve chunks for multiple queries concurrently.

        Args:
            queries: List of query strings.
            max_results: Results per query.

        Returns:
            List of chunk lists, one per query.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.retrieve, q, max_results)
            for q in queries
        ]
        return await asyncio.gather(*tasks)

    def sync_data_source(self, data_source_id: str | None = None) -> str:
        """
        Trigger a data source sync job for the knowledge base.

        Returns:
            The ingestion job ID.
        """
        cfg = get_config()
        ds_id = data_source_id or cfg.bedrock_knowledge_bases.data_source_id

        bedrock_agent = self._session.client("bedrock-agent", region_name=self.region)

        try:
            response = bedrock_agent.start_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=ds_id,
            )
        except ClientError as exc:
            logger.error("Data source sync failed", error=str(exc))
            raise

        job_id = response["ingestionJob"]["ingestionJobId"]
        logger.info("Data source sync started", job_id=job_id, data_source_id=ds_id)
        return job_id
