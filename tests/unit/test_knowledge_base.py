"""
Unit tests for BedrockKnowledgeBaseClient.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.knowledge_base import BedrockKnowledgeBaseClient, RAGResponse, RetrievedChunk


class TestBedrockKnowledgeBaseClient:
    """Tests for the Bedrock Knowledge Bases RAG client."""

    def _mock_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.bedrock.region = "ap-southeast-2"
        cfg.bedrock_knowledge_bases.knowledge_base_id = "KB123"
        cfg.bedrock_knowledge_bases.data_source_id = "DS456"
        cfg.bedrock_knowledge_bases.retrieval = {
            "max_results": 5,
            "search_type": "HYBRID",
            "relevance_threshold": 0.5,
        }
        cfg.get_model_id.return_value = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        return cfg

    @patch("src.rag.knowledge_base.get_config")
    @patch("boto3.Session")
    def test_retrieve_returns_sorted_chunks(
        self, mock_session_cls: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """retrieve() should return chunks sorted by score descending."""
        mock_get_config.return_value = self._mock_config()

        mock_retrieval_result = {
            "retrievalResults": [
                {
                    "content": {"text": "Low relevance content"},
                    "score": 0.55,
                    "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc1.pdf"}},
                    "metadata": {},
                },
                {
                    "content": {"text": "High relevance content"},
                    "score": 0.92,
                    "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc2.pdf"}},
                    "metadata": {},
                },
                {
                    "content": {"text": "Below threshold content"},
                    "score": 0.3,
                    "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc3.pdf"}},
                    "metadata": {},
                },
            ]
        }

        mock_client = MagicMock()
        mock_client.retrieve.return_value = mock_retrieval_result
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        kb = BedrockKnowledgeBaseClient()
        chunks = kb.retrieve("What is our cloud policy?", relevance_threshold=0.5)

        # Should have 2 chunks (0.3 filtered out)
        assert len(chunks) == 2
        # Should be sorted descending by score
        assert chunks[0].score == 0.92
        assert chunks[1].score == 0.55
        assert chunks[0].content == "High relevance content"

    @patch("src.rag.knowledge_base.get_config")
    @patch("boto3.Session")
    def test_retrieve_and_generate_returns_rag_response(
        self, mock_session_cls: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """retrieve_and_generate() should return a RAGResponse with answer and citations."""
        mock_get_config.return_value = self._mock_config()

        mock_rag_result = {
            "output": {"text": "Data is retained for 7 years per policy 4.2.1"},
            "citations": [
                {
                    "retrievedReferences": [
                        {
                            "location": {
                                "s3Location": {"uri": "s3://bucket/policy.pdf"}
                            }
                        }
                    ]
                }
            ],
            "sessionId": "sess-abc-123",
        }

        mock_client = MagicMock()
        mock_client.retrieve_and_generate.return_value = mock_rag_result
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        kb = BedrockKnowledgeBaseClient()
        response = kb.retrieve_and_generate("What is the data retention policy?")

        assert isinstance(response, RAGResponse)
        assert "7 years" in response.answer
        assert len(response.citations) == 1
        assert response.session_id == "sess-abc-123"
        assert "s3://bucket/policy.pdf" in response.source_uris

    def test_rag_response_format_with_citations(self) -> None:
        """format_with_citations() should append source list to answer."""
        response = RAGResponse(
            answer="Data retained for 7 years.",
            citations=[
                {
                    "retrievedReferences": [
                        {"location": {"s3Location": {"uri": "s3://bucket/policy.pdf"}}}
                    ]
                }
            ],
            session_id="sess-123",
        )

        formatted = response.format_with_citations()
        assert "Data retained for 7 years." in formatted
        assert "s3://bucket/policy.pdf" in formatted
        assert "Sources:" in formatted

    def test_retrieved_chunk_short_source(self) -> None:
        """short_source should return just the filename from the URI."""
        chunk = RetrievedChunk(
            content="Some content",
            score=0.8,
            source_uri="s3://my-bucket/documents/policy-v2.pdf",
        )
        assert chunk.short_source == "policy-v2.pdf"
