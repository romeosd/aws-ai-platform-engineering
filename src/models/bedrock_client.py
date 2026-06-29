"""
Amazon Bedrock foundation model client.

Provides a unified interface for:
- Synchronous and streaming text generation
- Multi-modal (image + text) invocation
- Embeddings generation
- Automatic retry with exponential backoff
- Token usage tracking and cost estimation
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

import boto3
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Cost per 1,000 tokens (USD) — ap-southeast-2 pricing, update as AWS changes rates
_TOKEN_COSTS: dict[str, dict[str, float]] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
    "amazon.titan-text-premier-v1:0": {"input": 0.0005, "output": 0.0015},
    "meta.llama3-70b-instruct-v1:0": {"input": 0.00265, "output": 0.0035},
    "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012},
}


@dataclass
class InferenceResult:
    """Structured result from a Bedrock model invocation."""

    content: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        return (
            f"InferenceResult(model={self.model_id}, "
            f"tokens={self.total_tokens}, "
            f"latency={self.latency_ms:.0f}ms, "
            f"cost=${self.estimated_cost_usd:.6f})"
        )


@dataclass
class EmbeddingResult:
    """Structured result from a Bedrock embedding invocation."""

    embedding: list[float]
    model_id: str
    input_tokens: int = 0
    dimensions: int = 0

    def __post_init__(self) -> None:
        self.dimensions = len(self.embedding)


class BedrockClient:
    """
    Production-grade Amazon Bedrock client.

    Example:
        client = BedrockClient()

        # Text generation
        result = client.invoke("Explain transformer attention mechanisms")
        print(result.content)

        # Streaming
        for chunk in client.invoke_stream("Write a short story about AWS"):
            print(chunk, end="", flush=True)

        # Embeddings
        emb = client.embed("Amazon Bedrock is a fully managed service")
        print(f"Embedding dimensions: {emb.dimensions}")
    """

    def __init__(
        self,
        model_key: str = "claude_sonnet",
        region: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        cfg = get_config()
        self.model_id = cfg.get_model_id(model_key)
        self.region = region or cfg.bedrock.region
        self.inference_cfg = cfg.bedrock.inference

        self._session = session or boto3.Session()
        self._client = self._session.client("bedrock-runtime", region_name=self.region)

        logger.info("BedrockClient initialised", model_id=self.model_id, region=self.region)

    @retry(
        retry=retry_if_exception_type((ClientError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def invoke(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        image_path: str | Path | None = None,
    ) -> InferenceResult:
        """
        Invoke a Bedrock foundation model synchronously.

        Args:
            prompt: The user message / prompt text.
            system: Optional system prompt.
            max_tokens: Override config max_tokens.
            temperature: Override config temperature.
            top_p: Override config top_p.
            image_path: Path to an image file for multi-modal models.

        Returns:
            InferenceResult with content, token counts, and cost estimate.
        """
        body = self._build_request_body(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens or self.inference_cfg["max_tokens"],
            temperature=temperature if temperature is not None else self.inference_cfg["temperature"],
            top_p=top_p or self.inference_cfg["top_p"],
            image_path=image_path,
        )

        start = time.perf_counter()
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            logger.error("Bedrock invocation failed", error_code=error_code, model_id=self.model_id)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        raw = json.loads(response["body"].read())

        return self._parse_response(raw, latency_ms)

    def invoke_stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from a Bedrock foundation model.

        Yields:
            Text chunks as they are received from the model.
        """
        body = self._build_request_body(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens or self.inference_cfg["max_tokens"],
            temperature=temperature if temperature is not None else self.inference_cfg["temperature"],
            top_p=self.inference_cfg["top_p"],
        )

        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as exc:
            logger.error("Bedrock stream invocation failed", error=str(exc))
            raise

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")

    def embed(
        self,
        text: str,
        model_key: str = "titan_embed",
        normalize: bool = True,
    ) -> EmbeddingResult:
        """
        Generate a vector embedding for the given text.

        Args:
            text: The text to embed.
            model_key: Config key for the embedding model.
            normalize: Whether to request normalised embeddings (Titan v2).

        Returns:
            EmbeddingResult containing the embedding vector.
        """
        cfg = get_config()
        embed_model_id = cfg.get_model_id(model_key)

        body: dict[str, Any] = {"inputText": text}
        if "titan-embed-text-v2" in embed_model_id:
            body["normalize"] = normalize
            body["dimensions"] = 1024

        response = self._client.invoke_model(
            modelId=embed_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        raw = json.loads(response["body"].read())

        return EmbeddingResult(
            embedding=raw["embedding"],
            model_id=embed_model_id,
            input_tokens=raw.get("inputTextTokenCount", 0),
        )

    def embed_batch(self, texts: list[str], model_key: str = "titan_embed") -> list[EmbeddingResult]:
        """Generate embeddings for a list of texts."""
        results = []
        for i, text in enumerate(texts):
            logger.debug("Embedding text batch", index=i, total=len(texts))
            results.append(self.embed(text, model_key=model_key))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_request_body(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build the Bedrock Messages API request body."""
        content: list[dict[str, Any]] = []

        if image_path is not None:
            img_bytes = Path(image_path).read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            suffix = Path(str(image_path)).suffix.lower().lstrip(".")
            media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type_map.get(suffix, "image/jpeg"),
                    "data": img_b64,
                },
            })

        content.append({"type": "text", "text": prompt})

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": content}],
        }

        if system:
            body["system"] = system

        return body

    def _parse_response(self, raw: dict[str, Any], latency_ms: float) -> InferenceResult:
        """Parse a raw Bedrock response into an InferenceResult."""
        content = ""
        for block in raw.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = raw.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        costs = _TOKEN_COSTS.get(self.model_id, {"input": 0.0, "output": 0.0})
        estimated_cost = (input_tokens / 1000 * costs["input"]) + (
            output_tokens / 1000 * costs["output"]
        )

        result = InferenceResult(
            content=content,
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=raw.get("stop_reason", ""),
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost,
            raw_response=raw,
        )

        logger.info(
            "Bedrock inference complete",
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=f"{latency_ms:.0f}",
            cost_usd=f"{estimated_cost:.6f}",
        )

        return result
