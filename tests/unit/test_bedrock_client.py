"""
Unit tests for BedrockClient using moto mocks.
Tests model invocation, streaming, embeddings, and error handling.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.bedrock_client import BedrockClient, EmbeddingResult, InferenceResult


class TestBedrockClient:
    """Tests for BedrockClient foundation model invocation."""

    @patch("src.models.bedrock_client.get_config")
    @patch("boto3.Session")
    def test_invoke_returns_inference_result(
        self, mock_session_cls: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """BedrockClient.invoke should return a populated InferenceResult."""
        # Arrange
        mock_config = MagicMock()
        mock_config.bedrock.region = "ap-southeast-2"
        mock_config.get_model_id.return_value = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        mock_config.bedrock.inference = {
            "max_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
            "timeout_seconds": 120,
        }
        mock_get_config.return_value = mock_config

        mock_response_body = json.dumps({
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }).encode()

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: mock_response_body)
        }

        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        # Act
        client = BedrockClient(model_key="claude_sonnet")
        result = client.invoke("Hello, Claude")

        # Assert
        assert isinstance(result, InferenceResult)
        assert result.content == "Hello from Claude"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.stop_reason == "end_turn"
        assert result.total_tokens == 15
        assert result.estimated_cost_usd > 0

    @patch("src.models.bedrock_client.get_config")
    @patch("boto3.Session")
    def test_invoke_with_system_prompt(
        self, mock_session_cls: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """System prompt should be included in the request body."""
        mock_config = MagicMock()
        mock_config.bedrock.region = "ap-southeast-2"
        mock_config.get_model_id.return_value = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        mock_config.bedrock.inference = {
            "max_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "timeout_seconds": 120
        }
        mock_get_config.return_value = mock_config

        mock_response_body = json.dumps({
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"input_tokens": 15, "output_tokens": 3},
            "stop_reason": "end_turn",
        }).encode()

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: mock_response_body)
        }
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        client = BedrockClient()
        client.invoke("What is 2+2?", system="You are a maths tutor.")

        call_kwargs = mock_client.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs.get("body", call_kwargs.args[0] if call_kwargs.args else "{}"))
        assert "system" in body
        assert body["system"] == "You are a maths tutor."

    @patch("src.models.bedrock_client.get_config")
    @patch("boto3.Session")
    def test_embed_returns_embedding_result(
        self, mock_session_cls: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """BedrockClient.embed should return an EmbeddingResult with correct dimensions."""
        mock_config = MagicMock()
        mock_config.bedrock.region = "ap-southeast-2"
        mock_config.get_model_id.side_effect = lambda key: {
            "claude_sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "titan_embed": "amazon.titan-embed-text-v2:0",
        }.get(key, "")
        mock_config.bedrock.inference = {
            "max_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "timeout_seconds": 120
        }
        mock_get_config.return_value = mock_config

        fake_embedding = [0.1] * 1024
        mock_embed_body = json.dumps({
            "embedding": fake_embedding,
            "inputTextTokenCount": 8,
        }).encode()

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: mock_embed_body)
        }
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        client = BedrockClient()
        result = client.embed("Amazon Bedrock is a managed AI service")

        assert isinstance(result, EmbeddingResult)
        assert result.dimensions == 1024
        assert len(result.embedding) == 1024
        assert result.input_tokens == 8

    @patch("src.models.bedrock_client.get_config")
    @patch("boto3.Session")
    def test_invoke_stream_yields_chunks(
        self, mock_session_cls: MagicMock, mock_get_config: MagicMock
    ) -> None:
        """invoke_stream should yield text chunks from the event stream."""
        mock_config = MagicMock()
        mock_config.bedrock.region = "ap-southeast-2"
        mock_config.get_model_id.return_value = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        mock_config.bedrock.inference = {
            "max_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "timeout_seconds": 120
        }
        mock_get_config.return_value = mock_config

        chunks = [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}},
            {"type": "message_stop"},
        ]

        mock_stream = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]

        mock_client = MagicMock()
        mock_client.invoke_model_with_response_stream.return_value = {"body": mock_stream}
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        client = BedrockClient()
        result = "".join(client.invoke_stream("Say hello"))
        assert result == "Hello world"

    def test_inference_result_total_tokens(self) -> None:
        """InferenceResult.total_tokens should sum input and output tokens."""
        result = InferenceResult(
            content="test",
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            input_tokens=100,
            output_tokens=50,
        )
        assert result.total_tokens == 150

    def test_embedding_result_dimensions(self) -> None:
        """EmbeddingResult dimensions should be computed from embedding length."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_id="amazon.titan-embed-text-v2:0",
        )
        assert result.dimensions == 4
