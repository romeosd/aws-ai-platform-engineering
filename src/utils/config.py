"""
Configuration loader for AWS AI Platform Engineering.
Loads and validates aws_config.yaml with environment variable substitution.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment variables."""
    pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return pattern.sub(replacer, value)


def _process_dict(data: Any) -> Any:
    """Recursively substitute environment variables in all string values."""
    if isinstance(data, dict):
        return {k: _process_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_process_dict(item) for item in data]
    elif isinstance(data, str):
        return _substitute_env_vars(data)
    return data


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load and return the platform configuration.

    Args:
        config_path: Path to aws_config.yaml. Defaults to config/aws_config.yaml
                     relative to the project root.

    Returns:
        Processed configuration dictionary with environment variables substituted.
    """
    if config_path is None:
        # Walk up from this file to find the project root
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "aws_config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Copy config/aws_config.yaml and populate your AWS resource ARNs."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    return _process_dict(raw)


class BedrockConfig(BaseModel):
    region: str = Field(default="ap-southeast-2")
    models: dict[str, str] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    streaming: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseConfig(BaseModel):
    knowledge_base_id: str = Field(default="")
    data_source_id: str = Field(default="")
    retrieval: dict[str, Any] = Field(default_factory=dict)
    vector_store: dict[str, Any] = Field(default_factory=dict)
    chunking: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    agent_id: str = Field(default="")
    agent_alias_id: str = Field(default="")
    foundation_model: str = Field(default="")
    idle_session_ttl: int = Field(default=600)
    action_groups: list[dict[str, str]] = Field(default_factory=list)
    memory: dict[str, Any] = Field(default_factory=dict)


class PlatformConfig(BaseSettings):
    """Top-level typed configuration for the AWS AI Platform."""

    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)
    bedrock_knowledge_bases: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    bedrock_agents: AgentConfig = Field(default_factory=AgentConfig)

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "PlatformConfig":
        """Load config from YAML and return a typed PlatformConfig instance."""
        raw = load_config(config_path)
        return cls(
            bedrock=BedrockConfig(**raw.get("bedrock", {})),
            bedrock_knowledge_bases=KnowledgeBaseConfig(
                **raw.get("bedrock_knowledge_bases", {})
            ),
            bedrock_agents=AgentConfig(**raw.get("bedrock_agents", {})),
        )

    def get_model_id(self, model_key: str) -> str:
        """Retrieve a Bedrock model ID by its config key."""
        model_id = self.bedrock.models.get(model_key)
        if not model_id:
            available = list(self.bedrock.models.keys())
            raise KeyError(
                f"Model '{model_key}' not found in config. "
                f"Available models: {available}"
            )
        return model_id


# Module-level singleton accessor
_config: PlatformConfig | None = None


def get_config() -> PlatformConfig:
    """Return the singleton PlatformConfig, loading it on first call."""
    global _config
    if _config is None:
        _config = PlatformConfig.from_yaml()
    return _config
