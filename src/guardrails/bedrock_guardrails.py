"""
Amazon Bedrock Guardrails — content safety enforcement.

Provides:
- Apply guardrails to arbitrary text (input and output scanning)
- Integrated guardrail invocation with model calls
- PII detection and redaction
- Intervention classification and audit logging
- Batch content scanning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GuardrailAction(str, Enum):
    NONE = "NONE"
    INTERVENED = "INTERVENED"


class ContentPolicyAction(str, Enum):
    BLOCKED = "BLOCKED"
    NONE = "NONE"


@dataclass
class PIIEntity:
    """A detected PII entity in content."""

    entity_type: str
    match: str
    action: str


@dataclass
class GuardrailResult:
    """Result of applying Bedrock Guardrails to content."""

    action: GuardrailAction
    output_text: str
    original_text: str

    # Policy interventions
    content_policy_action: ContentPolicyAction = ContentPolicyAction.NONE
    topic_policy_action: ContentPolicyAction = ContentPolicyContent = ContentPolicyAction.NONE
    word_policy_action: ContentPolicyAction = ContentPolicyAction.NONE
    sensitive_info_action: ContentPolicyAction = ContentPolicyAction.NONE
    grounding_action: ContentPolicyAction = ContentPolicyAction.NONE

    # Detected entities
    pii_entities: list[PIIEntity] = field(default_factory=list)
    blocked_topics: list[str] = field(default_factory=list)
    blocked_words: list[str] = field(default_factory=list)

    # Scores
    grounding_score: float = 0.0
    relevance_score: float = 0.0

    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def was_intervened(self) -> bool:
        return self.action == GuardrailAction.INTERVENED

    @property
    def safe_output(self) -> str:
        """Return output text — may be the guardrail-modified version."""
        return self.output_text

    def audit_summary(self) -> dict[str, Any]:
        """Return a structured summary for audit logging."""
        return {
            "action": self.action.value,
            "intervened": self.was_intervened,
            "pii_detected": len(self.pii_entities),
            "pii_types": [e.entity_type for e in self.pii_entities],
            "blocked_topics": self.blocked_topics,
            "blocked_words": self.blocked_words,
            "grounding_score": self.grounding_score,
            "relevance_score": self.relevance_score,
        }


@dataclass
class GuardrailResult:
    """Result of applying Bedrock Guardrails to content."""

    action: GuardrailAction
    output_text: str
    original_text: str
    content_policy_action: ContentPolicyAction = ContentPolicyAction.NONE
    topic_policy_action: ContentPolicyAction = ContentPolicyAction.NONE
    word_policy_action: ContentPolicyAction = ContentPolicyAction.NONE
    sensitive_info_action: ContentPolicyAction = ContentPolicyAction.NONE
    grounding_action: ContentPolicyAction = ContentPolicyAction.NONE
    pii_entities: list[PIIEntity] = field(default_factory=list)
    blocked_topics: list[str] = field(default_factory=list)
    blocked_words: list[str] = field(default_factory=list)
    grounding_score: float = 0.0
    relevance_score: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def was_intervened(self) -> bool:
        return self.action == GuardrailAction.INTERVENED

    @property
    def safe_output(self) -> str:
        return self.output_text

    def audit_summary(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "intervened": self.was_intervened,
            "pii_detected": len(self.pii_entities),
            "pii_types": [e.entity_type for e in self.pii_entities],
            "blocked_topics": self.blocked_topics,
            "blocked_words": self.blocked_words,
            "grounding_score": self.grounding_score,
            "relevance_score": self.relevance_score,
        }


class BedrockGuardrailsClient:
    """
    Production Bedrock Guardrails enforcement client.

    Apply guardrails as a scanning layer over any text — model inputs,
    model outputs, or arbitrary user content.

    Example:
        gc = BedrockGuardrailsClient()

        # Scan user input before sending to model
        result = gc.apply_guardrail(user_message, source="INPUT")
        if result.was_intervened:
            return "I cannot process that request."

        # Scan model output before returning to user
        result = gc.apply_guardrail(model_output, source="OUTPUT")
        return result.safe_output
    """

    def __init__(
        self,
        guardrail_id: str | None = None,
        guardrail_version: str | None = None,
        region: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        cfg = get_config()
        guard_cfg = cfg.bedrock_guardrails if hasattr(cfg, "bedrock_guardrails") else {}

        raw_cfg = get_config.__wrapped__() if hasattr(get_config, "__wrapped__") else {}

        from src.utils.config import load_config
        raw = load_config()
        guard_raw = raw.get("bedrock_guardrails", {})

        self.guardrail_id = guardrail_id or guard_raw.get("guardrail_id", "")
        self.guardrail_version = guardrail_version or guard_raw.get("guardrail_version", "DRAFT")
        self.region = region or raw.get("aws", {}).get("region", "ap-southeast-2")

        self._session = session or boto3.Session()
        self._client = self._session.client("bedrock-runtime", region_name=self.region)
        self._bedrock_client = self._session.client("bedrock", region_name=self.region)

        logger.info(
            "BedrockGuardrailsClient initialised",
            guardrail_id=self.guardrail_id,
            guardrail_version=self.guardrail_version,
        )

    def apply_guardrail(
        self,
        text: str,
        source: str = "INPUT",
        context: str | None = None,
    ) -> GuardrailResult:
        """
        Apply guardrails to a piece of text.

        Args:
            text: The text to evaluate.
            source: "INPUT" for user-provided text, "OUTPUT" for model-generated text.
            context: Optional grounding context for grounding checks (OUTPUT only).

        Returns:
            GuardrailResult with action, modified text, and all policy assessments.
        """
        content: list[dict[str, Any]] = [{"text": {"text": text}}]

        request: dict[str, Any] = {
            "guardrailIdentifier": self.guardrail_id,
            "guardrailVersion": self.guardrail_version,
            "source": source,
            "content": content,
        }

        if context and source == "OUTPUT":
            request["content"].append({
                "text": {"text": context, "qualifiers": ["grounding_source"]}
            })

        try:
            response = self._client.apply_guardrail(**request)
        except ClientError as exc:
            logger.error("Guardrail application failed", error=str(exc))
            raise

        return self._parse_guardrail_response(response, original_text=text)

    def scan_input(self, user_message: str) -> GuardrailResult:
        """Convenience wrapper — scan a user input before forwarding to a model."""
        result = self.apply_guardrail(user_message, source="INPUT")
        if result.was_intervened:
            logger.warning(
                "Guardrail blocked input",
                summary=result.audit_summary(),
                text_snippet=user_message[:80],
            )
        return result

    def scan_output(self, model_output: str, grounding_context: str | None = None) -> GuardrailResult:
        """Convenience wrapper — scan model output before returning to a user."""
        result = self.apply_guardrail(model_output, source="OUTPUT", context=grounding_context)
        if result.was_intervened:
            logger.warning(
                "Guardrail modified output",
                summary=result.audit_summary(),
            )
        return result

    def scan_batch(
        self, texts: list[str], source: str = "INPUT"
    ) -> list[GuardrailResult]:
        """Scan a batch of texts through guardrails."""
        results = []
        for i, text in enumerate(texts):
            logger.debug("Scanning batch item", index=i, total=len(texts))
            results.append(self.apply_guardrail(text, source=source))
        return results

    def detect_pii(self, text: str) -> list[PIIEntity]:
        """Extract only PII entities from text, ignoring other policy checks."""
        result = self.apply_guardrail(text, source="INPUT")
        return result.pii_entities

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_guardrail_response(
        self, response: dict[str, Any], original_text: str
    ) -> GuardrailResult:
        """Parse the raw AWS guardrail response into a GuardrailResult."""
        action = GuardrailAction(response.get("action", "NONE"))
        outputs = response.get("outputs", [])
        output_text = outputs[0].get("text", original_text) if outputs else original_text

        assessments = response.get("assessments", [])
        pii_entities: list[PIIEntity] = []
        blocked_topics: list[str] = []
        blocked_words: list[str] = []
        grounding_score = 0.0
        relevance_score = 0.0

        content_action = ContentPolicyAction.NONE
        topic_action = ContentPolicyAction.NONE
        word_action = ContentPolicyAction.NONE
        sensitive_action = ContentPolicyAction.NONE
        grounding_action = ContentPolicyAction.NONE

        for assessment in assessments:
            # Content filters (hate, violence, sexual, etc.)
            if "contentPolicy" in assessment:
                cp = assessment["contentPolicy"]
                if cp.get("filters"):
                    content_action = ContentPolicyAction.BLOCKED

            # Topic policies
            if "topicPolicy" in assessment:
                tp = assessment["topicPolicy"]
                for topic in tp.get("topics", []):
                    if topic.get("action") == "BLOCKED":
                        topic_action = ContentPolicyAction.BLOCKED
                        blocked_topics.append(topic.get("name", ""))

            # Word filters
            if "wordPolicy" in assessment:
                wp = assessment["wordPolicy"]
                for word in wp.get("customWords", []):
                    if word.get("action") == "BLOCKED":
                        word_action = ContentPolicyAction.BLOCKED
                        blocked_words.append(word.get("match", ""))

            # Sensitive information (PII)
            if "sensitiveInformationPolicy" in assessment:
                sip = assessment["sensitiveInformationPolicy"]
                for entity in sip.get("piiEntities", []):
                    if entity.get("action") in ("BLOCKED", "ANONYMIZED"):
                        sensitive_action = ContentPolicyAction.BLOCKED
                        pii_entities.append(PIIEntity(
                            entity_type=entity.get("type", ""),
                            match=entity.get("match", ""),
                            action=entity.get("action", ""),
                        ))

            # Grounding
            if "groundingPolicy" in assessment:
                gp = assessment["groundingPolicy"]
                grounding_score = gp.get("groundingScore", 0.0)
                relevance_score = gp.get("relevanceScore", 0.0)
                if gp.get("action") == "BLOCKED":
                    grounding_action = ContentPolicyAction.BLOCKED

        result = GuardrailResult(
            action=action,
            output_text=output_text,
            original_text=original_text,
            content_policy_action=content_action,
            topic_policy_action=topic_action,
            word_policy_action=word_action,
            sensitive_info_action=sensitive_action,
            grounding_action=grounding_action,
            pii_entities=pii_entities,
            blocked_topics=blocked_topics,
            blocked_words=blocked_words,
            grounding_score=grounding_score,
            relevance_score=relevance_score,
            usage=response.get("usage", {}),
            raw_response=response,
        )

        logger.info(
            "Guardrail assessment complete",
            action=action.value,
            pii_count=len(pii_entities),
            blocked_topics=blocked_topics,
            grounding_score=grounding_score,
        )

        return result
