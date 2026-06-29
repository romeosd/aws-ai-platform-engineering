"""
Amazon Bedrock Agents — orchestration client.

Provides:
- Agent session management and invocation
- Streaming agent traces for observability
- Action group response handling
- Multi-turn conversation with memory
- Inline agent creation for ephemeral tasks
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Generator

import boto3
from botocore.exceptions import ClientError

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentTrace:
    """A single trace event from a Bedrock Agent invocation."""

    trace_type: str           # orchestration | preProcessing | postProcessing | failureTrace
    step: str = ""
    thought: str = ""
    action: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    final_response: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Complete response from a Bedrock Agent invocation."""

    output: str
    session_id: str
    traces: list[AgentTrace] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reasoning_steps(self) -> list[str]:
        """Extract the chain-of-thought reasoning from orchestration traces."""
        return [
            t.thought
            for t in self.traces
            if t.trace_type == "orchestration" and t.thought
        ]

    @property
    def action_calls(self) -> list[dict[str, Any]]:
        """Extract all tool/action invocations from traces."""
        return [t.action for t in self.traces if t.action]

    def print_reasoning(self) -> None:
        """Print the agent's reasoning chain to stdout."""
        for i, step in enumerate(self.reasoning_steps, 1):
            print(f"\n[Step {i}] {step}")
        print(f"\n[Final Answer]\n{self.output}")


class BedrockAgentClient:
    """
    Production-grade Bedrock Agents client.

    Manages agent sessions, handles streaming event streams, and
    parses agent traces for observability.

    Example:
        agent = BedrockAgentClient()
        session_id = agent.new_session()

        # Single-turn invocation
        response = agent.invoke("What is our current cloud spend?", session_id=session_id)
        print(response.output)

        # Multi-turn conversation
        response2 = agent.invoke(
            "Break it down by service", session_id=session_id
        )
        response2.print_reasoning()
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_alias_id: str | None = None,
        region: str | None = None,
        session: boto3.Session | None = None,
        enable_trace: bool = True,
    ) -> None:
        cfg = get_config()
        agent_cfg = cfg.bedrock_agents

        self.agent_id = agent_id or agent_cfg.agent_id
        self.agent_alias_id = agent_alias_id or agent_cfg.agent_alias_id
        self.region = region or cfg.bedrock.region
        self.enable_trace = enable_trace

        self._session = session or boto3.Session()
        self._client = self._session.client(
            "bedrock-agent-runtime", region_name=self.region
        )

        logger.info(
            "BedrockAgentClient initialised",
            agent_id=self.agent_id,
            agent_alias_id=self.agent_alias_id,
            region=self.region,
        )

    @staticmethod
    def new_session() -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def invoke(
        self,
        input_text: str,
        session_id: str | None = None,
        session_attributes: dict[str, str] | None = None,
        prompt_session_attributes: dict[str, str] | None = None,
        memory_id: str | None = None,
    ) -> AgentResponse:
        """
        Invoke the Bedrock Agent and collect the full response.

        Args:
            input_text: The user message to the agent.
            session_id: Session ID for multi-turn conversations.
                       Generate with BedrockAgentClient.new_session().
            session_attributes: Key-value pairs stored in the session context.
            prompt_session_attributes: Key-value pairs injected into prompts.
            memory_id: Memory ID for long-term agent memory.

        Returns:
            AgentResponse containing the final output and all traces.
        """
        sid = session_id or self.new_session()

        request: dict[str, Any] = {
            "agentId": self.agent_id,
            "agentAliasId": self.agent_alias_id,
            "sessionId": sid,
            "inputText": input_text,
            "enableTrace": self.enable_trace,
        }

        if session_attributes:
            request["sessionState"] = {
                "sessionAttributes": session_attributes
            }
        if prompt_session_attributes:
            request.setdefault("sessionState", {})
            request["sessionState"]["promptSessionAttributes"] = prompt_session_attributes
        if memory_id:
            request["memoryId"] = memory_id

        logger.info(
            "Invoking Bedrock Agent",
            agent_id=self.agent_id,
            session_id=sid,
            input_snippet=input_text[:80],
        )

        try:
            response = self._client.invoke_agent(**request)
        except ClientError as exc:
            logger.error("Agent invocation failed", error=str(exc), agent_id=self.agent_id)
            raise

        return self._process_event_stream(response["completion"], sid)

    def invoke_stream(
        self,
        input_text: str,
        session_id: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream the agent's output tokens as they arrive.

        Yields text chunks from the agent's final response.
        Traces are discarded in streaming mode for simplicity.
        """
        sid = session_id or self.new_session()

        response = self._client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=sid,
            inputText=input_text,
            enableTrace=False,
        )

        for event in response["completion"]:
            if "chunk" in event:
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    yield chunk_data["bytes"].decode("utf-8")

    def get_agent_memory(self, memory_id: str) -> list[dict[str, Any]]:
        """Retrieve stored agent memory for a given memory ID."""
        try:
            response = self._client.get_agent_memory(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                memoryType="SESSION_SUMMARY",
                memoryId=memory_id,
            )
            return response.get("memoryContents", [])
        except ClientError as exc:
            logger.error("Failed to retrieve agent memory", error=str(exc))
            raise

    def delete_agent_memory(self, memory_id: str) -> None:
        """Delete agent memory for a given memory ID."""
        self._client.delete_agent_memory(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            memoryId=memory_id,
        )
        logger.info("Agent memory deleted", memory_id=memory_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_event_stream(
        self, event_stream: Any, session_id: str
    ) -> AgentResponse:
        """Parse the Bedrock Agent event stream into an AgentResponse."""
        output_parts: list[str] = []
        traces: list[AgentTrace] = []
        files: list[dict[str, Any]] = []

        for event in event_stream:
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    output_parts.append(chunk["bytes"].decode("utf-8"))

            elif "trace" in event and self.enable_trace:
                trace = self._parse_trace_event(event["trace"])
                if trace:
                    traces.append(trace)

            elif "files" in event:
                files.extend(event["files"].get("files", []))

            elif "returnControl" in event:
                # Handle return-control for custom action groups
                logger.debug("Agent returned control", event=event["returnControl"])

        final_output = "".join(output_parts)

        logger.info(
            "Agent invocation complete",
            output_length=len(final_output),
            trace_count=len(traces),
            session_id=session_id,
        )

        return AgentResponse(
            output=final_output,
            session_id=session_id,
            traces=traces,
            files=files,
        )

    def _parse_trace_event(self, trace_event: dict[str, Any]) -> AgentTrace | None:
        """Extract structured information from a raw agent trace event."""
        trace = trace_event.get("trace", {})

        # Orchestration trace — contains reasoning and action steps
        if "orchestrationTrace" in trace:
            orch = trace["orchestrationTrace"]
            thought = orch.get("modelInvocationInput", {}).get("text", "")
            
            action: dict[str, Any] = {}
            if "invocationInput" in orch:
                inv = orch["invocationInput"]
                action = {
                    "type": inv.get("invocationType", ""),
                    "action_group": inv.get("actionGroupInvocationInput", {}).get("actionGroupName", ""),
                    "function": inv.get("actionGroupInvocationInput", {}).get("function", ""),
                    "parameters": inv.get("actionGroupInvocationInput", {}).get("parameters", []),
                    "knowledge_base_id": inv.get("knowledgeBaseLookupInput", {}).get("knowledgeBaseId", ""),
                }

            observation = ""
            if "observation" in orch:
                obs = orch["observation"]
                if "actionGroupInvocationOutput" in obs:
                    observation = obs["actionGroupInvocationOutput"].get("text", "")
                elif "knowledgeBaseLookupOutput" in obs:
                    refs = obs["knowledgeBaseLookupOutput"].get("retrievedReferences", [])
                    observation = f"Retrieved {len(refs)} knowledge base chunks"
                elif "finalResponse" in obs:
                    observation = obs["finalResponse"].get("text", "")

            return AgentTrace(
                trace_type="orchestration",
                thought=thought,
                action=action,
                observation=observation,
                raw=trace,
            )

        # Pre/post processing traces
        for trace_type in ("preProcessingTrace", "postProcessingTrace"):
            if trace_type in trace:
                return AgentTrace(
                    trace_type=trace_type.replace("Trace", "").lower(),
                    raw=trace,
                )

        # Failure trace
        if "failureTrace" in trace:
            failure_reason = trace["failureTrace"].get("failureReason", "Unknown failure")
            logger.error("Agent failure trace", reason=failure_reason)
            return AgentTrace(
                trace_type="failureTrace",
                thought=failure_reason,
                raw=trace,
            )

        return None
