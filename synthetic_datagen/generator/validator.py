"""
generator/validator.py
----------------------
Conversation Validator Agent — validates the final conversation artifact.

Validates against the FINAL conversation, not just the sampled chain.
May reject conversations and trigger regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of validating one conversation."""
    conversation_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


class ConversationValidator:
    """Validates generated conversations for structural and semantic correctness."""

    def validate(self, conversation: dict) -> ValidationResult:
        """
        Validate a conversation record.

        Checks:
          - has_messages: conversation has messages
          - has_tool_calls: at least one tool call present
          - multi_step: at least 3 tool calls (PDF requirement)
          - multi_tool: at least 2 distinct tools (PDF requirement)
          - chain_alignment: tool calls match sampled chain
          - metadata_complete: all required metadata fields present
          - tool_outputs_present: all tool calls have corresponding outputs
        """
        conv_id = conversation.get("metadata", {}).get("conversation_id", "unknown")
        result = ValidationResult(conversation_id=conv_id, passed=True)
        messages = conversation.get("messages", [])
        tool_calls = conversation.get("tool_calls", [])
        tool_outputs = conversation.get("tool_outputs", [])
        metadata = conversation.get("metadata", {})

        # Check: has messages
        result.checks["has_messages"] = len(messages) > 0
        if not result.checks["has_messages"]:
            result.errors.append("Conversation has no messages")

        # Check: has tool calls
        result.checks["has_tool_calls"] = len(tool_calls) > 0
        if not result.checks["has_tool_calls"]:
            result.errors.append("Conversation has no tool calls")

        # Short-mode conversations (1–2 tool calls, 2–3 turns) are exempt from
        # the multi_step and multi_tool hard checks. The 50–60% multi-step/multi-tool
        # target is met by the ~75% of conversations sampled in longer modes.
        sampling_mode = metadata.get("sampling_mode", "")
        is_short = sampling_mode == "short"

        # Check: multi_step (>= 3 tool calls) — skipped for short conversations
        if is_short:
            result.checks["multi_step"] = len(tool_calls) >= 1
            if not result.checks["multi_step"]:
                result.errors.append("Short conversation has no tool calls")
        else:
            result.checks["multi_step"] = len(tool_calls) >= 3
            if not result.checks["multi_step"]:
                result.errors.append(f"Only {len(tool_calls)} tool calls (minimum 3 required)")

        # Check: multi_tool (>= 2 distinct tools) — skipped for short conversations
        distinct_tools = set(tc.get("name", "").split("::")[0] for tc in tool_calls)
        if is_short:
            result.checks["multi_tool"] = len(distinct_tools) >= 1
        else:
            result.checks["multi_tool"] = len(distinct_tools) >= 2
            if not result.checks["multi_tool"]:
                result.errors.append(f"Only {len(distinct_tools)} distinct tools (minimum 2 required)")

        # Check: tool_outputs_present
        result.checks["tool_outputs_present"] = len(tool_outputs) == len(tool_calls)
        if not result.checks["tool_outputs_present"]:
            result.warnings.append(
                f"Tool outputs count ({len(tool_outputs)}) != tool calls count ({len(tool_calls)})"
            )

        # Check: metadata_complete
        required_meta = ["seed", "tool_ids_used", "num_turns", "num_clarification_questions",
                         "memory_grounding_rate", "corpus_memory_enabled"]
        missing_meta = [k for k in required_meta if k not in metadata]
        result.checks["metadata_complete"] = len(missing_meta) == 0
        if missing_meta:
            result.warnings.append(f"Missing metadata fields: {missing_meta}")

        # Check: chain alignment (tool call order matches sampled chain)
        sampled_chain = metadata.get("endpoint_ids", [])
        if sampled_chain and tool_calls:
            actual_endpoints = [tc.get("name", "") for tc in tool_calls]
            result.checks["chain_alignment"] = actual_endpoints == sampled_chain
            if not result.checks["chain_alignment"]:
                result.warnings.append("Tool call order does not match sampled chain")
        else:
            result.checks["chain_alignment"] = True  # can't check without chain

        # Overall pass/fail: hard errors cause failure
        hard_checks = ["has_messages", "has_tool_calls", "multi_step", "multi_tool"]
        result.passed = all(result.checks.get(c, True) for c in hard_checks)

        return result
