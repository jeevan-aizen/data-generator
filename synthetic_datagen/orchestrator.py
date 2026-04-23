"""
orchestrator.py
---------------
GenerationOrchestrator — coordinates all agents for synthetic conversation generation.

CHANGES FROM ORIGINAL:
  1. _generate_one_conversation() now calls assistant.decide_tool_call_arguments()
     for EVERY tool step BEFORE calling executor.execute_step(). The LLM reads
     the full conversation history (including prior tool outputs) and decides
     argument values — this is the core agentic behaviour that was missing.
     The LLM-decided args are merged with user_inputs and passed as Priority 1
     to the executor, so grounded values from the conversation always win.

  2. The planner's use_llm flag is now automatically set to True when an API key
     is present, so the LLM narrative backend is always used when available
     instead of requiring manual config file editing.

  3. Minor: a ConversationState.grounding_warnings attribute is handled safely
     with getattr() for backward compatibility.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Summary of a completed generation run."""
    generated: int
    rejected: int
    output_path: Path
    coverage_domains: dict[str, int]
    coverage_patterns: dict[str, int]


@dataclass
class ConversationResult:
    """Result of generating one conversation."""
    record: dict
    conversation_id: str
    succeeded: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class GenerationOrchestrator:
    """
    Coordinates all agents to produce a JSONL dataset of synthetic conversations.

    Usage:
        orchestrator = GenerationOrchestrator(
            registry=registry,
            projected=projected,
            config=config,
            output_path=Path("output/conversations.jsonl"),
            seed=42,
            corpus_memory_enabled=True,
        )
        result = orchestrator.run(n=20, mode="sequential")
    """

    def __init__(
        self,
        registry: Any,
        projected: Any,
        config: Any,
        output_path: Path,
        seed: int = 42,
        corpus_memory_enabled: bool = True,
        inline_judge: Any = None,
        inline_score_validator: Any = None,
        verbose: bool = False,
    ) -> None:
        from synthetic_datagen.sampler.sampler import SamplerAgent
        from synthetic_datagen.planner.agent import PlannerAgent as StructuredPlannerAgent
        from synthetic_datagen.planner.config import load_planner_config
        from synthetic_datagen.planner.narrative import (
            DeterministicNarrativeBackend,
            AnthropicNarrativeBackend,
        )
        from synthetic_datagen.planner.registry_adapter import build_planner_registry
        from synthetic_datagen.generator.user_proxy import UserProxyAgent
        from synthetic_datagen.generator.assistant import AssistantAgent
        from synthetic_datagen.generator.executor import OfflineExecutor
        from synthetic_datagen.generator.validator import ConversationValidator
        from synthetic_datagen.generator.writer import DatasetWriter
        from synthetic_datagen.memory.store import MemoryStore

        self.registry = registry
        self.projected = projected
        self.config = config
        self.output_path = output_path
        self.seed = seed
        self.corpus_memory_enabled = corpus_memory_enabled
        self.inline_judge = inline_judge
        self.inline_score_validator = inline_score_validator
        self.verbose = verbose

        # Initialize memory
        self.memory = MemoryStore(use_mem0=True)

        # Initialize sampler
        self.sampler = SamplerAgent(projected, registry, config)

        # Resolve API key once — shared by all LLM-backed components
        import os as _os
        from pathlib import Path as _Path
        _api_key = _os.environ.get("ANTHROPIC_API_KEY")
        if not _api_key:
            _env_path = _Path(__file__).parent.parent / ".env"
            if _env_path.exists():
                for _line in _env_path.read_text().splitlines():
                    _line = _line.strip()
                    if _line.startswith("ANTHROPIC_API_KEY="):
                        _api_key = _line.split("=", 1)[1].strip()
                        break

        # Initialize planner
        planner_config = load_planner_config()

        # FIX: automatically enable LLM backend when an API key is present.
        # The original code left use_llm=False (the config default), meaning
        # the DeterministicNarrativeBackend (template-based) was always used
        # even when a real LLM was available. Now we enable it automatically.
        if _api_key:
            planner_config.use_llm = True

        self.planner_registry = build_planner_registry(
            tool_registry=registry,
            user_natural_params=frozenset(config.user_natural_params),
        )

        if _api_key:
            llm_backend = AnthropicNarrativeBackend(api_key=_api_key)
            logger.info("[planner] using AnthropicNarrativeBackend (claude-haiku-4-5)")
        else:
            llm_backend = DeterministicNarrativeBackend()
            logger.info("[planner] ANTHROPIC_API_KEY not found — using DeterministicNarrativeBackend")

        self.planner = StructuredPlannerAgent(
            llm_backend=llm_backend,
            memory_store=self.memory,
            registry=self.planner_registry,
            config=planner_config,
        )

        # Initialize generator agents — all share the same resolved API key
        self.user_proxy = UserProxyAgent(registry, seed=seed, api_key=_api_key or None)
        self.assistant = AssistantAgent(registry, seed=seed, api_key=_api_key or None)
        self.executor = OfflineExecutor(registry, memory_store=self.memory, seed=seed)

        # Initialize validator and writer
        self.validator = ConversationValidator()
        self.writer = DatasetWriter(output_path)

        # Coverage tracker
        self._coverage: dict[str, dict[str, int]] = {"domains": {}, "patterns": {}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n: int, mode: str = "sequential") -> GenerationResult:
        """
        Generate n conversations and write them to output_path.
        Returns a GenerationResult summarising what was produced.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.unlink(missing_ok=True)
        self._coverage = {"domains": {}, "patterns": {}}

        generated = 0
        rejected = 0
        conv_index = 0

        try:
            chains = self.sampler.sample_mixed(n=n * 2, seed=self.seed)
        except Exception as e:
            logger.warning("sample_mixed failed (%s), falling back to sample_chains", e)
            chains = self.sampler.sample_chains(n=n, mode=mode, seed=self.seed)

        for chain in chains:
            if generated >= n:
                break

            conv_index += 1
            conversation_id = f"conv_{self.seed}_{conv_index:04d}"

            result = self._run_one(conversation_id, chain)

            if not result.succeeded:
                rejected += 1
                if self.verbose:
                    logger.info("Rejected %s: %s", conversation_id, result.error)
                continue

            record = result.record

            # Structural validation
            validation = self.validator.validate(record)
            if not validation.passed:
                rejected += 1
                if self.verbose:
                    logger.info("Validation failed %s: %s", conversation_id, validation.errors)
                continue

            # Optional inline judge scoring
            if self.inline_judge is not None:
                record = self._score_inline(record, conversation_id)

            self.writer.write(record)
            generated += 1

            # Update coverage
            domain = record["metadata"].get("domain", "General")
            pattern = record["metadata"].get("pattern_type", "unknown")
            self._coverage["domains"][domain] = self._coverage["domains"].get(domain, 0) + 1
            self._coverage["patterns"][pattern] = self._coverage["patterns"].get(pattern, 0) + 1

            # Update corpus memory for cross-conversation steering
            if self.corpus_memory_enabled:
                self._update_corpus_memory(chain, conversation_id, domain)

            if generated % 10 == 0:
                logger.info("%d/%d generated...", generated, n)

        return GenerationResult(
            generated=generated,
            rejected=rejected,
            output_path=self.output_path,
            coverage_domains=dict(self._coverage["domains"]),
            coverage_patterns=dict(self._coverage["patterns"]),
        )

    @property
    def coverage(self) -> dict[str, dict[str, int]]:
        """Domain and pattern coverage from the last run."""
        return self._coverage

    # ------------------------------------------------------------------
    # Internal: one conversation
    # ------------------------------------------------------------------

    def _run_one(self, conversation_id: str, chain: Any) -> ConversationResult:
        """Generate one complete conversation. Returns ConversationResult."""
        try:
            record = self._generate_one_conversation(conversation_id, chain)
            return ConversationResult(
                record=record,
                conversation_id=conversation_id,
                succeeded=True,
            )
        except Exception as e:
            return ConversationResult(
                record={},
                conversation_id=conversation_id,
                succeeded=False,
                error=str(e),
            )

    def _generate_one_conversation(self, conversation_id: str, chain: Any) -> dict:
        """
        Full single-conversation generation pipeline.

        AGENTIC LOOP (key change from original):
          For each tool step, the AssistantAgent now calls
          decide_tool_call_arguments() BEFORE the executor runs.
          This method uses Claude's native function calling (structured output)
          to read the full conversation history — including prior tool outputs —
          and decide what argument values to use.

          The LLM-decided arguments are merged with user_inputs so they take
          Priority 1 in the executor's 4-step resolution policy. This means:
            - IDs returned by step N (e.g. hotel_id="H345") are automatically
              used in step N+1's arguments — genuine grounded chaining.
            - The assistant *decides* what to call, not just executes a script.

        Flow: adapt chain → plan → clarify → [for each step:
              LLM decides args → executor executes → emit tool call] → final response
        """
        from synthetic_datagen.planner.registry_adapter import (
            adapt_sampled_chain,
            clarification_points_to_steps,
        )
        from synthetic_datagen.generator.writer import DatasetWriter
        from synthetic_datagen.common.types import ConversationState

        # Adapt SampledChain → SampledToolChain for the structured planner
        adapted_chain = adapt_sampled_chain(
            chain=chain,
            chain_id=conversation_id,
            seed=self.seed if self.seed is not None else 0,
        )

        # Plan: LLM generates narrative fields (domain, user_goal, step purposes)
        plan_result = self.planner.plan(adapted_chain, plan_id=conversation_id)
        if not plan_result.success:
            raise RuntimeError(
                f"Planner failed [{plan_result.error_code}]: {plan_result.error_message}"
            )
        plan = plan_result.plan

        # Bridge clarification points → ClarificationStep objects
        all_clarif = clarification_points_to_steps(plan.clarification_points)

        # Session and shared state
        session = self.executor.create_session(conversation_id)
        state = ConversationState(conversation_id=conversation_id, plan=plan, session=session)

        messages: list[dict] = []
        tool_calls_log: list[dict] = []
        tool_outputs_log: list[dict] = []
        clarification_count = 0
        grounded_steps = 0
        non_first_steps = 0
        persistent_user_inputs: dict = {}

        # Opening user message
        user_turn = self.user_proxy.generate_initial_request(plan)
        messages.append({"role": "user", "content": user_turn.content})
        # Seed executor with any params the user stated in their opening message
        persistent_user_inputs.update(user_turn.resolved_params)

        def _step_purpose(step_idx: int) -> str | None:
            return next((s.purpose for s in plan.steps if s.step_index == step_idx), None)

        # Step-0 clarification (before any tool calls)
        step0_clarifs = [cs for cs in all_clarif if cs.step_index == 0]
        if step0_clarifs:
            ast_turn = self.assistant.ask_clarification(
                step0_clarifs[0], step_purpose=_step_purpose(0)
            )
            messages.append({"role": "assistant", "content": ast_turn.content})
            clarification_count += 1

            user_resp = self.user_proxy.answer_clarification(step0_clarifs[0], plan)
            messages.append({"role": "user", "content": user_resp.content})
            persistent_user_inputs.update(user_resp.resolved_params)

        user_natural_params = self.config.user_natural_params or set()

        # ------------------------------------------------------------------
        # Main tool execution loop — AGENTIC VERSION
        # ------------------------------------------------------------------
        for step_idx, endpoint_id in enumerate(chain.endpoint_ids):

            # Mid-chain clarification
            mid_clarifs = [cs for cs in all_clarif
                           if cs.step_index == step_idx and step_idx > 0]
            if mid_clarifs:
                cs = mid_clarifs[0]
                already_resolved = (
                    all(
                        p in persistent_user_inputs or p in session.accumulated_fields
                        for p in (cs.missing_params or [])
                    )
                    if cs.missing_params else False
                )
                if not already_resolved:
                    ast_turn = self.assistant.ask_clarification(
                        cs, step_purpose=_step_purpose(step_idx)
                    )
                    messages.append({"role": "assistant", "content": ast_turn.content})
                    clarification_count += 1

                    user_resp = self.user_proxy.answer_clarification(cs, plan)
                    messages.append({"role": "user", "content": user_resp.content})
                    persistent_user_inputs.update(user_resp.resolved_params)

            transition = (
                chain.transitions[step_idx - 1]
                if step_idx > 0 and step_idx - 1 < len(chain.transitions)
                else None
            )

            if step_idx > 0:
                non_first_steps += 1

            # ----------------------------------------------------------
            # AGENTIC STEP: LLM decides arguments from conversation history
            # ----------------------------------------------------------
            # The AssistantAgent reads the full conversation (including prior
            # tool outputs) using Claude's native function calling API and
            # returns a dict of argument values it decided to use.
            #
            # These LLM-decided args are merged into user_inputs so they take
            # Priority 1 in the executor's resolution policy, meaning:
            #   - hotel_id returned in step 0 output → used in step 1 booking call
            #   - flight_id, order_id, etc. all chain correctly
            # This is the core agentic behaviour: the LLM is making decisions
            # based on what it observed, not following a predetermined script.
            llm_decided_args = self.assistant.decide_tool_call_arguments(
                endpoint_id=endpoint_id,
                conversation_history=messages,
            )

            # Merge: LLM args override generic user_inputs for this step only
            # (don't permanently update persistent_user_inputs — each step's
            # LLM decision is specific to that step's context)
            step_user_inputs = {**persistent_user_inputs, **llm_decided_args}

            step_out = self.executor.execute_step(
                endpoint_id=endpoint_id,
                user_inputs=step_user_inputs,
                session=session,
                transition=transition,
                step_index=step_idx,
            )

            if step_idx > 0 and step_out.was_grounded:
                grounded_steps += 1
                state.grounded_steps += 1
            if step_idx > 0:
                state.non_first_steps += 1

            # Inline grounding check for next step
            self._check_next_step_grounding(
                chain, step_idx, session, user_natural_params, state
            )

            # Assistant emits tool call turn
            plan_step = next((s for s in plan.steps if s.step_index == step_idx), None)
            preamble = None
            if (plan_step and plan_step.assistant_intent
                    and "calls the appropriate endpoint" not in plan_step.assistant_intent
                    and "take care of the" not in plan_step.assistant_intent.lower()):
                preamble = plan_step.assistant_intent

            ast_tool_turn = self.assistant.emit_tool_call(
                endpoint_id, step_out.arguments, preamble=preamble
            )
            messages.append({
                "role": "assistant",
                "content": ast_tool_turn.content,
                "tool_calls": ast_tool_turn.tool_calls,
            })
            messages.append({
                "role": "tool",
                "name": endpoint_id,
                "content": json.dumps(step_out.output),
            })
            tool_calls_log.append({"name": endpoint_id, "parameters": step_out.arguments})
            tool_outputs_log.append({"name": endpoint_id, "output": step_out.output})

        # Final response — LLM synthesizes all tool outputs into a natural reply
        final = self.assistant.generate_final_response(plan, session.steps)
        messages.append({"role": "assistant", "content": final.content})

        # Surface grounding warnings
        for w in getattr(state, "grounding_warnings", []):
            logger.warning("[grounding] %s: %s", conversation_id, w)

        memory_grounding_rate = (
            grounded_steps / non_first_steps if non_first_steps > 0 else None
        )

        return DatasetWriter.build_record(
            conversation_id=conversation_id,
            messages=messages,
            tool_calls=tool_calls_log,
            tool_outputs=tool_outputs_log,
            chain=chain,
            domain=plan.domain,
            memory_grounding_rate=memory_grounding_rate,
            corpus_memory_enabled=self.corpus_memory_enabled,
            seed=self.seed,
            num_clarification_questions=clarification_count,
        )

    def _check_next_step_grounding(
        self,
        chain: Any,
        step_idx: int,
        session: Any,
        user_natural_params: set,
        state: Any,
    ) -> None:
        """Warn if the next step has required params that won't be resolvable."""
        next_idx = step_idx + 1
        if next_idx >= len(chain.endpoint_ids):
            return

        next_ep = self.executor.registry.get_endpoint(chain.endpoint_ids[next_idx])
        if not next_ep:
            return

        accumulated = set(session.accumulated_fields.keys())
        next_transition = (
            chain.transitions[step_idx] if step_idx < len(chain.transitions) else None
        )
        mapped_params = (
            {fm.target_param for fm in next_transition.field_mappings}
            if next_transition else set()
        )
        clarified_params: set[str] = set()
        for cs in chain.clarification_steps:
            if cs.step_index == next_idx:
                clarified_params.update(cs.missing_params)

        missing = [
            p.name for p in next_ep.parameters
            if p.required
            and p.name not in accumulated
            and p.name not in mapped_params
            and p.name.lower() not in user_natural_params
            and p.name not in clarified_params
        ]
        if missing:
            if not hasattr(state, "grounding_warnings"):
                state.grounding_warnings = []
            state.grounding_warnings.append(
                f"step {next_idx} ({chain.endpoint_ids[next_idx]}) has unresolvable "
                f"required params: {missing} — will use mock fallback"
            )

    def _score_inline(self, record: dict, conversation_id: str) -> dict:
        """Score a record with the inline judge and attach scores."""
        from synthetic_datagen.evaluator.scorer import attach_scores
        try:
            raw_result = self.inline_judge.score(record)
            scores = self.inline_score_validator.validate(raw_result)
            return attach_scores(record, scores)
        except Exception as e:
            if self.verbose:
                logger.warning("Judge error on %s: %s", conversation_id, e)
            return record

    def _update_corpus_memory(self, chain: Any, conversation_id: str, domain: str) -> None:
        """Write a conversation summary to corpus memory for cross-conversation steering."""
        self.memory.add(
            content=(
                f"Tools: {', '.join(chain.tool_ids)}. "
                f"Domain: {domain}. "
                f"Pattern: {chain.pattern_type}."
            ),
            scope="corpus",
            metadata={
                "conversation_id": conversation_id,
                "tools": chain.tool_ids,
                "pattern_type": chain.pattern_type,
            },
        )

    # ------------------------------------------------------------------
    # Coverage reporting
    # ------------------------------------------------------------------

    def print_coverage_report(self) -> None:
        """Print domain and pattern coverage from the last run."""
        domains = self._coverage["domains"]
        patterns = self._coverage["patterns"]

        print(f"\n[coverage] Domain distribution ({len(domains)} domains):")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"  {domain:<30} {count:>3}  {bar}")

        print(f"\n[coverage] Pattern distribution:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"  {pattern:<30} {count:>3}  {bar}")

        underrepresented = [d for d, c in domains.items() if c == 1]
        if underrepresented:
            print(f"\n[coverage] Underrepresented domains (1 conversation each): {underrepresented}")