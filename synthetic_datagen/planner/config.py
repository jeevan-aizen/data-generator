"""
planner/config.py
-----------------
PlannerConfig dataclass and YAML-backed loader.

CHANGES FROM ORIGINAL:
  - use_llm default changed from False → True in both _DEFAULTS and PlannerConfig.
    The original False default meant the DeterministicNarrativeBackend (template-
    based stub) was always used even when an API key was present, because the
    orchestrator respected this flag. Now the LLM is used whenever an API key
    is available, which is the intended production behaviour.
  - All other fields unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    # Retry policy
    "max_retries": 3,

    # Chain quality constraints
    "min_steps": 1,
    "min_distinct_tools": 1,
    "allow_parallel_patterns": True,

    # Validation strictness
    "strict_validation": False,
    "use_registry_metadata": True,

    # Registry and memory strictness
    "registry_strict_mode": False,
    "memory_strict_mode": False,

    # Corpus memory query
    "corpus_query_term": "tool conversation",
    "corpus_query_limit": 10,

    # Seed
    "seed": None,

    # LLM narrative mode
    # FIX: changed from False → True so LLM is used when an API key is present.
    # The orchestrator also sets this at runtime when it detects an API key, but
    # setting the default here ensures it works even when the config file is used
    # directly without going through the orchestrator.
    "use_llm": True,
}

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "planner_config.yaml"


# ---------------------------------------------------------------------------
# PlannerConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    """
    All planner behavior parameters.

    Loaded from config/planner_config.yaml with in-code defaults as fallback.
    Every field has a default so the system works with zero config files.

    use_llm:
        True  → use the real LLM backend passed to PlannerAgent.__init__()
        False → use DeterministicNarrativeBackend (offline, no LLM dependency)
        Default is now True. Set to False explicitly for offline/test environments.

    Retry policy:
        On PlannerOutputValidationError or narrative parse error:
            retry up to max_retries, then raise PlannerRetryExhaustedError
        On PlannerInvalidInputError or PlannerChainTooShortError:
            fail immediately (resample at orchestrator level)
        On PlannerConfigError or PlannerDependencyError:
            hard stop (orchestrator must not retry)
    """

    # Retry policy
    max_retries: int = 3

    # Chain quality constraints
    min_steps: int = 1
    min_distinct_tools: int = 1
    allow_parallel_patterns: bool = True

    # Validation strictness
    strict_validation: bool = False
    use_registry_metadata: bool = True

    # Registry and memory strictness
    registry_strict_mode: bool = False
    memory_strict_mode: bool = False

    # Corpus memory query
    corpus_query_term: str = "tool conversation"
    corpus_query_limit: int = 10

    # Seed
    seed: int | None = None

    # FIX: default changed from False → True
    use_llm: bool = True

    def validate(self) -> None:
        """Validate config values. Raises ValueError on invalid config."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {self.min_steps}")
        if self.min_distinct_tools < 1:
            raise ValueError(
                f"min_distinct_tools must be >= 1, got {self.min_distinct_tools}"
            )
        if self.corpus_query_limit < 1:
            raise ValueError(
                f"corpus_query_limit must be >= 1, got {self.corpus_query_limit}"
            )
        if not self.corpus_query_term or not self.corpus_query_term.strip():
            raise ValueError("corpus_query_term must not be empty.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int or None, got {type(self.seed).__name__}")

    def __repr__(self) -> str:
        return (
            f"PlannerConfig("
            f"max_retries={self.max_retries}, "
            f"min_steps={self.min_steps}, "
            f"min_tools={self.min_distinct_tools}, "
            f"strict={self.strict_validation}, "
            f"registry_strict={self.registry_strict_mode}, "
            f"memory_strict={self.memory_strict_mode}, "
            f"use_llm={self.use_llm}, "
            f"seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_planner_config(config_path: Path | str | None = None) -> PlannerConfig:
    """
    Load PlannerConfig from YAML.

    Loading pattern:
      1. Start with in-code defaults (use_llm=True)
      2. Load YAML if file exists
      3. Merge YAML over defaults (YAML wins, defaults fill missing keys)
      4. Validate the resolved config
      5. Return typed PlannerConfig

    Returns default config if YAML is missing — never raises on missing file.
    """
    resolved = _DEFAULTS.copy()

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            if isinstance(yaml_data, dict):
                for key in _DEFAULTS:
                    if key in yaml_data:
                        resolved[key] = yaml_data[key]
        except Exception as e:
            print(f"[planner_config] Warning: could not load {path} ({e}), using defaults")
    else:
        print(f"[planner_config] Config not found at {path}, using defaults")

    config = PlannerConfig(
        max_retries=int(resolved["max_retries"]),
        min_steps=int(resolved["min_steps"]),
        min_distinct_tools=int(resolved["min_distinct_tools"]),
        allow_parallel_patterns=bool(resolved["allow_parallel_patterns"]),
        strict_validation=bool(resolved["strict_validation"]),
        use_registry_metadata=bool(resolved["use_registry_metadata"]),
        registry_strict_mode=bool(resolved["registry_strict_mode"]),
        memory_strict_mode=bool(resolved["memory_strict_mode"]),
        corpus_query_term=str(resolved["corpus_query_term"]),
        corpus_query_limit=int(resolved["corpus_query_limit"]),
        seed=int(resolved["seed"]) if resolved["seed"] is not None else None,
        use_llm=bool(resolved["use_llm"]),
    )

    config.validate()
    return config


# ---------------------------------------------------------------------------
# Canonical planner_config.yaml template
# ---------------------------------------------------------------------------
# Drop this file at config/planner_config.yaml to override any default.
#
# planner_config.yaml
# -------------------
# max_retries: 3
# min_steps: 1
# min_distinct_tools: 1
# allow_parallel_patterns: true
# strict_validation: false
# use_registry_metadata: true
# registry_strict_mode: false
# memory_strict_mode: false
# corpus_query_term: "tool conversation"
# corpus_query_limit: 10
# seed: null
# use_llm: true   # set to false to force DeterministicNarrativeBackend (offline/test)