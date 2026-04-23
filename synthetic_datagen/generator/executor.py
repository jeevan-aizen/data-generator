"""
generator/executor.py
---------------------
Offline Executor — required by the PDF as a first-class component.

Responsibilities:
  - Validate arguments against endpoint schema
  - Emit deterministic mock outputs consistent with response schema
  - Maintain lightweight session state so later calls reference earlier outputs
  - Resolve argument values using the 4-step precedence policy

Value resolution precedence (at executor time only):
  1. Explicit user input from conversation
  2. Transition.field_mappings from prior step output
  3. Session memory retrieval
  4. Default/mock fallback

No concrete values are pre-filled by Sampler or Planner.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from synthetic_datagen.common.types import Transition, FieldMapping
from synthetic_datagen.graph.registry import ToolRegistry, Endpoint, NormalizedParameter
from synthetic_datagen.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """Record of one tool call's inputs and outputs."""
    step_index: int
    endpoint_id: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    was_grounded: bool = False  # True if session memory was used for args


@dataclass
class SessionState:
    """Lightweight session state for one conversation."""
    conversation_id: str
    steps: list[StepOutput] = field(default_factory=list)
    accumulated_fields: dict[str, Any] = field(default_factory=dict)

    def record_step(self, step: StepOutput) -> None:
        self.steps.append(step)
        # Accumulate all output fields for downstream use
        self._flatten_into(step.output, self.accumulated_fields)

    def _flatten_into(self, obj: Any, target: dict, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                target[k] = v
                if isinstance(v, (dict, list)):
                    self._flatten_into(v, target, k)
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            self._flatten_into(obj[0], target, prefix)

    def get_field(self, field_name: str) -> Any | None:
        return self.accumulated_fields.get(field_name)


# ---------------------------------------------------------------------------
# Mock value generators
# ---------------------------------------------------------------------------

_MOCK_VALUES: dict[str, Any] = {
    "string":  "mock_value",
    "integer": 1,
    "number":  1.0,
    "boolean": True,
    "array":   [],
    "object":  {},
    "unknown": "mock_value",
}

_FIELD_MOCK_POOLS: dict[str, list] = {
    # Entity IDs returned by search/retrieve steps
    "flight_id":        ["FL101", "FL247", "FL389", "FL512", "FL634", "FL778", "FL890"],
    "hotel_id":         ["H201", "H345", "H467", "H589", "H612", "H734", "H856"],
    "product_id":       ["P102", "P235", "P371", "P448", "P562", "P689", "P714"],
    "recipe_id":        ["RC201", "RC342", "RC456", "RC578", "RC612"],
    "event_id":         ["EVT301", "EVT422", "EVT543", "EVT667", "EVT789"],
    "job_id":           ["J401", "J523", "J645", "J768", "J891"],
    "restaurant_id":    ["R501", "R623", "R745", "R867", "R912"],
    "article_id":       ["N601", "N724", "N845", "N967", "N112"],
    "booking_id":       ["BK701", "BK823", "BK945", "BK167", "BK289"],
    "reservation_id":   ["RES801", "RES923", "RES145", "RES267", "RES389"],
    "order_id":         ["ORD901", "ORD123", "ORD245", "ORD367", "ORD489"],
    "message_id":       ["MSG101", "MSG223", "MSG345", "MSG467", "MSG589"],
    "ticket_id":        ["TKT201", "TKT345", "TKT467", "TKT589", "TKT612"],
    "confirmation":     ["CONF2341", "CONF5678", "CONF8912", "CONF3456", "CONF7890"],
    "confirmation_code":["CONF2341", "CONF5678", "CONF8912", "CONF3456", "CONF7890"],
    # User-natural params — realistic values so mock fallback is never "mock_value"
    "origin":           ["JFK", "LAX", "ORD", "DFW", "SFO", "MIA", "SEA", "BOS"],
    "destination":      ["CDG", "LHR", "NRT", "SYD", "DXB", "SIN", "AMS", "FCO"],
    "city":             ["Paris", "London", "Tokyo", "New York", "Sydney", "Berlin", "Rome", "Barcelona"],
    "location":         ["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston", "Miami"],
    "from_currency":    ["USD", "EUR", "GBP", "JPY", "CAD"],
    "to_currency":      ["EUR", "GBP", "JPY", "AUD", "CHF"],
    "currency":         ["USD", "EUR", "GBP"],
    "symbol":           ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"],
    "query":            ["available options", "top results", "recommended picks", "popular choices"],
    "keyword":          ["travel deals", "budget options", "top picks", "popular choices"],
    "date":             ["2024-06-15", "2024-07-20", "2024-08-05", "2024-09-12", "2024-10-18"],
    "departure_date":   ["2024-06-15", "2024-07-20", "2024-08-05", "2024-09-12", "2024-10-18"],
    "return_date":      ["2024-06-22", "2024-07-27", "2024-08-12", "2024-09-19", "2024-10-25"],
    "check_in":         ["2024-06-15", "2024-07-20", "2024-08-05", "2024-09-12", "2024-10-18"],
    "check_out":        ["2024-06-18", "2024-07-24", "2024-08-09", "2024-09-16", "2024-10-22"],
    "start_date":       ["2024-06-15", "2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01"],
    "end_date":         ["2024-06-30", "2024-07-31", "2024-08-31", "2024-09-30", "2024-10-31"],
    "from_date":        ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
    "to_date":          ["2024-06-15", "2024-07-15", "2024-08-15", "2024-09-15", "2024-10-15"],
    "start_datetime":   ["2024-06-15T10:00:00", "2024-07-20T14:00:00", "2024-08-05T09:00:00"],
    "end_datetime":     ["2024-06-15T11:00:00", "2024-07-20T15:00:00", "2024-08-05T10:00:00"],
    "time":             ["09:00", "12:00", "14:30", "17:00", "19:30"],
    "language":         ["en", "fr", "de", "es", "ja", "pt"],
    "target_language":  ["fr", "de", "es", "ja", "pt", "zh"],
    "source_language":  ["en", "fr", "de", "es"],
    "country":          ["us", "gb", "fr", "de", "jp", "au", "ca"],
    "address":          ["123 Main St, New York, NY", "456 Oak Ave, Chicago, IL", "789 Pine Rd, Los Angeles, CA"],
    "email":            ["alice@example.com", "bob@example.com", "carol@example.com", "david@example.com"],
    "buyer_email":      ["alice@example.com", "bob@example.com", "carol@example.com", "david@example.com"],
    "passenger_email":  ["alice@example.com", "bob@example.com", "carol@example.com", "david@example.com"],
    "name":             ["Alice Chen", "Bob Martinez", "Carol Johnson", "David Kim", "Emma Wilson"],
    "guest_name":       ["Alice Chen", "Bob Martinez", "Carol Johnson", "David Kim", "Emma Wilson"],
    "passenger_name":   ["Alice Chen", "Bob Martinez", "Carol Johnson", "David Kim", "Emma Wilson"],
    "job_title":        ["Software Engineer", "Product Manager", "Data Scientist", "UX Designer", "Marketing Manager"],
    "title":            ["Team Meeting", "Project Kickoff", "Client Call", "Weekly Sync", "Planning Session"],
    "topic":            ["technology", "finance", "health", "travel", "sports", "business"],
    "subject":          ["Meeting Request", "Follow-up", "Introduction", "Project Update", "Schedule Confirmation"],
    "message":          ["I would like to schedule a meeting.", "Please confirm your availability.", "Looking forward to connecting."],
    "preferences":      ["budget-friendly", "central location", "free cancellation", "family-friendly", "highly rated"],
    "updated_fields":   [["preferences"], ["settings"], ["profile"], ["language"]],
    "text":             ["Hello, how are you?", "I need help with my order.", "Can you assist me?", "Please confirm my booking."],
    "category":         ["general", "business", "sports", "technology", "entertainment", "science"],
    "type":             ["standard", "premium", "economy", "business", "first-class"],
    "ticket_type":      ["general", "vip", "student", "group", "early-bird"],
    "amount":           [50.0, 75.0, 100.0, 120.0, 200.0, 350.0, 500.0],
    "quantity":         [1, 2, 3, 5],
    "party_size":       [1, 2, 3, 4, 6],
    "passengers":       [1, 2, 3, 4],
}


def _mock_value_for_param(param: NormalizedParameter, rng: random.Random | None = None) -> Any:
    """Generate a randomized mock value for a parameter."""
    name_lower = param.name.lower()

    # Pick randomly from pool if available
    if name_lower in _FIELD_MOCK_POOLS:
        pool = _FIELD_MOCK_POOLS[name_lower]
        return (rng or random).choice(pool)

    # Check enum
    if param.enum:
        return (rng or random).choice(param.enum)

    # Check default
    if param.default is not None:
        return param.default

    # Fall back to type-based mock
    return _MOCK_VALUES.get(param.type, "mock_value")


def _generate_mock_output(endpoint: Endpoint, arguments: dict, rng: random.Random) -> dict:
    """
    Generate a deterministic mock output consistent with the endpoint's response schema.

    Uses a deterministic sub-RNG seeded by endpoint_id + key argument values so the
    same call always produces the same output — fixing the inconsistency where
    hotel_id=H345 returned different hotel names on different calls.
    """
    schema = endpoint.returns_schema

    if not schema:
        # Return a minimal but non-empty output with the endpoint name so the
        # judge can see what was called rather than a generic "mock_result"
        ep_label = endpoint.endpoint_id.split("::")[-1].replace("_", " ")
        return {"status": "ok", "endpoint": ep_label, "completed": True}

    # Seed a deterministic sub-RNG from (endpoint_id, sorted argument values)
    # so the same inputs always produce the same mock output
    arg_fingerprint = "|".join(
        f"{k}={v}" for k, v in sorted(arguments.items())
        if isinstance(v, (str, int, float, bool))
    )
    seed_str = f"{endpoint.endpoint_id}:{arg_fingerprint}"
    deterministic_rng = random.Random(hash(seed_str) & 0xFFFFFFFF)

    result = _fill_schema(schema, arguments, endpoint.endpoint_id, deterministic_rng)
    return result


_PRODUCT_NAMES = [
    "Wireless Headphones Pro", "Smart Watch Series 5", "Running Shoes Ultra",
    "Laptop Stand Adjustable", "Portable Bluetooth Speaker", "USB-C Hub 7-in-1",
    "Mechanical Keyboard RGB", "Gaming Mouse Wireless", "Noise-Cancelling Earbuds",
]
_RECIPE_NAMES = [
    "Classic Spaghetti Carbonara", "Chicken Tikka Masala", "Beef Tacos",
    "Caesar Salad", "Vegetable Stir Fry", "Mushroom Risotto", "Lemon Herb Salmon",
]
_HOTEL_NAMES = [
    "The Grand Hotel", "City Suites Downtown", "Luxury Inn & Spa",
    "Boutique Hotel Central", "Harbor View Resort", "The Royal Palace Hotel",
]
_RESTAURANT_NAMES = [
    "La Maison", "Bella Italia", "The Garden Grill", "Saffron Kitchen",
    "The Rustic Table", "Blue Harbor Bistro", "Le Petit Café",
]
_JOB_TITLES = [
    "Senior Software Engineer", "Software Engineer II", "ML Engineer",
    "Data Scientist", "Product Manager", "Full Stack Developer",
    "DevOps Engineer", "Backend Engineer", "Frontend Engineer",
]
_TECH_COMPANIES = [
    "Google", "Amazon", "Apple", "Microsoft", "Meta", "Stripe",
    "Airbnb", "Lyft", "Salesforce", "Adobe", "Nvidia", "OpenAI",
]
_EVENT_TITLES = [
    "Tech Summit 2024", "Developer Conference", "Product Launch Event",
    "Industry Forum 2024", "Annual Meetup", "Innovation Expo",
]


_LOCATION_ARG_KEYS = ("city", "location", "destination", "origin", "address", "place", "region", "area")
_STREET_NAMES = ["Main St", "Oak Ave", "Park Blvd", "Market St", "Broadway", "5th Ave", "Elm St", "Maple Dr"]


def _resolve_location(arguments: dict) -> str | None:
    """Extract the most relevant location string from arguments."""
    for key in _LOCATION_ARG_KEYS:
        val = arguments.get(key)
        if val and isinstance(val, str) and val.lower() not in ("mock_value", "string"):
            return val
    return None


def _fill_schema(schema: Any, arguments: dict, endpoint_id: str, rng: random.Random) -> Any:
    """Recursively fill a schema template with mock values."""
    ep_lower = endpoint_id.lower()

    def _context_name() -> str:
        """Return a name appropriate for the endpoint's domain.
        Ordered most-specific first to avoid false matches.
        """
        if "hotel" in ep_lower:
            return rng.choice(_HOTEL_NAMES)
        if "restaurant" in ep_lower:
            return rng.choice(_RESTAURANT_NAMES)
        if "recipe" in ep_lower or "food_" in ep_lower:
            return rng.choice(_RECIPE_NAMES)
        if any(kw in ep_lower for kw in ("product", "ecommerce", "shop", "item", "cart")):
            return rng.choice(_PRODUCT_NAMES)
        if any(kw in ep_lower for kw in ("job", "career", "employ")):
            return rng.choice(_JOB_TITLES)
        if any(kw in ep_lower for kw in ("event", "ticket", "concert")):
            return rng.choice(_EVENT_TITLES)
        return rng.choice(_FIELD_MOCK_POOLS["name"])

    def _fill_value(key: str, value: Any) -> Any:
        """Fill a single key-value pair using context-aware logic."""
        # Priority 1: exact match in arguments
        if key in arguments and arguments[key] not in ("mock_value", "string", None):
            return arguments[key]

        # Priority 2: location-aware filling — use the user's stated location
        # so output addresses/cities match what the user actually asked for
        if key == "address":
            loc = _resolve_location(arguments)
            if loc:
                street_num = rng.randint(100, 999)
                return f"{street_num} {rng.choice(_STREET_NAMES)}, {loc}"
            return rng.choice(_FIELD_MOCK_POOLS["address"])

        if key == "city":
            loc = _resolve_location(arguments)
            return loc if loc else rng.choice(_FIELD_MOCK_POOLS["city"])

        if key == "location":
            loc = _resolve_location(arguments)
            return loc if loc else rng.choice(_FIELD_MOCK_POOLS["location"])

        # Priority 3: context-aware title — endpoint type determines what "title" means
        if key == "title":
            if any(kw in ep_lower for kw in ("job", "career", "employ")):
                return rng.choice(_JOB_TITLES)
            if any(kw in ep_lower for kw in ("event", "ticket", "concert", "show")):
                return rng.choice(_EVENT_TITLES)
            if any(kw in ep_lower for kw in ("calendar", "meeting", "schedule")):
                return rng.choice(_FIELD_MOCK_POOLS["title"])
            # Default: avoid meeting-specific names for non-meeting endpoints
            return rng.choice(_JOB_TITLES + _EVENT_TITLES)

        # Priority 3b: context-aware company — tech companies for job endpoints
        if key == "company":
            if any(kw in ep_lower for kw in ("job", "career", "employ")):
                return rng.choice(_TECH_COMPANIES)
            return rng.choice(_FIELD_MOCK_POOLS.get("company", _TECH_COMPANIES))

        # Priority 4: context-aware name
        if key == "name":
            return _context_name()

        # Priority 5: pool lookup
        if key in _FIELD_MOCK_POOLS:
            return rng.choice(_FIELD_MOCK_POOLS[key])

        return None  # caller uses original value as fallback

    if isinstance(schema, dict):
        filled: dict = {}
        for key, value in schema.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Placeholder — fill using context-aware logic
                result = _fill_value(key, value)
                filled[key] = result if result is not None else f"mock_{key}"
            elif isinstance(value, list):
                # For "*_fields" / "*_keys" outputs that list what was updated,
                # reflect the actual argument names so the output is coherent
                # (e.g. update_preferences → updated_fields: ["preferences"]).
                if key.endswith("_fields") or key.endswith("_keys"):
                    param_names = [k for k in arguments if k not in ("status", "ok")][:3]
                    filled[key] = param_names if param_names else [_fill_schema(item, arguments, endpoint_id, rng) for item in value[:1]]
                elif key in _FIELD_MOCK_POOLS:
                    filled[key] = rng.choice(_FIELD_MOCK_POOLS[key])
                else:
                    filled[key] = [_fill_schema(item, arguments, endpoint_id, rng) for item in value[:1]]
            elif isinstance(value, dict):
                filled[key] = _fill_schema(value, arguments, endpoint_id, rng)
            elif isinstance(value, (int, float)):
                if key in arguments and isinstance(arguments[key], (int, float)):
                    filled[key] = arguments[key]
                else:
                    filled[key] = value
            elif isinstance(value, str):
                result = _fill_value(key, value)
                filled[key] = result if result is not None else value
            else:
                filled[key] = value
        return filled
    elif isinstance(schema, list):
        return [_fill_schema(item, arguments, endpoint_id, rng) for item in schema[:1]]
    else:
        return schema


# ---------------------------------------------------------------------------
# Offline Executor
# ---------------------------------------------------------------------------

class OfflineExecutor:
    """
    Validates and executes tool calls offline.
    Maintains session state so later calls reference earlier outputs.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory_store: MemoryStore | None = None,
        seed: int | None = None,
    ):
        self.registry = registry
        self.memory = memory_store
        self.rng = random.Random(seed)

    def execute_step(
        self,
        endpoint_id: str,
        user_inputs: dict[str, Any],
        session: SessionState,
        transition: Transition | None = None,
        step_index: int = 0,
    ) -> StepOutput:
        """
        Execute one tool call step.

        Args:
            endpoint_id:  which endpoint to call
            user_inputs:  values explicitly provided by the user
            session:      current session state
            transition:   transition leading to this step (for field_mappings)
            step_index:   position in the chain (0 = first step)

        Returns:
            StepOutput with validated arguments and mock output
        """
        ep = self.registry.get_endpoint(endpoint_id)
        if ep is None:
            raise ValueError(f"[executor] Unknown endpoint: {endpoint_id}")

        # Resolve arguments using precedence policy
        arguments, was_grounded = self._resolve_arguments(
            endpoint=ep,
            user_inputs=user_inputs,
            session=session,
            transition=transition,
            step_index=step_index,
        )

        # Validate arguments
        validation_errors = self._validate_arguments(ep, arguments)
        if validation_errors:
            # Fill missing required params with mocks rather than failing
            for param in ep.parameters:
                if param.required and param.name not in arguments:
                    arguments[param.name] = _mock_value_for_param(param, self.rng)

        # Generate mock output
        output = _generate_mock_output(ep, arguments, self.rng)

        step = StepOutput(
            step_index=step_index,
            endpoint_id=endpoint_id,
            arguments=arguments,
            output=output,
            was_grounded=was_grounded,
        )

        # Record in session
        session.record_step(step)

        # Write to session memory — scoped per conversation so sessions don't bleed
        if self.memory:
            self.memory.add(
                content=json.dumps(output),
                scope=f"session_{session.conversation_id}",
                metadata={
                    "conversation_id": session.conversation_id,
                    "step": step_index,
                    "endpoint": endpoint_id,
                },
            )

        return step

    def _resolve_arguments(
        self,
        endpoint: Endpoint,
        user_inputs: dict[str, Any],
        session: SessionState,
        transition: Transition | None,
        step_index: int,
    ) -> tuple[dict[str, Any], bool]:
        """
        Resolve argument values using the 4-step precedence policy.

        Returns: (arguments dict, was_grounded by memory)

        PDF grounding definition: was_grounded = True whenever memory.search()
        returns at least one result for this step. The [Memory context] prompt
        is built once per step before filling any parameters.
        """
        arguments: dict[str, Any] = {}
        was_grounded = False

        # --- Step-level session memory retrieval (PDF requirement) ---
        # Query memory ONCE per step before filling any arguments.
        # Build the [Memory context] prompt exactly as the PDF specifies.
        # was_grounded is set here if search() returns any result at all.
        memory_field_cache: dict[str, Any] = {}

        if self.memory and step_index > 0:
            step_results = self.memory.search(
                query=f"{endpoint.name} {endpoint.intent}",
                scope=f"session_{session.conversation_id}",
                top_k=5,
            )
            if step_results:
                # PDF definition: grounded = search() returned >= 1 result
                was_grounded = True

                # Build the [Memory context] prompt the PDF requires:
                #   [Memory context]
                #   {retrieved_entries}
                #   Given the above context and the current tool schema,
                #   fill in the arguments for {endpoint_name}.
                retrieved_entries = "\n".join(
                    r.get("memory", "") for r in step_results if r.get("memory")
                )
                argument_filling_prompt = (
                    f"[Memory context]\n{retrieved_entries}\n\n"
                    f"Given the above context and the current tool schema, "
                    f"fill in the arguments for {endpoint.name}."
                )
                # Store prompt on session for audit/logging
                if not hasattr(session, "_memory_prompts"):
                    session._memory_prompts = []
                session._memory_prompts.append({
                    "step": step_index,
                    "endpoint": endpoint.endpoint_id,
                    "prompt": argument_filling_prompt,
                })

                # Build flat field cache from all retrieved entries for extraction
                for r in step_results:
                    try:
                        mem_content = json.loads(r.get("memory", "{}"))
                        if isinstance(mem_content, dict):
                            memory_field_cache.update(mem_content)
                    except (json.JSONDecodeError, TypeError):
                        pass

        # --- Per-parameter resolution ---
        for param in endpoint.parameters:
            if not param.required and param.name not in user_inputs:
                if param.default is not None:
                    arguments[param.name] = param.default
                continue

            # Priority 1: Explicit user input
            if param.name in user_inputs:
                arguments[param.name] = user_inputs[param.name]
                continue

            # Priority 2: field_mappings from prior transition
            if transition and step_index > 0:
                for fm in transition.field_mappings:
                    if fm.target_param == param.name:
                        field_value = session.get_field(fm.source_field)
                        if field_value is not None:
                            arguments[param.name] = field_value
                            break

            if param.name in arguments:
                continue

            # Priority 2.5: Direct accumulated_fields lookup
            # Catches semantic/category edges where field_mappings is empty but a prior
            # step output contains the required field by exact name match. This is
            # more reliable than memory retrieval because it's an in-process record.
            if step_index > 0:
                direct_value = session.accumulated_fields.get(param.name)
                if direct_value is not None:
                    arguments[param.name] = direct_value
                    continue

            # Priority 3: Extract from step-level memory cache
            if step_index > 0 and param.name in memory_field_cache:
                arguments[param.name] = memory_field_cache[param.name]
                continue

            # Priority 4: Default/mock fallback
            arguments[param.name] = _mock_value_for_param(param, self.rng)

        return arguments, was_grounded

    def _validate_arguments(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
    ) -> list[str]:
        """Validate arguments against endpoint schema. Returns list of errors."""
        errors: list[str] = []
        for param in endpoint.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required param: {param.name}")
            if param.enum and param.name in arguments:
                if str(arguments[param.name]) not in [str(e) for e in param.enum]:
                    errors.append(f"Invalid enum value for {param.name}: {arguments[param.name]}")
        return errors

    def create_session(self, conversation_id: str) -> SessionState:
        """Create a new session for a conversation."""
        return SessionState(conversation_id=conversation_id)
