"""
generator/user_proxy.py
-----------------------
User-Proxy Agent — simulates the human side of the conversation.

Generates initial requests, answers clarification questions,
and provides follow-up confirmations.

Initially template-driven for determinism.
Architecture allows LLM-based wording to be swapped in later
without changing the structural interface.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthetic_datagen.common.types import ClarificationStep
from synthetic_datagen.graph.registry import ToolRegistry
from synthetic_datagen.planner import StructuredConversationPlan


@dataclass
class UserTurn:
    """One user turn in the conversation."""
    role: str = "user"
    content: str = ""
    resolved_params: dict[str, Any] = field(default_factory=dict)


class UserProxyAgent:
    """Simulates user utterances based on the conversation plan.

    When an Anthropic API key is available, LLM-backed generation is used for
    initial requests, intent clarifications, and confirmations — producing
    natural, contextually grounded user messages.

    The missing_required_param branch of answer_clarification() always stays
    template-driven because it must return typed canonical values for executor
    grounding. LLM cannot reliably produce a specific typed value that matches
    what the executor expects.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        seed: int | None = None,
        api_key: str | None = None,
        llm_model: str = "claude-haiku-4-5-20251001",
    ):
        self.registry = registry
        self.rng = random.Random(seed)
        self._api_key = api_key
        self._llm_model = llm_model
        self._llm_client = None

    def _get_llm_client(self) -> Any:
        """Lazy-initialise Anthropic client. Returns None if unavailable."""
        if self._llm_client is not None:
            return self._llm_client
        try:
            import anthropic
        except ImportError:
            return None
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("ANTHROPIC_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass
        if not api_key:
            return None
        self._llm_client = anthropic.Anthropic(api_key=api_key)
        return self._llm_client

    # ------------------------------------------------------------------
    # LLM-backed generation (used when API key is available)
    # ------------------------------------------------------------------

    def _llm_initial_request(self, plan: StructuredConversationPlan) -> str | None:
        """Ask the LLM to write a natural opening user message for this plan."""
        client = self._get_llm_client()
        if client is None:
            return None
        tool_names = ", ".join(
            s.endpoint_id.split("::")[-1].replace("_", " ")
            for s in sorted(plan.steps, key=lambda x: x.step_index)
        ) if plan.steps else "various tools"
        prompt = (
            f"You are simulating a user in a chat with an AI assistant.\n"
            f"The user wants to accomplish: {plan.user_goal}\n"
            f"Domain: {plan.domain}\n"
            f"The assistant will use these tools: {tool_names}\n"
            f"Conversation style: {plan.conversation_style}\n\n"
            f"Write ONE short, natural opening message (1-2 sentences) from the user "
            f"that reflects their goal. Do not mention tool names. "
            f"Sound like a real person, not a template. Output only the message text."
        )
        try:
            response = client.messages.create(
                model=self._llm_model,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip().strip('"')
            return text if text else None
        except Exception:
            return None

    def _llm_intent_clarification(self, plan: StructuredConversationPlan) -> str | None:
        """Ask the LLM to write a natural clarification of the user's intent."""
        client = self._get_llm_client()
        if client is None:
            return None
        prompt = (
            f"You are simulating a user in a chat with an AI assistant.\n"
            f"The assistant just asked you to clarify your request.\n"
            f"Your actual goal is: {plan.user_goal}\n"
            f"Domain: {plan.domain}\n\n"
            f"Write ONE short, natural response (1-2 sentences) that clarifies "
            f"what you want. Sound like a real person. Output only the message text."
        )
        try:
            response = client.messages.create(
                model=self._llm_model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip().strip('"')
            return text if text else None
        except Exception:
            return None

    def _llm_confirmation(self, plan: StructuredConversationPlan) -> str | None:
        """Ask the LLM to write a natural confirmation message."""
        client = self._get_llm_client()
        if client is None:
            return None
        prompt = (
            f"You are simulating a user in a chat with an AI assistant.\n"
            f"The assistant just completed your request: {plan.user_goal}\n"
            f"Write ONE short, natural confirmation or thank-you (1 sentence). "
            f"Sound like a real person. Output only the message text."
        )
        try:
            response = client.messages.create(
                model=self._llm_model,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip().strip('"')
            return text if text else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_params_from_plan(self, plan: StructuredConversationPlan) -> dict:
        """
        Extract tool parameter values from the plan so the user's opening message
        implicitly supplies them to the executor.

        This ensures the executor uses the user's stated goal as the `query`/`keyword`
        rather than falling back to generic mock values like 'best options near me'.
        """
        resolved: dict = {}
        goal = plan.user_goal or ""

        # Build a concise search query from the user goal by stripping common preambles.
        # The executor uses this for Priority 1 resolution of `query`/`keyword` params.
        query_text = goal
        _ALL_STRIP = [
            "I need help to ", "I need help with ", "Help me ", "I'd like to ",
            "Please ", "Can you ", "Could you ", "I want to ", "I'm trying to ",
            "I would like to ", "I am looking to ", "I need to ",
            "Find relevant ", "Search for relevant ", "Retrieve relevant ",
            "Search for available ", "Find available ", "Find and book ",
            "Find and purchase ", "Book a ", "Book the ",
            "Look up ", "Retrieve ", "Search for ", "Find ", "Get ",
        ]
        for prefix in _ALL_STRIP:
            if query_text.lower().startswith(prefix.lower()):
                query_text = query_text[len(prefix):]
                break
        # Cut at "and [action verb]" conjunctions — these indicate the second half of a
        # compound goal and shouldn't be included in a keyword search query.
        # e.g. "job openings and review listings that match..." → "job openings"
        import re as _re
        query_text = _re.split(
            r'\s+and\s+(?:review|book|track|compare|check|apply|analyze|manage|schedule|send|create|update|find|get|retrieve|purchase|use|explore)\b',
            query_text, maxsplit=1, flags=_re.IGNORECASE
        )[0].strip()
        # Also cut at relative clauses that add noise to keyword queries
        query_text = _re.split(r'\s+(?:that\s+match|to\s+apply\s+for|in\s+order\s+to|for\s+the\s+purpose)', query_text, maxsplit=1)[0].strip()
        query_text = query_text.strip().rstrip(".")[:80]
        if query_text:
            resolved["query"] = query_text
            resolved["keyword"] = query_text[:60]
            resolved["search_term"] = query_text[:60]

        # Location — extract from user goal text
        location = self._extract_location_from_goal(goal)
        if location:
            resolved["location"] = location
            resolved["city"] = location

        # Currency pair extraction — "500 EUR to USD", "convert 200 GBP to JPY"
        import re as _re2
        curr_m = _re2.search(
            r'(\d+(?:\.\d+)?)\s+([A-Z]{3})\b.*?\bto\s+([A-Z]{3})\b',
            goal, _re2.IGNORECASE
        )
        if curr_m:
            amt, from_c, to_c = curr_m.groups()
            resolved.setdefault("from_currency", from_c.upper())
            resolved.setdefault("to_currency", to_c.upper())
            resolved.setdefault("amount", float(amt))

        # Domain-specific defaults so common required params are pre-filled
        domain_lower = plan.domain.lower() if plan.domain else ""
        if any(kw in domain_lower for kw in ("job", "career", "employ", "recruitment")):
            resolved.setdefault("job_title", "Software Engineer")

        return resolved

    def generate_initial_request(self, plan: StructuredConversationPlan) -> UserTurn:
        """Generate the opening user message grounded in domain and conversation style.

        Tries LLM first for a natural, contextually grounded message.
        Falls back to domain-opener templates if LLM is unavailable or fails.
        """
        # Extract tool params from plan up front — applied regardless of LLM/template path
        initial_params = self._extract_params_from_plan(plan)

        llm_text = self._llm_initial_request(plan)
        if llm_text:
            return UserTurn(content=llm_text, resolved_params=initial_params)
        # --- Template fallback ---
        _DOMAIN_OPENERS = {
            # Title-case keys (legacy planner)
            "Travel":          ["I'm planning a trip and need help.",
                                "I'd like to arrange some travel.",
                                "I need assistance with a travel booking."],
            "Finance":         ["I need some help with my finances.",
                                "I'd like to check on a financial matter.",
                                "I have a finance-related question."],
            "Food":            ["I'm looking for somewhere to eat.",
                                "I need help finding a restaurant.",
                                "I'd like to make a dining reservation."],
            "Weather":         ["I need to check the weather for my plans.",
                                "Can you help me with a weather lookup?",
                                "I'd like to know the forecast."],
            "Shopping":        ["I'm looking to buy something online.",
                                "I need help finding a product.",
                                "I'd like to search for an item."],
            "News":            ["I'd like to catch up on the latest news.",
                                "Can you find some recent news for me?",
                                "I need some information on current events."],
            "Jobs":            ["I'm looking for job opportunities.",
                                "I need help with a job search.",
                                "I'd like to explore some career options."],
            "Events":          ["I'm looking for events to attend.",
                                "I need help finding tickets for an event.",
                                "I'd like to book tickets for something."],
            # Lower-case multi-word keys (DeterministicNarrativeBackend)
            "travel planning": ["I'm planning a trip and need help.",
                                "I'd like to arrange some travel.",
                                "I need assistance booking flights and a hotel."],
            "weather information": ["I need to check the weather for my plans.",
                                   "Can you help me with a weather lookup?",
                                   "I'd like to know the forecast."],
            "financial services":  ["I need some help with a financial matter.",
                                    "I'd like to check some financial data.",
                                    "I have a finance-related question."],
            "e-commerce":          ["I'm looking to buy something online.",
                                   "I need help finding a product.",
                                   "I'd like to compare and order an item."],
            "food and dining":     ["I'm looking for somewhere to eat.",
                                   "I need help finding a restaurant.",
                                   "I'd like to find a recipe or dining option."],
            "career services":     ["I'm looking for job opportunities.",
                                   "I need help with a job search.",
                                   "I'd like to explore career options."],
            "entertainment":       ["I'm looking for events to attend.",
                                   "I need help finding tickets for a show.",
                                   "I'd like to book something fun this weekend."],
            "news and media":      ["I'd like to catch up on the latest news.",
                                   "Can you find some recent articles for me?",
                                   "I need to stay informed on current events."],
            "maps and navigation": ["I need help finding directions.",
                                   "I'd like to find places near me.",
                                   "Can you help me navigate to a location?"],
            "productivity":        ["I need help scheduling a meeting.",
                                   "I'd like to organize my calendar.",
                                   "Can you help me set up a task or reminder?"],
            "communication":       ["I need to send a message to someone.",
                                   "I'd like help composing an email.",
                                   "Can you help me contact someone?"],
            "general assistance":  ["I need help completing a few tasks.",
                                   "I have something I'd like assistance with.",
                                   "Can you help me get a few things done?"],
        }

        _STYLE_SUFFIXES = {
            "direct":              " {goal}",
            "exploratory":         " I was thinking — {goal}",
            "underspecified":      " I need some help, not sure exactly where to start.",
            "preference_driven":   " I have some specific preferences. {goal}",
            "correction_heavy":    " {goal} — though I may need to adjust as we go.",
            "comparison_oriented": " I'd like to compare options. {goal}",
            "goal_driven":         " {goal} — that's my main goal right now.",
        }

        domain = plan.domain
        style = plan.conversation_style

        # When the user goal is a generic multi-task phrase, force "general assistance"
        # opener so the opener and goal don't contradict each other (e.g. "I'm looking
        # for events to attend. I have some tasks to get done." is incoherent).
        _MULTI_TASK_MARKERS = (
            "I have a few different tasks",
            "I need assistance with several things",
            "I have some tasks to get done",
            "I need to take care of a few things",
            "I have a couple of tasks",
        )
        if any(marker in plan.user_goal for marker in _MULTI_TASK_MARKERS):
            domain = "general assistance"

        # Enumerate specific tasks so the assistant has clear context before
        # asking for params. Apply to:
        # - "general assistance" chains (multi-domain, always enumerate)
        # - Any chain with 3+ steps (user should know what's coming)
        if plan.steps and (domain == "general assistance" or len(plan.steps) >= 3):
            task_labels = []
            for s in sorted(plan.steps, key=lambda x: x.step_index):
                label = self._endpoint_to_task_label(s.endpoint_id)
                if label and label not in task_labels:
                    task_labels.append(label)
            if len(task_labels) >= 2:
                tasks_str = (", ".join(task_labels[:-1]) + f" and {task_labels[-1]}"
                             if len(task_labels) > 1 else task_labels[0])
                openers_list = [
                    f"I need help with a few things: {tasks_str}.",
                    f"I have several tasks I'd like to complete: {tasks_str}.",
                    f"Could you help me with the following: {tasks_str}?",
                ]
                return UserTurn(content=self.rng.choice(openers_list), resolved_params=initial_params)

        opener = self.rng.choice(
            _DOMAIN_OPENERS.get(domain, [f"I need help with a {domain.lower()} task."])
        )
        suffix_template = _STYLE_SUFFIXES.get(style, " {goal}")

        # For underspecified style, don't append the full goal
        if style == "underspecified":
            content = opener + suffix_template
        else:
            content = opener + suffix_template.format(goal=plan.user_goal)

        return UserTurn(content=content, resolved_params=initial_params)

    def _endpoint_to_task_label(self, endpoint_id: str) -> str:
        """Convert an endpoint_id like 'hotel_booking::search_hotels' to a user-friendly task label."""
        ep = endpoint_id.split("::")[-1].lower() if "::" in endpoint_id else endpoint_id.lower()
        tool = endpoint_id.split("::")[0].lower() if "::" in endpoint_id else endpoint_id.lower()

        # Match on endpoint name keywords (order matters — more specific first)
        _EP_LABELS: list[tuple[tuple[str, ...], str]] = [
            # Hotel
            (("book_hotel", "reserve_hotel", "hotel_booking"), "booking a hotel"),
            (("search_hotel", "find_hotel", "hotel_search", "hotel_availability"), "searching for hotels"),
            (("get_hotel", "hotel_detail", "hotel_info"), "looking up hotel details"),
            # Flight
            (("book_flight", "flight_booking"), "booking a flight"),
            (("search_flight", "find_flight", "flight_search", "get_flight"), "searching for flights"),
            # Weather
            (("weather_forecast", "get_forecast", "forecast"), "checking the weather forecast"),
            (("current_weather", "weather_current", "weather"), "checking current weather"),
            # Currency / Finance
            (("convert_currency", "currency_convert", "currency_exchange"), "converting currency"),
            (("exchange_rate",), "looking up exchange rates"),
            (("stock_price", "get_stock", "company_financials", "historical_price", "financial"), "checking financial data"),
            # Restaurant
            (("book_restaurant", "restaurant_reservation", "make_reservation"), "booking a restaurant"),
            (("search_restaurant", "find_restaurant", "restaurant_search"), "finding restaurants"),
            (("get_restaurant", "restaurant_detail", "restaurant_menu", "restaurant_info"), "looking up a restaurant"),
            # Recipe
            (("search_recipe", "find_recipe", "recipe_search"), "searching for recipes"),
            (("get_recipe", "recipe_detail", "recipe_info"), "getting recipe details"),
            # Jobs
            (("search_job", "find_job", "job_search"), "searching for jobs"),
            (("get_job", "job_detail", "job_info"), "looking up job details"),
            (("salary",), "checking salary information"),
            # Events / Entertainment
            (("purchase_ticket", "buy_ticket", "book_ticket"), "purchasing event tickets"),
            (("search_event", "find_event", "event_search"), "searching for events"),
            (("get_event", "event_ticket", "event_detail", "event_info"), "looking up event details"),
            # News
            (("search_news", "find_news", "news_search"), "checking the latest news"),
            (("get_article", "article_detail", "article_info"), "reading an article"),
            # Products
            (("search_product", "find_product", "product_search"), "searching for a product"),
            (("get_product", "product_detail", "product_review", "product_info"), "looking up a product"),
            (("add_to_cart", "purchase_product", "order_product", "cart"), "adding to cart"),
            # User profile
            (("update_preference", "save_preference", "update_profile", "save_profile", "update_setting"), "updating preferences"),
            (("get_profile", "user_profile", "profile_info"), "looking up account info"),
            # Translation / Language
            (("translate", "translation"), "translating text"),
            (("detect_language", "identify_language"), "detecting language"),
            # Maps
            (("geocode", "get_direction", "search_nearby", "nearby_place", "find_location"), "looking up location"),
            # Calendar
            (("schedule", "create_event", "calendar"), "scheduling an event"),
            # Email / Communication
            (("send_email", "compose_email", "email"), "sending an email"),
        ]
        for keywords, label in _EP_LABELS:
            if any(kw in ep for kw in keywords):
                return label
        # Fallback: humanize the endpoint name
        return ep.replace("_", " ")

    def _purpose_to_task_label(self, purpose: str) -> str:
        """Convert a step purpose to a short task label for the user's opening message.

        E.g. "Book a hotel for your stay." → "booking a hotel"
             "Search for available flights." → "searching for flights"
             "Convert the currency amount." → "converting currency"
        """
        p = purpose.strip().rstrip(".")
        p = p.replace("the user's", "your").replace("for the user", "for you")
        p_lower = p.lower()
        # Map to concise gerund phrases
        # Specific endpoint-level labels take priority
        _SPECIFIC_LABELS: list[tuple[tuple[str, ...], str]] = [
            (("book hotel", "hotel booking", "reserve hotel"), "booking a hotel"),
            (("hotel", "accommodation"), "looking up hotel options"),
            (("book flight", "flight booking", "purchase ticket"), "booking a flight"),
            (("search flight", "find flight", "flight search"), "searching for flights"),
            (("weather forecast", "check the weather", "fetch weather"), "checking the weather"),
            (("current weather",), "getting current weather conditions"),
            (("convert currency", "currency convert"), "converting currency"),
            (("exchange rate", "stock price", "company financial", "market data"), "checking financial data"),
            (("book restaurant", "restaurant reservation", "make reservation"), "booking a restaurant"),
            (("search restaurant", "find restaurant", "restaurant option"), "finding restaurants"),
            (("get restaurant", "restaurant menu", "restaurant detail"), "looking up a restaurant"),
            (("search recipe", "find recipe"), "searching for recipes"),
            (("get recipe", "recipe detail"), "getting recipe details"),
            (("get hotel", "hotel detail", "look up hotel"), "looking up hotel details"),
            (("search job", "find job", "job search"), "searching for jobs"),
            (("get job", "job detail"), "looking up job details"),
            (("salary",), "checking salary information"),
            (("search event", "find event", "event search"), "searching for events"),
            (("purchase ticket", "book ticket", "buy ticket"), "purchasing event tickets"),
            (("get event", "event ticket", "event detail"), "looking up event details"),
            (("search news", "find news", "news search"), "checking the latest news"),
            (("get article", "article detail"), "reading an article"),
            (("search product", "find product", "product search"), "searching for a product"),
            (("get product", "product detail"), "looking up a product"),
            (("add to cart", "purchase", "order product"), "adding to cart"),
            (("update preference", "save preference", "update profile"), "updating preferences"),
            (("user profile", "get profile"), "checking account info"),
            (("translate",), "translating text"),
            (("geocode", "direction", "nearby place", "map"), "getting directions"),
            (("schedule", "calendar", "create event"), "scheduling an event"),
            (("send email", "compose email"), "sending an email"),
        ]
        for keywords, label in _SPECIFIC_LABELS:
            if any(kw in p_lower for kw in keywords):
                return label
        return p_lower[:50]  # fallback to truncated purpose

    def answer_clarification(
        self,
        clarification: ClarificationStep,
        plan: StructuredConversationPlan,
    ) -> UserTurn:
        """Generate a user response to a clarification question.

        Returns a UserTurn with both the natural-language content AND
        resolved_params — the machine-readable values that back the utterance.
        These are passed to executor.execute_step(user_inputs=...) so tool
        arguments are consistent with what the user said.
        """
        if clarification.reason == "intent_ambiguity":
            llm_text = self._llm_intent_clarification(plan)
            if llm_text:
                return UserTurn(content=llm_text)
            # Template fallback
            templates = [
                f"I'd like to {plan.user_goal.lower().replace('help me', '').strip()}",
                f"Specifically, I need to complete a task related to {plan.domain.lower()}",
                f"I want to accomplish: {plan.user_goal}",
            ]
            return UserTurn(content=self.rng.choice(templates))

        # missing_required_param — collect utterance and resolved value per param
        if clarification.missing_params:
            parts = []
            resolved: dict[str, Any] = {}
            for param in clarification.missing_params:
                utterance, value = self._param_value_utterance(
                    param, domain=plan.domain, user_goal=plan.user_goal
                )
                parts.append(utterance)
                if value is not None:
                    resolved[param] = value
            content = " and ".join(parts) if parts else "Here is the information you need."
            return UserTurn(content=content, resolved_params=resolved)

        return UserTurn(content="Here is the additional information you need.")

    def generate_confirmation(self, plan: StructuredConversationPlan) -> UserTurn:
        """Generate a brief user confirmation or follow-up.

        Tries LLM first for a natural, goal-aware confirmation.
        Falls back to canned confirmations if LLM is unavailable or fails.
        """
        llm_text = self._llm_confirmation(plan)
        if llm_text:
            return UserTurn(content=llm_text)
        # Template fallback
        confirmations = [
            "That looks great, thank you!",
            "Perfect, please proceed with that.",
            "Yes, that's exactly what I need.",
            "Great, go ahead.",
        ]
        return UserTurn(content=self.rng.choice(confirmations))

    def _extract_location_from_goal(self, user_goal: str) -> str | None:
        """
        Extract a city/location mentioned in the user's goal string.
        Used so clarification answers reflect the destination the user already stated
        rather than a hardcoded fallback city like 'New York'.
        """
        import re
        # Common prepositions that introduce a destination in the user goal
        patterns = [
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",       # "in Boston", "in New York"
            r"\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",        # "to Paris"
            r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",       # "for Chicago"
            r"\bnear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",      # "near Seattle"
        ]
        _STOP_WORDS = {
            "the", "a", "an", "my", "your", "our", "their", "this", "that",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        }
        for pattern in patterns:
            matches = re.findall(pattern, user_goal)
            for m in matches:
                if m.lower() not in _STOP_WORDS and len(m) > 2:
                    return m
        return None

    def _param_value_utterance(self, param_name: str, domain: str = "", user_goal: str = "") -> tuple[str, Any]:
        """Return (natural-language utterance, canonical typed value) for a param.

        Both are derived from the same source — the utterance is for the
        conversation text, the value is for executor.execute_step(user_inputs).
        No LLM needed: since we generated the utterance, we always know the value.
        """
        # Each entry: utterance text, canonical typed value
        _PARAM_MAP: dict[str, tuple[str, Any]] = {
            "origin":           ("I'll be flying from New York (JFK)",  "JFK"),
            "destination":      ("I want to go to Paris (CDG)",         "CDG"),
            "city":             ("I'm looking at Paris",                 "Paris"),
            "location":         ("I'm in New York",                      "New York"),
            "date":             ("The date is June 15th",                "2024-06-15"),
            "departure_date":   ("I want to leave on June 15th",         "2024-06-15"),
            "return_date":      ("I'll return on June 22nd",             "2024-06-22"),
            "check_in":         ("Check-in would be June 15th",          "2024-06-15"),
            "check_out":        ("Check-out would be June 17th",         "2024-06-17"),
            "start_date":       ("Starting June 1st",                    "2024-06-01"),
            "end_date":         ("Ending June 30th",                     "2024-06-30"),
            "from_date":        ("From January 1st",                     "2024-01-01"),
            "to_date":          ("To June 15th",                         "2024-06-15"),
            "from_currency":    ("I have US dollars",                    "USD"),
            "to_currency":      ("I need euros",                         "EUR"),
            "amount":           ("The amount is 100",                    100.0),
            "currency":         ("In US dollars",                        "USD"),
            "symbol":           ("I'm interested in Apple (AAPL)",       "AAPL"),
            "query":            ("I'm looking for options in my area",     "options near me"),
            "keyword":          ("The keyword is travel deals",          "travel deals"),
            "name":             ("The name is Jane Smith",               "Jane Smith"),
            "guest_name":       ("The guest name is Jane Smith",         "Jane Smith"),
            "passenger_name":   ("The passenger is Jane Smith",          "Jane Smith"),
            "email":            ("My email is user@example.com",         "user@example.com"),
            "buyer_email":      ("My email is user@example.com",         "user@example.com"),
            "passenger_email":  ("My email is user@example.com",         "user@example.com"),
            "party_size":       ("There will be 2 of us",                2),
            "passengers":       ("Just 1 passenger",                     1),
            "quantity":         ("I'd like 2",                           2),
            "time":             ("I'd like a reservation at 7:30 PM",    "19:30"),
            "address":          ("The address is 123 Main St, New York", "123 Main St, New York, NY"),
            "job_title":        ("I'm a Software Engineer",              "Software Engineer"),
            "language":         ("In English",                           "en"),
            "target_language":  ("Translate to French",                  "fr"),
            "source_language":  ("From English",                         "en"),
            "text":             ("The text is: Hello, how are you?",     "Hello, how are you?"),
            "country":          ("In the US",                            "us"),
            "category":         ("General category",                     "general"),
            "type":             ("Standard type",                        "standard"),
            "preferences":      ("I prefer budget-friendly options",     "budget-friendly"),
            "start_datetime":   ("Starting June 15th at 10am",           "2024-06-15T10:00:00"),
            "end_datetime":     ("Ending at 11am",                       "2024-06-15T11:00:00"),
            "title":            ("The title is Meeting",                 "Meeting"),
            "topic":            ("The topic is technology",              "technology"),
            "subject":          ("Subject: Meeting Request",             "Meeting Request"),
            "message":          ("Message: I'd like to schedule a meeting", "I'd like to schedule a meeting"),
            # Entity IDs — the user references a specific item from a prior interaction
            "flight_id":        ("I'd like to use flight FL247",           "FL247"),
            "hotel_id":         ("The hotel ID is H345",                   "H345"),
            "product_id":       ("The product ID is P235",                 "P235"),
            "recipe_id":        ("I'm looking at recipe RC342",            "RC342"),
            "event_id":         ("The event ID is EVT422",                 "EVT422"),
            "job_id":           ("I'm interested in job J523",             "J523"),
            "restaurant_id":    ("The restaurant ID is R623",              "R623"),
            "article_id":       ("The article ID is N724",                 "N724"),
            "booking_id":       ("My booking ID is BK823",                 "BK823"),
            "reservation_id":   ("The reservation ID is RES923",           "RES923"),
            "order_id":         ("My order ID is ORD123",                  "ORD123"),
            "message_id":       ("The message ID is MSG223",               "MSG223"),
            "ticket_id":        ("My ticket ID is TKT301",                 "TKT301"),
            "ticket_type":      ("I'd like general admission tickets",      "general"),
            "user_id":          ("My user ID is U101",                     "U101"),
            "contact_id":       ("The contact ID is C205",                 "C205"),
            "event_name":       ("The event is the Tech Summit",           "Tech Summit"),
            "hotel_name":       ("I'm thinking of the Grand Hotel",        "Grand Hotel"),
            "restaurant_name":  ("The restaurant is Bella Italia",         "Bella Italia"),
            "company":          ("The company is TechCorp",                "TechCorp"),
            "description":      ("Here is a brief description of my request", "brief description"),
            "servings":         ("I'd like to make 4 servings",              4),
            "num_results":      ("Please show me up to 5 results",           5),
            "limit":            ("Show me up to 10",                         10),
            "max_results":      ("I'd like up to 10 results",                10),
            "page":             ("Start from the first page",                1),
            "sort_by":          ("Sorted by relevance",                      "relevance"),
            "price_range":      ("My budget is up to $200",                  "0-200"),
            "min_price":        ("My minimum budget is $20",                 20.0),
            "max_price":        ("My maximum is $200",                       200.0),
            "radius":           ("Within 10 miles",                          10),
            "format":           ("In standard format",                       "standard"),
            "recipient":        ("Please send it to john.doe@example.com",   "john.doe@example.com"),
            "sender":           ("My email is user@example.com",              "user@example.com"),
            "phone":            ("My phone number is +1-555-0100",            "+1-555-0100"),
            "cabin_class":      ("Economy class is fine",                     "economy"),
            "sort":             ("Sort by relevance",                         "relevance"),
            "guests":           ("There will be 2 guests",                    2),
            "room_type":        ("I'd like a standard room",                  "standard"),
            "airline":              ("Any airline is fine",                       "any"),
            # Product params
            "product_name":         ("I'm looking for Wireless Headphones Pro",     "Wireless Headphones Pro"),
            "product_category":     ("In the electronics category",                  "electronics"),
            "brand":                ("Any brand is fine",                            "any"),
            "model":                ("The model is standard",                        "standard"),
            # Duration / scheduling
            "duration_minutes":     ("I'd like 60 minutes",                          60),
            "duration":             ("For about an hour",                            60),
            "num_participants":     ("There will be 3 participants",                  3),
            "participants":         ("There will be 3 participants",                  3),
            "attendees":            ("There will be 3 attendees",                     3),
            # Health
            "medication_name":      ("The medication is Aspirin",                    "Aspirin"),
            "symptom":              ("I have a headache",                            "headache"),
            "age":                  ("I'm 30 years old",                             30),
            # Sports
            "sport":                ("Basketball",                                   "basketball"),
            "league":               ("The NBA",                                      "NBA"),
            "team":                 ("The Lakers",                                   "Lakers"),
            "player_name":          ("LeBron James",                                 "LeBron James"),
            # Career
            "salary_range":         ("Around $80,000 to $120,000",                  "80000-120000"),
            "experience_level":     ("Mid-level",                                    "mid"),
            "skills":               ("Python and data analysis",                     "Python"),
            # Social
            "platform":             ("Twitter",                                      "twitter"),
            "hashtag":              ("The hashtag is #technology",                   "#technology"),
            "username":             ("My username is user123",                       "user123"),
        }

        if param_name in _PARAM_MAP:
            import re as _re

            # Special case: location/city params — use the city from user's goal
            # so the tool searches where the user asked, not a hardcoded fallback
            if param_name in ("location", "city") and user_goal:
                extracted = self._extract_location_from_goal(user_goal)
                if extracted:
                    return (f"I'm looking at {extracted}", extracted)

            # Special case: destination/origin — derive from plan goal
            if param_name == "destination" and user_goal:
                extracted = self._extract_location_from_goal(user_goal)
                if extracted:
                    return (f"My destination is {extracted}", extracted)

            if param_name == "origin" and user_goal:
                # Look for "from X" pattern
                m = _re.search(r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", user_goal)
                if m:
                    loc = m.group(1)
                    return (f"I'll be departing from {loc}", loc)

            # Special case: currency params — extract from goal pattern like "500 EUR to USD"
            if param_name in ("from_currency", "to_currency", "amount") and user_goal:
                # Match "X CURR to CURR" or "X in CURR" patterns
                m = _re.search(
                    r'(\d+(?:\.\d+)?)\s+([A-Z]{3})\b.*?\bto\s+([A-Z]{3})\b',
                    user_goal, _re.IGNORECASE
                )
                if not m:
                    m = _re.search(
                        r'\bconvert\s+(\d+(?:\.\d+)?)\s+([A-Z]{3})\b.*?\bto\s+([A-Z]{3})\b',
                        user_goal, _re.IGNORECASE
                    )
                if m:
                    amt, from_c, to_c = m.groups()
                    from_c, to_c = from_c.upper(), to_c.upper()
                    if param_name == "from_currency":
                        return (f"I have {from_c}", from_c)
                    if param_name == "to_currency":
                        return (f"I need {to_c}", to_c)
                    if param_name == "amount":
                        return (f"The amount is {amt}", float(amt))

            # Special case: make `query` domain-specific
            if param_name == "query":
                # Derive a natural search query from the user goal.
                # Strip both user-natural preambles and planner-style action verbs
                # so the query contains the meaningful domain terms, not boilerplate.
                if user_goal:
                    q = user_goal.strip().rstrip(".")
                    _STRIP_PREFIXES = [
                        # User-natural preambles
                        "I need help to ", "I need help with ", "Help me ",
                        "I'd like to ", "Please ", "Can you ", "Could you ",
                        "I want to ", "I'm trying to ", "I would like to ",
                        "I am looking to ", "I need to ",
                        # Planner-style action verbs (LLM-generated goals often start with these)
                        "Find relevant ", "Search for relevant ", "Retrieve relevant ",
                        "Search for available ", "Find available ", "Find and book ",
                        "Find and purchase ", "Find and order ", "Book a ", "Book the ",
                        "Look up ", "Retrieve ", "Search for ", "Find ", "Get ",
                    ]
                    for prefix in _STRIP_PREFIXES:
                        if q.lower().startswith(prefix.lower()):
                            q = q[len(prefix):]
                            break
                    # Cut at "and [action verb]" conjunctions
                    q = _re.split(
                        r'\s+and\s+(?:review|book|track|compare|check|apply|analyze|manage|schedule|send|create|update|find|get|retrieve|purchase|explore)\b',
                        q, maxsplit=1, flags=_re.IGNORECASE
                    )[0].strip()
                    q = q.strip()[:80]
                    if q and len(q) > 5:
                        return (f"I'm looking for {q.lower()}", q)
            return _PARAM_MAP[param_name]

        # Generic fallback — use realistic values, never literal "{param}_value"
        readable = param_name.replace("_", " ")
        # IDs get a consistent alphanumeric mock
        if param_name.endswith("_id") or param_name.endswith("_key"):
            mock_id = param_name.upper().replace("_", "")[:3] + "101"
            return (f"The {readable} is {mock_id}", mock_id)
        # Names get a realistic product/item name
        if "name" in param_name:
            return (f"The {readable} is Wireless Headphones Pro", "Wireless Headphones Pro")
        # Duration/count params get a number
        if any(kw in param_name for kw in ("duration", "minutes", "hours", "count", "number", "num")):
            return (f"I'd like {readable} of 60", 60)
        # Budget/price params
        if any(kw in param_name for kw in ("budget", "price", "cost", "amount", "rate")):
            return (f"My {readable} is around $100", 100.0)
        # Generic string fallback — readable phrase, not a literal placeholder
        return (f"The {readable} is standard", "standard")
