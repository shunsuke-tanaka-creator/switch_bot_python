"""Versioned cross-client event contract. Coordinates intentionally never leave iPhone."""

from datetime import datetime, timedelta, timezone
import math
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["research-v2"] = "research-v2"
    event_id: str = Field(min_length=1, max_length=200)
    participant_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    occurred_at: datetime
    kind: Literal[
        "weather_mood",
        "place_visit",
        "typing_reason",
        "intervention_outcome",
        "question_state",
        "preferences",
    ]
    payload: dict

    @model_validator(mode="after")
    def valid(self):
        if self.occurred_at.tzinfo is None or self.occurred_at > datetime.now(
            timezone.utc
        ) + timedelta(minutes=5):
            raise ValueError("timezone required; future events rejected")
        p = self.payload
        fields = {
            "weather_mood": ({"weather"}, {"question_id"}),
            "place_visit": ({"place_id", "phase", "accuracy_m"}, {"device_id"}),
            "typing_reason": ({"question_id", "reason"}, set()),
            "intervention_outcome": (
                {"question_id", "action", "executed"},
                {"helpfulness"},
            ),
            "question_state": ({"question_id", "state"}, set()),
            "preferences": (
                set(),
                {
                    "questions_enabled",
                    "interventions_enabled",
                    "daily_limit",
                    "intervention_limit",
                    "gap_minutes",
                },
            ),
        }
        required, optional = fields[self.kind]
        if not required <= p.keys() or p.keys() - (required | optional):
            raise ValueError("invalid payload fields")
        if self.kind == "weather_mood" and p["weather"] not in (
            "happy",
            "normal",
            "sad",
            "angry",
        ):
            raise ValueError("invalid weather")
        if self.kind == "place_visit":
            if not isinstance(p["place_id"], str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,100}", p["place_id"]
            ):
                raise ValueError("invalid place id")
            if p["phase"] not in ("arrival", "departure"):
                raise ValueError("invalid phase")
            if (
                type(p["accuracy_m"]) not in (int, float)
                or not math.isfinite(p["accuracy_m"])
                or not 0 <= p["accuracy_m"] <= 10000
            ):
                raise ValueError("invalid accuracy")
        if "question_id" in p and (
            not isinstance(p["question_id"], str)
            or not 1 <= len(p["question_id"]) <= 200
        ):
            raise ValueError("invalid question id")
        if self.kind == "typing_reason" and p["reason"] not in (
            "mistype",
            "rewrite",
            "conversion",
            "unknown",
        ):
            raise ValueError("invalid reason")
        if self.kind == "intervention_outcome":
            if (
                p["action"] not in ("short_break", "refresh", "switch_task")
                or type(p["executed"]) is not bool
            ):
                raise ValueError("invalid action outcome")
            if p.get("helpfulness") not in (
                None,
                "helpful",
                "unchanged",
                "worse",
                "unknown",
            ):
                raise ValueError("invalid helpfulness")
            if not p["executed"] and p.get("helpfulness") not in (None, "unknown"):
                raise ValueError("unexecuted action cannot have an effect")
        if self.kind == "question_state" and p["state"] not in (
            "shown",
            "scheduled",
            "deferred",
            "declined",
            "stop",
            "resume",
        ):
            raise ValueError("invalid state")
        if self.kind == "preferences":
            if not p:
                raise ValueError("empty preferences")
            for k, v in p.items():
                if k.endswith("_enabled"):
                    if type(v) is not bool:
                        raise ValueError("boolean required")
                elif (
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or int(v) != v
                    or not 1 <= v <= (240 if k == "gap_minutes" else 12)
                ):
                    raise ValueError("invalid budget")
                else:
                    p[k] = int(v)
        return self
