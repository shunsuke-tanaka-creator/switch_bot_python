"""Validated aggregate inputs. No personal text or individual observations."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .data import SCALES, dt

SCHEMA = "aggregate-tendencies-v1"
P001_PROFILE = Path(__file__).parent / "profiles" / "p001.json"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value, low, high, field):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{field} must be a finite number in [{low}, {high}]")


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"aggregate fields must be {sorted(required)} (optional: {sorted(optional)})")


def validate_profile(value):
    """Reject unrecognized fields rather than accidentally accept raw narratives."""
    _keys(value, ("schema_version", "profile_version", "participant_id", "measured", "assumed"))
    if value["schema_version"] != SCHEMA or not isinstance(value["participant_id"], str) or value["participant_id"].endswith("-sample"):
        raise ValueError("invalid profile schema or source participant")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", value["participant_id"]):
        raise ValueError("invalid participant_id")
    if not isinstance(value["profile_version"], str) or not value["profile_version"]:
        raise ValueError("profile_version is required")
    m, a = value["measured"], value["assumed"]
    _keys(m, ("scales", "baseline_method", "answers_per_day", "scope_ratio", "answer_time_peaks",
              "weather_probabilities", "calendar", "topology"))
    if m["baseline_method"] != "condition-weighted daily mean; sd floor 0.5":
        raise ValueError("baseline_method must identify the production aggregation and SD floor")
    _keys(m["scales"], SCALES)
    for name, scale in m["scales"].items():
        _keys(scale, ("mean", "sd"), ("value_counts",))
        _number(scale["mean"], 1, 5, name + ".mean")
        _number(scale["sd"], .5, 2, name + ".sd")
        if scale["sd"] > max(.5, math.sqrt((scale["mean"] - 1) * (5 - scale["mean"]))):
            raise ValueError(name + ".sd is impossible for the bounded mean (including the SD floor)")
        if "value_counts" in scale:
            _keys(scale["value_counts"], ("1", "2", "3", "4", "5"))
            for count in scale["value_counts"].values():
                if type(count) is not int or count < 0:
                    raise ValueError("value_counts must be nonnegative aggregate integers")
            if sum(scale["value_counts"].values()) == 0:
                raise ValueError("value_counts cannot be empty")
    _number(m["answers_per_day"], .01, 20, "answers_per_day")
    _keys(m["scope_ratio"], ("current", "daily"))
    for v in m["scope_ratio"].values():
        _number(v, .01, 1000, "scope_ratio")
    if m["answer_time_peaks"] != ["22:00", "01:00", "09:00-11:00"]:
        raise ValueError("answer_time_peaks must describe the aggregate time bands")
    _keys(m["weather_probabilities"], ("happy", "normal", "sad", "angry"))
    for v in m["weather_probabilities"].values():
        _number(v, 0, 1, "weather probability")
    if not math.isclose(sum(m["weather_probabilities"].values()), 1, abs_tol=1e-9):
        raise ValueError("weather probabilities must sum to one")
    c = m["calendar"]
    _keys(c, ("lab_weekday", "work_weekday_weights", "scheduled_day_ratio", "block_median_hours", "start_hours"))
    if type(c["lab_weekday"]) is not int or not 0 <= c["lab_weekday"] < 7:
        raise ValueError("lab_weekday must be 0..6")
    _keys(c["work_weekday_weights"], ("0", "1", "2", "3"))
    for v in c["work_weekday_weights"].values():
        _number(v, 0, 100, "work weight")
    if not sum(c["work_weekday_weights"].values()):
        raise ValueError("work weights must have positive mass")
    _number(c["scheduled_day_ratio"], 0, 1, "scheduled_day_ratio")
    _number(c["block_median_hours"], .25, 12, "block_median_hours")
    if not isinstance(c["start_hours"], list) or not c["start_hours"]:
        raise ValueError("start_hours is required")
    for h in c["start_hours"]:
        _number(h, 0, 21, "start hour")
    t = m["topology"]
    _keys(t, ("states", "dominant_states", "feature_ranges"))
    if t["states"] != 5 or t["dominant_states"] != 1:
        raise ValueError("this generator uses five states with one dominant state")
    _keys(t["feature_ranges"], ("input", "corrections", "pauses"))
    for name, bounds in t["feature_ranges"].items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("feature range must contain two numbers")
        for v in bounds:
            _number(v, 0, 1, name)
        if bounds[0] >= bounds[1]:
            raise ValueError("feature ranges must increase")
    _keys(a, ("start_date", "rho", "factor_loadings", "noise_sd", "recall_noise_sd",
              "work_effect", "lab_effect", "intraday_effect", "no_answer_probability",
              "completion_probability", "time_mixture", "time_jitter_minutes",
              "state_weights", "dwell_median_minutes", "steps_per_state_hour",
              "anomaly_probability", "switchbot_enabled"))
    if not isinstance(a["start_date"], str):
        raise ValueError("start_date must be an ISO date")
    dt(a["start_date"] + "T00:00:00+09:00")
    for name, low, high in (("rho", 0, .99), ("noise_sd", .3, 2), ("recall_noise_sd", .3, 2),
                            ("no_answer_probability", 0, .95), ("completion_probability", .1, 1),
                            ("time_jitter_minutes", 1, 120), ("dwell_median_minutes", 30, 240),
                            ("anomaly_probability", 0, .2)):
        _number(a[name], low, high, name)
    _keys(a["factor_loadings"], SCALES)
    for name in ("work_effect", "lab_effect", "intraday_effect"):
        if not isinstance(a[name], list) or len(a[name]) != 3:
            raise ValueError(name + " needs three latent factor effects")
        for v in a[name]:
            _number(v, -2, 2, name)
    for values in a["factor_loadings"].values():
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError("factor_loadings need vitality/sleep pressure/mental load")
        for v in values:
            _number(v, -2, 2, "factor loading")
    for name, size in (("time_mixture", 3), ("state_weights", 5), ("steps_per_state_hour", 5)):
        if not isinstance(a[name], list) or len(a[name]) != size:
            raise ValueError(name + " has incorrect length")
        for v in a[name]:
            _number(v, 0, 10000 if name == "steps_per_state_hour" else 1, name)
        if name != "steps_per_state_hour" and not math.isclose(sum(a[name]), 1, abs_tol=1e-9):
            raise ValueError(name + " must sum to one")
    if type(a["switchbot_enabled"]) is not bool:
        raise ValueError("switchbot_enabled must be boolean")
    return copy.deepcopy(value)


def load_profile(path=P001_PROFILE):
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))


def category_probabilities(scale):
    """Smoothed counts first; otherwise maximum entropy on bounded first/second moments.

    The supplied SD is a daily baseline SD with a floor, not an identified raw
    response distribution. Treat its conversion to raw moments as an assumption.
    """
    if "value_counts" in scale:
        counts = np.array([scale["value_counts"][str(i)] for i in range(1, 6)], float)
        return (counts + .05) / (counts.sum() + .25)
    x = np.arange(1, 6, dtype=float)
    mean = scale["mean"]
    if mean in (1, 5):
        return np.eye(5)[int(mean) - 1]
    fraction = mean % 1
    variance = np.clip(scale["sd"] ** 2, fraction * (1 - fraction) + 1e-5,
                       (mean - 1) * (5 - mean) - 1e-5)
    features = np.column_stack((x, x ** 2))
    target = np.array([mean, mean ** 2 + variance])
    theta = np.zeros(2)
    for _ in range(200):
        logits = features @ theta
        p = np.exp(logits - logits.max())
        p /= p.sum()
        error = p @ features - target
        if np.max(np.abs(error)) < 1e-9:
            break
        centered = features - p @ features
        covariance = (centered.T * p) @ centered + np.eye(2) * 1e-9
        delta = np.linalg.solve(covariance, error)
        loss = np.log(np.exp(logits - logits.max()).sum()) + logits.max() - target @ theta
        step = 1.
        for _ in range(30):
            trial = theta - step * delta
            z = features @ trial
            new_loss = np.log(np.exp(z - z.max()).sum()) + z.max() - target @ trial
            if new_loss <= loss:
                theta = trial
                break
            step /= 2
    return p
