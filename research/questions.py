"""One durable shared queue/budget for keyboard, parent app and local notifications."""

from datetime import timedelta
from uuid import uuid4
from .data import dt
from .insights import recommend


def settings(rows):
    value = dict(
        questions_enabled=True,
        interventions_enabled=True,
        daily_limit=4,
        intervention_limit=2,
        gap_minutes=60,
    )
    explicit = set()
    for r in sorted(rows, key=lambda r: (dt(r["at"]), r.get("seq", 0))):
        if r["kind"] == "heartbeat":
            for k in ("questions_enabled", "interventions_enabled"):
                if k in r["payload"] and k not in explicit:
                    value[k] = r["payload"][k]
        if r["kind"] == "intervention_feedback" and r["payload"].get("outcome") in (
            "stop",
            "resume",
        ):
            value["interventions_enabled"] = r["payload"]["outcome"] == "resume"
        if r["kind"] == "research_event":
            p = r["payload"]
            if p["kind"] == "preferences":
                value.update(p["payload"])
                explicit.update(p["payload"])
            if p["kind"] == "question_state" and p["payload"]["state"] in (
                "stop",
                "resume",
            ):
                value["interventions_enabled"] = p["payload"]["state"] == "resume"
    return value


class QuestionQueue:
    def __init__(self, store, insights):
        self.store = store
        self.insights = insights

    def pending(self, p, at):
        at = dt(at)
        # GET is an atomic reservation. Different surfaces reuse the same pending ID.
        with self.store.lock:
            rows = [
                r
                for r in self.store.rows(p)
                if dt(r["at"]) <= at
                and dt(r["payload"].get("received_at", r["at"])) <= at
            ]
            pref = settings(rows)
            if not pref["questions_enabled"] or not 8 <= at.hour < 24:
                return None
            session = next((r for r in rows if r["kind"] == "session"), None)
            context = (session["payload"].get("context") or {}) if session else {}
            calendar_at = (context.get("measurement_updated_at") or {}).get(
                "calendar_events"
            )
            if calendar_at and timedelta(0) <= at - dt(calendar_at) <= timedelta(
                hours=2
            ):
                if any(
                    dt(e["start"]) <= at < dt(e["end"])
                    for e in context.get("calendar_events") or []
                ):
                    return None
            issued = [r for r in rows if r["kind"] == "question_issued"]
            answered = {r["id"] for r in rows if r["kind"] == "narrative"}
            states = {}
            weather_q = set()
            reason_q = set()
            for r in reversed(rows):
                if r["kind"] != "research_event":
                    continue
                ev = r["payload"]
                body = ev["payload"]
                qid = body.get("question_id")
                if ev["kind"] == "question_state":
                    states[qid] = (body["state"], dt(r["at"]))
                if ev["kind"] == "weather_mood":
                    weather_q.add(qid)
                if ev["kind"] == "typing_reason":
                    reason_q.add(qid)
            done = answered | weather_q | reason_q

            def expiry(r):
                return dt(
                    r["payload"].get("expiresAt") or (dt(r["at"]) + timedelta(hours=1))
                )

            def completed(r):
                return r["id"] in done or states.get(r["id"], ("",))[0] in (
                    "declined",
                    "stop",
                )

            for r in issued:
                if completed(r) or expiry(r) <= at:
                    continue
                if (
                    r["payload"]["kind"] == "breakCheck"
                    and not pref["interventions_enabled"]
                ):
                    continue
                if states.get(r["id"], ("",))[0] == "deferred":
                    if at - states[r["id"]][1] < timedelta(minutes=30):
                        return None
                return r["payload"]
            # Actual display counts plus live reservations bound cross-surface races.
            shown = {
                r["payload"]["payload"]["question_id"]: dt(r["at"])
                for r in rows
                if r["kind"] == "research_event"
                and r["payload"]["kind"] == "question_state"
                and r["payload"]["payload"]["state"] in ("shown", "scheduled")
                and dt(r["at"]).date() == at.date()
            }
            budget = {
                r["id"]
                for r in issued
                if dt(r["at"]).date() == at.date()
                and (expiry(r) > at or r["id"] in done)
            } | shown.keys()
            if len(budget) >= pref["daily_limit"]:
                return None
            # Keep one assessment opportunity for the daily head at night.
            if (
                at.hour < 22
                and pref["daily_limit"] > 1
                and len(budget) >= pref["daily_limit"] - 1
            ):
                return None
            recent = [dt(r["at"]) for r in issued if dt(r["at"]).date() == at.date()]
            if recent and at - max(recent) < timedelta(minutes=pref["gap_minutes"]):
                return None
            insight = self.insights.snapshot(p, at)
            fatigue, typing = insight["fatigue"], insight["typing"]
            recent_reason = next(
                (
                    r["payload"]["payload"]["reason"]
                    for r in rows
                    if r["kind"] == "research_event"
                    and r["payload"]["kind"] == "typing_reason"
                    and at - dt(r["at"]) <= timedelta(hours=2)
                ),
                None,
            )
            offered = [
                r
                for r in issued
                if r["payload"]["kind"] == "breakCheck"
                and dt(r["at"]).date() == at.date()
            ]
            kind = prompt = event = None
            options = []
            ttl = 60

            def opts(pairs):
                return [dict(label=a, value=b) for a, b in pairs]

            # Outcome is asked only after explicit acceptance, and never inferred from clicking alone.
            for r in issued:
                if r["payload"]["kind"] != "breakCheck" or not timedelta(
                    minutes=15
                ) <= at - dt(r["at"]) <= timedelta(hours=3):
                    continue
                answer = next(
                    (
                        x["payload"].get("answer", {}).get("label", "")
                        for x in rows
                        if x["kind"] == "narrative" and x["id"] == r["id"]
                    ),
                    "",
                )
                if not answer.startswith("accept:"):
                    continue
                if any(x["payload"].get("relatedID") == r["id"] for x in issued):
                    continue
                action = answer.split(":", 1)[1]
                kind = "supportOutcome"
                prompt = "先ほどの提案を実行しましたか？ 実行した場合、どうでしたか？"
                event = "outcome:" + r["id"]
                options = opts(
                    [
                        ("実行して役立った", f"outcome:{action}:helpful"),
                        ("実行して変わらない", f"outcome:{action}:unchanged"),
                        ("実行して悪化した", f"outcome:{action}:worse"),
                        ("実行していない", f"outcome:{action}:not_executed"),
                    ]
                )
                break
            if (
                kind is None
                and pref["interventions_enabled"]
                and len(offered) < pref["intervention_limit"]
                and (
                    fatigue.get("acute")
                    or fatigue.get("high")
                    or (typing.get("unusual") and recent_reason == "mistype")
                )
            ):
                reason = (
                    "typing"
                    if typing.get("unusual") and recent_reason == "mistype"
                    else "fatigue"
                )
                event = "support_" + reason
                kind = "breakCheck"
                ranked = recommend(rows, at, fatigue, typing)
                prompt = (
                    "入力の修正が普段より多めです。少し気分を変えますか？"
                    if reason == "typing"
                    else "疲労が高めと推定されています。少し休憩しませんか？"
                )
                if ranked[0]["based_on_event_id"]:
                    prompt += (
                        " 以前、似た状況で役立ったと回答した行動を先頭にしています。"
                    )
                options = opts(
                    [(a["label"], "accept:" + a["action"]) for a in ranked]
                    + [("今はしない", "declined"), ("提案を停止", "stop")]
                )
            places = insight["places"]
            anomaly = places["anomaly"] or places["time_anomaly"]
            if kind is None and anomaly:
                token = "outing:" + anomaly["at"]
                if not any(r["payload"].get("eventType") == token for r in issued):
                    kind = "outingDestination"
                    event = token
                    ttl = 30
                    prompt = (
                        "普段と違う外出の候補があります。今日はどちらへお出かけですか？"
                    )
                    options = opts(
                        [
                            ("いつもの場所", "usual"),
                            ("新しい場所", "new_place"),
                            ("予定変更", "changed_plan"),
                            ("外出していない", "not_outside"),
                            ("答えない", "declined"),
                        ]
                    )
            if kind is None and typing.get("unusual") and recent_reason is None:
                token = "typing:" + typing["evidence_id"]
                if not any(r["payload"].get("eventType") == token for r in issued):
                    kind = "typingReason"
                    event = token
                    prompt = "入力の修正が増えています。主な理由はどれですか？"
                    options = opts(
                        [
                            ("誤入力", "mistype"),
                            ("文章を考え直した", "rewrite"),
                            ("変換が合わない", "conversion"),
                            ("わからない", "unknown"),
                        ]
                    )
            if kind is None:
                # Routine reports are kept alongside event-triggered reports to reduce selection bias.
                today = [r for r in issued if dt(r["at"]).date() == at.date()]
                if at.hour >= 22:
                    kind = "nightReflection"
                    prompt = "今日全体の状態を振り返りますか？"
                    options = opts([("回答を始める", "start_daily_reflection")])
                elif not any(r["payload"]["kind"] == "weatherMood" for r in today):
                    kind = "weatherMood"
                    prompt = "今の気分を天気でいうと？"
                    options = opts(
                        [
                            ("☀️ 晴れ・うれしい", "happy"),
                            ("☁️ くもり・ふつう", "normal"),
                            ("🌧️ 雨・悲しい", "sad"),
                            ("⛈️ 雷・いらだち", "angry"),
                        ]
                    )
                else:
                    kind = "currentState"
                    prompt = "今の状態を教えてください"
                    options = opts([("回答を始める", "start_current_state")])
                    previous = [
                        dt(r["at"])
                        for r in today
                        if r["payload"]["kind"] == "currentState"
                    ]
                    if previous and at - max(previous) < timedelta(hours=3):
                        return None
                if kind == "nightReflection" and any(
                    r["payload"]["kind"] == kind for r in today
                ):
                    return None
                event = "routine_" + kind
            q = dict(
                id=str(uuid4()),
                participantID=p,
                kind=kind,
                prompt=prompt,
                options=options,
                allowFreeText=False,
                eventType=event,
                detectedAt=at.isoformat(timespec="milliseconds"),
                questionShownAt=None,
                snoozeCount=0,
                expiresAt=(at + timedelta(minutes=ttl)).isoformat(
                    timespec="milliseconds"
                ),
                context=dict(
                    time_of_day="night" if at.hour >= 22 else "afternoon",
                    typing_active=False,
                    calendar_busy=False,
                    external_trigger="server-v2",
                ),
            )
            if kind == "supportOutcome":
                q["relatedID"] = event.split(":", 1)[1]
            self.store.save("question_issued", q["id"], p, at, q)
            return q
