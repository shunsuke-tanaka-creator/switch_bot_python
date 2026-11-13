"""One durable shared queue/budget for keyboard, parent app and local notifications."""

from datetime import timedelta
from uuid import uuid4
from .data import dt
from .insights import recommend


def settings(rows):
    value = dict(
        questions_enabled=True,
        interventions_enabled=True,
        # 追加: 歩数同意とは独立した明示設定で、既定は無効。
        exercise_proposals_enabled=False,
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

    def _candidate(self, p, rows, insight, pref, at, issued_history_rows, persist):
        """Shared selector for issuing and read-only previews.

        Trigger and answer evidence comes from ``rows`` (the requested mode).
        Pending state, budgets, gaps, intervention counts, and duplicate suppression
        come from ``issued_history_rows`` (always the real participant history).
        """
        at = dt(at)
        gates = {
            name: dict(passed=False, detail="not evaluated")
            for name in (
                "questions_enabled",
                "active_hours",
                "calendar_suppressed",
                "already_pending",
                "daily_budget",
                "gap",
                "interventions_enabled",
                "intervention_limit",
                "exercise_proposals_enabled",
                "exercise_data",
                "exercise_cooldown",
                "fatigue_high",
                "fatigue_acute",
                "typing_reason_chain",
                "place_anomaly",
                "followup_outcome",
                "scheduled_kinds",
                "scheduled_weather_mood",
                "scheduled_current_state",
                "scheduled_night_reflection",
            )
        }

        def gate(name, passed, detail):
            gates[name] = dict(passed=bool(passed), detail=detail)
            return passed

        def result(question):
            return dict(question=question, gates=gates)

        # GET is an atomic reservation. Different surfaces reuse the same pending ID.
        with self.store.lock:
            rows = [
                r
                for r in rows
                if dt(r["at"]) <= at
                and dt(r["payload"].get("received_at", r["at"])) <= at
            ]
            history_rows = [
                r
                for r in issued_history_rows
                if dt(r["at"]) <= at
                and dt(r["payload"].get("received_at", r["at"])) <= at
            ]
            if not gate(
                "questions_enabled",
                pref["questions_enabled"],
                "enabled" if pref["questions_enabled"] else "disabled by preferences",
            ):
                return result(None)
            if not gate(
                "active_hours",
                8 <= at.hour < 24,
                f"hour={at.hour}; active range is 08:00-24:00",
            ):
                return result(None)
            session = next((r for r in rows if r["kind"] == "session"), None)
            context = (session["payload"].get("context") or {}) if session else {}
            calendar_at = (context.get("measurement_updated_at") or {}).get(
                "calendar_events"
            )
            calendar_busy = False
            if calendar_at and timedelta(0) <= at - dt(calendar_at) <= timedelta(
                hours=2
            ):
                calendar_busy = any(
                    dt(e["start"]) <= at < dt(e["end"])
                    for e in context.get("calendar_events") or []
                )
            if not gate(
                "calendar_suppressed",
                not calendar_busy,
                "active recent calendar event" if calendar_busy else "not suppressed",
            ):
                return result(None)
            issued = [r for r in history_rows if r["kind"] == "question_issued"]
            answered = {r["id"] for r in history_rows if r["kind"] == "narrative"}
            states = {}
            weather_q = set()
            reason_q = set()
            for r in reversed(history_rows):
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
            context_q={r["payload"]["payload"].get("question_id") for r in history_rows
                       if r["kind"]=="research_event" and r["payload"]["kind"]=="context_label"}
            done = answered | weather_q | reason_q | context_q

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
                    r["payload"]["kind"] in ("breakCheck", "exerciseProposal")
                    and not pref["interventions_enabled"]
                ):
                    continue
                if (
                    r["payload"]["kind"] == "exerciseProposal"
                    and not pref["exercise_proposals_enabled"]
                ):
                    continue
                if states.get(r["id"], ("",))[0] == "deferred":
                    if at - states[r["id"]][1] < timedelta(minutes=30):
                        gate("already_pending", False, "pending question is deferred")
                        return result(None)
                gate("already_pending", False, f"reusing pending {r['payload']['kind']}")
                return result(r["payload"])
            gate("already_pending", True, "none")
            # Actual display counts plus live reservations bound cross-surface races.
            shown = {
                r["payload"]["payload"]["question_id"]: dt(r["at"])
                for r in history_rows
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
            if not gate(
                "daily_budget",
                len(budget) < pref["daily_limit"],
                f"{len(budget)}/{pref['daily_limit']} used",
            ):
                return result(None)
            # Keep one assessment opportunity for the daily head at night.
            if (
                at.hour < 22
                and pref["daily_limit"] > 1
                and len(budget) >= pref["daily_limit"] - 1
            ):
                gates["daily_budget"]["passed"] = False
                gates["daily_budget"]["detail"] += "; reserved night slot"
                return result(None)
            recent = [dt(r["at"]) for r in issued if dt(r["at"]).date() == at.date()]
            gap_ok = not recent or at - max(recent) >= timedelta(
                minutes=pref["gap_minutes"]
            )
            if not gate(
                "gap",
                gap_ok,
                "no issue today"
                if not recent
                else f"{int((at-max(recent)).total_seconds()//60)}m since last; requires {pref['gap_minutes']}m",
            ):
                return result(None)
            if insight is None:
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
                if r["payload"]["kind"] in ("breakCheck", "exerciseProposal")
                and dt(r["at"]).date() == at.date()
            ]
            gate(
                "interventions_enabled",
                pref["interventions_enabled"],
                "enabled" if pref["interventions_enabled"] else "disabled by preferences",
            )
            gate(
                "intervention_limit",
                len(offered) < pref["intervention_limit"],
                f"{len(offered)}/{pref['intervention_limit']} offered today",
            )
            gate(
                "fatigue_high",
                bool(fatigue.get("high")),
                "high fatigue detected" if fatigue.get("high") else "not detected",
            )
            gate(
                "fatigue_acute",
                bool(fatigue.get("acute")),
                "acute fatigue detected" if fatigue.get("acute") else "not detected",
            )
            gate(
                "typing_reason_chain",
                bool(typing.get("unusual") and recent_reason == "mistype"),
                f"typing unusual={bool(typing.get('unusual'))}; recent reason={recent_reason or 'none'}",
            )
            kind = prompt = event = None
            pmtt = None
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
            gate(
                "followup_outcome",
                kind == "supportOutcome",
                "eligible accepted breakCheck found"
                if kind == "supportOutcome"
                else "no eligible accepted breakCheck",
            )
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
            # 追加: 状態条件付き歩数不足を同じ提案上限・停止・延期基盤へ統合する。
            exercise = insight.get("exercise") or {}
            exercise_ready = exercise.get("status") == "available" and exercise.get(
                "low"
            )
            exercise_context = exercise.get("context") or {}
            exercise_token = (
                "exercise:"
                + at.date().isoformat()
                + ":"
                + (exercise_context.get("state_id") or "unknown")
            )
            exercise_duplicate = any(
                r["payload"]["kind"] == "exerciseProposal"
                and r["payload"].get("eventType") == exercise_token
                for r in issued
            )
            gate(
                "exercise_proposals_enabled",
                pref["exercise_proposals_enabled"],
                "enabled"
                if pref["exercise_proposals_enabled"]
                else "disabled by preferences",
            )
            gate(
                "exercise_data",
                exercise_ready,
                (
                    f"status={exercise.get('status')}; low={bool(exercise.get('low'))}; "
                    f"reason={exercise.get('reason', 'none')}"
                ),
            )
            gate(
                "exercise_cooldown",
                not exercise_duplicate,
                "same-day state already offered" if exercise_duplicate else "clear",
            )
            if (
                kind is None
                and pref["interventions_enabled"]
                and pref["exercise_proposals_enabled"]
                and len(offered) < pref["intervention_limit"]
                and exercise_ready
                and not exercise_duplicate
            ):
                event = exercise_token
                kind = "exerciseProposal"
                place = exercise_context.get("place_name") or exercise_context.get(
                    "place_id"
                )
                action = exercise_context.get("action_name") or exercise_context.get(
                    "action_id"
                )
                condition = f"「{place or 'この場所'}・{action or 'この行動'}」"
                prompt = (
                    f"{condition}では普段より歩数が"
                    f"{exercise['shortage_steps']}歩少なめです。"
                    f"100歩/分の歩行相当で約{exercise['walking_equivalent_minutes']:g}分です。"
                    "運動候補を選びますか？"
                )
                options = opts(
                    [
                        (
                            f"{candidate['label']} {candidate['minutes']}分"
                            f"（{candidate['mets']:.1f} METs）",
                            "accept:exercise:" + candidate["action"],
                        )
                        for candidate in exercise["candidates"]
                    ]
                    + [("今はしない", "declined"), ("提案を停止", "stop")]
                )
                pmtt = dict(
                    state=exercise_context,
                    usual_steps=exercise["usual_steps"],
                    lower_quartile_steps=exercise["lower_quartile_steps"],
                    today_steps=exercise["today_steps"],
                    shortage_steps=exercise["shortage_steps"],
                    minimum_shortage_steps=exercise["minimum_shortage_steps"],
                    minimum_shortage_fraction=exercise[
                        "minimum_shortage_fraction"
                    ],
                    cadence_steps_per_minute=exercise["cadence_steps_per_minute"],
                    walking_equivalent_minutes=exercise[
                        "walking_equivalent_minutes"
                    ],
                    baseline_days=exercise["baseline_days"],
                    baseline_occurrences=exercise["baseline_occurrences"],
                    baseline_stage=exercise["baseline_stage"],
                    provisional=exercise["provisional"],
                    candidates=exercise["candidates"],
                    evidence_id=exercise["evidence_id"],
                    carry_evidence_id=exercise["carry_evidence_id"],
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
            # Four default opportunities: context, weather, current, night.
            # Event-triggered proposals remain available with an explicitly larger budget.
            if pref['daily_limit'] <= 4 or at.hour >= 22:
                kind = None
            # One shared budget/gap for all surfaces. Alternate axes across days.
            if kind is None and at.hour < 22:
                from .context_data import catalog, fresh, summary, ident
                if fresh(rows,at):
                    nodes,_=catalog(rows,at)
                    context_history=[r for r in issued if r["payload"]["kind"] in ("placeCheck","taskCheck")]
                    axis="task" if context_history and context_history[0]["payload"]["kind"]=="placeCheck" else "place"
                    previous=[r for r in context_history if dt(r["at"]).date()==at.date()]
                    if len(previous)<1:
                        kind="placeCheck" if axis=="place" else "taskCheck"
                        prompt="今いる拠点はどれですか？ 作業が変わっただけなら同じ拠点です。" if axis=="place" else "今のタスクはどれですか？ 拠点が変わっても同じ作業なら同じタスクです。"
                        event="context:"+axis
                        noun="拠点" if axis=="place" else "タスク"
                        current=summary(rows,at)['current'].get(axis) or {}
                        candidate=current.get('candidate_id') or current.get('id')
                        ordered=sorted(nodes[axis].values(),key=lambda n:(n['id']!=candidate,not n['confirmed'],n['id']))
                        options=opts([(n['name'] or noun+' '+n['id'][-5:],n['id']) for n in ordered[:4]])
                        options+=opts([('新しい'+noun,ident(axis,at.isoformat())),('わからない','unknown')])
                        if axis=='place':options+=opts([('移動中','moving')])
                        ttl=30
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
                    if previous and (pref['daily_limit'] <= 4 or at - max(previous) < timedelta(hours=3)):
                        gate("scheduled_current_state", False, "issued within 3h")
                        return result(None)
                if kind == "nightReflection" and any(
                    r["payload"]["kind"] == kind for r in today
                ):
                    gate(
                        "scheduled_night_reflection",
                        False,
                        "already issued today",
                    )
                    return result(None)
                event = "routine_" + kind
            gate(
                "scheduled_kinds",
                kind in ("weatherMood", "currentState", "nightReflection"),
                f"selected {kind}"
                if kind in ("weatherMood", "currentState", "nightReflection")
                else "event-triggered candidate selected",
            )
            gate(
                "scheduled_weather_mood",
                kind == "weatherMood",
                "selected" if kind == "weatherMood" else "not selected",
            )
            gate(
                "scheduled_current_state",
                kind == "currentState",
                "selected" if kind == "currentState" else "not selected",
            )
            gate(
                "scheduled_night_reflection",
                kind == "nightReflection",
                "selected" if kind == "nightReflection" else "not selected",
            )
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
            if pmtt is not None:
                q["pmtt"] = pmtt
            if persist:
                self.store.save("question_issued", q["id"], p, at, q)
            return result(q)

    def pending(self, p, at):
        at = dt(at)
        with self.store.lock:
            rows = self.store.rows(p)
            visible = [
                r
                for r in rows
                if dt(r["at"]) <= at
                and dt(r["payload"].get("received_at", r["at"])) <= at
            ]
            selected = self._candidate(
                p,
                rows,
                None,
                settings(visible),
                at,
                rows,
                True,
            )
            return selected["question"]

    def preview(self, p, at, rows, insight, real_rows):
        """Return selection and explanatory gates without saving any record."""
        at = dt(at)
        visible_real = [
            r
            for r in real_rows
            if dt(r["at"]) <= at
            and dt(r["payload"].get("received_at", r["at"])) <= at
        ]
        return self._candidate(
            p, rows, insight, settings(visible_real), at, real_rows, False
        )
