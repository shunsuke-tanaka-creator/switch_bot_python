"""Synthetic regression tests for the discovery-only support policy and queue."""
from datetime import timedelta
import json
from types import SimpleNamespace

from lifelog.store import Store
from research.data import dt, SCALES, Timeline
from research.data_modes import is_training_row
from research.discovery_support import (
    PRE, BREAK, OUTCOME, POST, BUCKET, evaluate, decision_with_actions,
    rank_actions, followup, effects,
)
from research.questions import QuestionQueue, settings

AT = dt("2026-11-24T15:00:00+09:00")
P = "p001"


def row(kind, identifier, at, payload):
    return dict(kind=kind, id=identifier, at=at.isoformat(), participant=P, payload=payload)


def self_report(ident, at, values, event="routine_currentState"):
    return row("narrative", ident, at, dict(
        detected_at=at.isoformat(), answered_at=at.isoformat(), event_type=event,
        question=dict(kind="currentState"),
        answer=dict(label="current_state", free_text=json.dumps(values))))


def references(n=3, identity="task_study"):
    rows = []
    for i in range(n):
        t = AT - timedelta(days=i+1, minutes=55)
        rows.append(self_report(f"label{i}", t, dict(fatigue=2, concentration=4, sleepiness=2)))
        rows.append(row("research_event", f"context{i}", t+timedelta(seconds=1),
                        dict(kind="context_label", payload=dict(axis="task", class_id=identity,
                             start=(t-BUCKET).isoformat(), end=t.isoformat()))))
    return rows


def observed_prediction(b, values, model="m1", identity="task_study", mode="real", retrospective=False):
    obs = b + timedelta(seconds=30)
    t = b + timedelta(minutes=2)
    prediction = dict(at=t.isoformat(), available_at=t.isoformat(), architecture="discovery",
                      data_mode=mode, mode="review" if retrospective else "live",
                      contains_sample=mode!="real", status="ready", model_id=model,
                      behavior=dict(status="estimated", id=identity) if identity else dict(status="unknown"),
                      current={s: dict(value=v, status="estimated", source="discovery", provisional=False)
                               for s,v in zip(("fatigue","concentration","sleepiness"), values)})
    return [row("switchbot_observation", "sensor"+b.isoformat(), obs, {}),
            row("inference", "pred"+b.isoformat(), t+timedelta(seconds=1), prediction)]


def records(n=3, identity="task_study", mode="real"):
    rows = references(n)
    for offset in (7,6,1,0):
        b = AT - (offset+1)*BUCKET
        values = (2,4.2,2) if offset>=6 else (3.7,2.7,3.6)
        rows += observed_prediction(b, values, identity=identity, mode=mode, retrospective=mode=="sample")
    return rows


def test_behavior_day_weighting_and_directional_flags():
    rows = records()
    # Extra reports on the same day do not add a fourth day.
    result = evaluate(rows, AT)
    assert result["eligible"]
    assert result["behavior_id"] == "task_study"
    for s in ("fatigue","concentration","sleepiness"):
        item = result["scales"][s]
        assert item["baseline"]["basis"] == "behavior"
        assert item["baseline"]["days"] == 3
        assert item["difference"] > 1
        assert item["z"] >= 2
        assert "relative" in item["flags"]
        assert "trend" in item["flags"]
    assert result["scales"]["concentration"]["value"] == 2.7


def test_unknown_uses_personal_reference_and_sparse_history_suppresses():
    result = evaluate(records(5, identity=None), AT)
    assert result["eligible"]
    assert result["scales"]["fatigue"]["baseline"]["basis"] == "personal"
    assert not evaluate(records(2), AT)["eligible"]
    assert not evaluate(records(3, identity=None), AT)["eligible"]


def test_distinct_buckets_and_model_identity_are_required():
    rows = records()
    missing = [r for r in rows if not (r["kind"] == "switchbot_observation" and
               AT-2*BUCKET <= dt(r["at"]) < AT-BUCKET)]
    assert not evaluate(missing, AT)["eligible"]
    changed = records()
    for r in changed:
        if r["kind"]=="inference" and dt(r["at"]) >= AT-BUCKET:
            r["payload"]["model_id"]="m2"
    assert evaluate(changed, AT)["reason"] == "model_changed"
    result=decision_with_actions(rows,AT,"real",{},"m2")
    assert not result["eligible"] and result["reason"]=="active_model_changed"


def test_no_sample_or_review_predictions_in_real_decision():
    assert not evaluate(records(mode="sample"), AT, "real")["eligible"]
    assert evaluate(records(mode="sample"), AT, "sample")["eligible"]
    assert not evaluate(records(), AT, "combined")["eligible"]
    assert evaluate(records(), AT, config={"discovery_support_mode":"off"})["status"]=="disabled"


def test_missing_scale_and_future_observation_do_not_trigger():
    rows=records()
    for r in rows:
        if r["kind"]=="inference":
            r["payload"]["current"]={"fatigue":dict(value=5,status="insufficient_labels",source="discovery")}
    assert not evaluate(rows,AT)["eligible"]
    rows=records()
    for r in rows:
        if r["kind"]=="switchbot_observation":
            r["payload"]["received_at"]=(AT+BUCKET).isoformat()
    assert not evaluate(rows,AT)["eligible"]


def test_actions_prioritize_supported_dimensions():
    result=evaluate(records(),AT)
    result["scales"]["fatigue"]["value"]=5
    for s in ("concentration","sleepiness"):
        result["scales"][s]["value"]=None
    assert rank_actions([],AT,result)[0]["action"]=="short_break"
    result["scales"]["fatigue"]["value"]=None
    result["scales"]["concentration"]["value"]=1
    assert rank_actions([],AT,result)[0]["action"]=="switch_task"


class Insights:
    def __init__(self, mode):
        self.support_mode=mode
        self.decision=decision_with_actions(records(),AT,"real",{},"m1")
    def snapshot(self,p,at):
        return dict(support=self.decision, fatigue=dict(acute=False,high=False),
                    typing=dict(unusual=False),exercise=dict(status="disabled",low=False),
                    places=dict(anomaly=None,time_anomaly=None))


def setup_queue(mode="live", daily_limit=12):
    store=Store(":memory:")
    store.save("research_event","prefs",P,AT-timedelta(days=1),
               dict(kind="preferences",payload=dict(daily_limit=daily_limit,
                    intervention_limit=2,gap_minutes=60)))
    return store, QuestionQueue(store,Insights(mode))


def answer(store,q,at,value,text=None):
    store.save("narrative",q["id"],P,at,dict(
        question=dict(kind=q["kind"]), event_type=q["eventType"],
        detected_at=q["detectedAt"], answered_at=at.isoformat(),
        answer=dict(label=value,free_text=json.dumps(text) if text is not None else None)))


BEFORE=dict(mood=3,fatigue=4,stress=3,concentration=2,sleepiness=4)
AFTER=dict(mood=4,fatigue=2,stress=2,concentration=4,sleepiness=2)


def test_complete_linked_episode_and_effect():
    store,queue=setup_queue()
    pre=queue.pending(P,AT)
    assert pre["eventType"].startswith(PRE) and pre["kind"]=="currentState"
    assert queue.pending(P,AT+timedelta(seconds=5))["id"]==pre["id"]
    answer(store,pre,AT+timedelta(minutes=1),"current_state",BEFORE)
    offer=queue.pending(P,AT+timedelta(minutes=2))
    assert offer["eventType"]==BREAK+pre["id"]
    action=offer["options"][0]["value"]
    answer(store,offer,AT+timedelta(minutes=3),action)
    assert queue.pending(P,AT+timedelta(minutes=4)) is None
    outcome=queue.pending(P,AT+timedelta(minutes=18))
    assert outcome["eventType"]==OUTCOME+offer["id"] and outcome["relatedID"]==offer["id"]
    outcome_value="outcome:"+action[7:]+":helpful"
    t=AT+timedelta(minutes=19)
    store.save("research_event","executed",P,t,dict(kind="intervention_outcome",
                payload=dict(question_id=offer["id"],action=action[7:],executed=True,helpfulness="helpful")))
    answer(store,outcome,t,outcome_value)
    post=queue.pending(P,AT+timedelta(minutes=20))
    assert post["eventType"]==POST+offer["id"]
    answer(store,post,AT+timedelta(minutes=21),"current_state",AFTER)
    result=dict(effects(store.rows(P),AT+timedelta(minutes=22)))
    assert result[offer["id"]]["delta"]["fatigue"]==-2
    assert result[offer["id"]]["improvement"]==1.6
    assert not followup(store.rows(P),AT+timedelta(minutes=22))[0]
    for r in store.rows(P):
        if r["kind"]=="narrative" and r["id"] in (pre["id"],post["id"]):
            assert not is_training_row(r)
            assert not Timeline([r]).labels
    store.close()


def test_shadow_preview_never_reserves_questions():
    store,queue=setup_queue("shadow")
    rows=store.rows(P)
    selection=queue.preview(P,AT,rows,queue.insights.snapshot(P,AT),rows)
    assert selection["question"]["eventType"].startswith(PRE)
    assert store.rows(P)==rows
    actual=queue.pending(P,AT)
    assert not actual["eventType"].startswith(PRE)
    store.close()


def test_four_slots_and_night_reserve_required():
    store,queue=setup_queue(daily_limit=4)
    actual=queue.pending(P,AT)
    assert not actual["eventType"].startswith(PRE)
    store.close()


def test_partial_pre_and_declined_episode_do_not_offer():
    store,queue=setup_queue()
    pre=queue.pending(P,AT)
    answer(store,pre,AT+timedelta(minutes=1),"current_state",dict(fatigue=4))
    assert queue.pending(P,AT+timedelta(minutes=2)) is None
    assert not list(effects(store.rows(P),AT+timedelta(minutes=2)))
    store.close()


def test_pre_start_is_capped_at_two_even_if_not_accepted():
    store,queue=setup_queue()
    for i in range(2):
        t=AT-timedelta(hours=i+2)
        store.save("question_issued","old"+str(i),P,t,dict(
            kind="currentState",eventType=PRE+t.isoformat(),expiresAt=(t+BUCKET).isoformat()))
    actual=queue.pending(P,AT)
    assert not actual["eventType"].startswith(PRE)
    store.close()


def test_audit_worker_records_once_per_bucket_without_issuing():
    from research.discovery_service import DiscoveryInsights
    store=Store(":memory:")
    for r in records():
        store.save(r["kind"],r["id"],P,dt(r["at"]),r["payload"])
    insights=DiscoveryInsights(SimpleNamespace(config={},store=store))
    prediction=dict(data_mode="real",contains_sample=False,status="ready",model_id="m1")
    insights.record_support(P,AT,prediction)
    insights.record_support(P,AT+timedelta(seconds=1),prediction)
    logs=[r for r in store.rows(P) if r["kind"]=="support_decision"]
    assert len(logs)==1 and logs[0]["payload"]["eligible"]
    assert logs[0]["payload"]["delivery_mode"]=="shadow"
    assert not any(r["kind"]=="question_issued" for r in store.rows(P))
    store.close()
