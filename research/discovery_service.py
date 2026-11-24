"""Opt-in integration: one checkpoint and one trainer for all prediction heads."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import copy
import json
from .data import FEATURES, SCALES, Timeline, dt
from .context_data import (observations, fresh, catalog, annotations, label_at,
                           resolve, cutoffs, transitions)
from .context_service import ContextEngine
from .topology import available_rows
from .discovery_map import DiscoveryMap

VERSION = "life-discovery-v1"


class DiscoveryRuntime:
    def __init__(self, engine):
        self.engine, self.config, self.store = engine, engine.config, engine.store
        self.cache = {}

    def directory(self, participant, mode="real"):
        return self.engine.directory(participant, mode)/"discovery"

    def status(self, participant, mode="real"):
        directory = self.directory(participant, mode)
        path = directory/"status.json"
        result = json.loads(path.read_text()) if path.exists() else dict(status="untrained", version=VERSION)
        error = directory/"error.json"
        if error.exists():
            result["last_error"] = json.loads(error.read_text())
        return dict(result, architecture="discovery", data_mode=mode,
                    contains_sample=mode!="real" or participant==self.engine.sample_participant)

    def load(self, participant, mode="real"):
        path = self.directory(participant, mode)/"active.pt"
        if not path.exists():
            return None
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(key)
        if cached is not None:
            if mode!="real":
                from .data_modes import source_identity
                identity=source_identity(self.store,participant,mode,self.engine.sample_participant,self.engine.sample_manifest)
                if cached[0]["report"].get("sample_manifest_hash")!=identity["sample_manifest_hash"]:
                    return None
            return cached
        import torch
        from .discovery_model import DiscoveryModel
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version")!=VERSION or state.get("features")!=FEATURES:
            return None
        if mode!="real":
            from .data_modes import source_identity
            identity=source_identity(self.store,participant,mode,self.engine.sample_participant,self.engine.sample_manifest)
            if state["report"].get("sample_manifest_hash")!=identity["sample_manifest_hash"]:
                return None
        model = DiscoveryModel(**state["settings"])
        model.load_state_dict(state["weights"]); model.eval()
        maps = {a:DiscoveryMap.from_dict(s) for a,s in state["maps"].items()}
        loaded = (state,model,maps)
        # Return the local snapshot: concurrent training/another participant can
        # replace or clear the bounded cache after this assignment.
        self.cache = {key:loaded}
        return loaded

    def train(self, participant, mode="real", force=False):
        from .discovery_learning import train_discovery
        if self.config.get("research_require_cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required but unavailable")
        from .data_modes import mode_snapshot
        rows,identity=mode_snapshot(self.store,participant,mode,self.engine.sample_participant,self.engine.sample_manifest)
        result = train_discovery(
            rows, self.directory(participant,mode),
            self.engine.preview_at(participant,mode), epochs=self.config.get("discovery_epochs",12),
            hidden=self.config.get("discovery_hidden",48), text_model=self.config.get("text_model"),
            device=self.config.get("context_device","auto"), force=force,
            data_mode=mode, contains_sample=mode!="real" or participant==self.engine.sample_participant,
            sample_manifest_hash=identity["sample_manifest_hash"],
        )
        self.cache.clear()
        loaded = self.load(participant,mode)
        if mode=="real" and loaded:
            for axis,mapping in loaded[2].items():
                wire = "task" if axis=="behavior" else "place"
                for node in mapping.nodes.values():
                    self.store.save("context_candidate",node["id"],participant,dt(node["anchor"]),
                                    dict(axis=wire,class_id=node["id"],anchor=node["anchor"],
                                         received_at=datetime.now(timezone.utc).isoformat(),
                                         architecture="discovery",model_id=loaded[0]["report"]["experiment_id"]))
        error = self.directory(participant,mode)/"error.json"
        if error.exists() and result.get("status")=="trained":
            error.unlink()
        return result

    def infer(self, participant, at=None, mode="real", rows=None, retrospective=False, available_at=None):
        preview_at = getattr(self.engine, "preview_at", None)
        at = dt(preview_at(participant,mode,at) if preview_at is not None else at or datetime.now(timezone.utc))
        known_at = dt(available_at or at)
        rows = self.engine.rows(participant,mode) if rows is None else rows
        if mode=="combined":
            from .data import inference_rows
            rows, sensor_source = inference_rows(rows,known_at,participant)
        result = dict(at=at.isoformat(),available_at=known_at.isoformat(),architecture="discovery",version=VERSION,
                      source="model",status="untrained",current=None,daily=None,comparisons={},activity=None,
                      activity_role="posthoc_only",place=None,behavior=None,task=None,gps_used=False,
                      data_mode=mode,contains_sample=mode!="real" or participant==self.engine.sample_participant,
                      mode="review" if retrospective else "live")
        if mode=="combined":
            result["sensor_source"]=sensor_source
        loaded = self.load(participant,mode)
        if not loaded:
            if (self.directory(participant,mode)/"active.pt").exists():
                result["status"] = "incompatible_checkpoint"
            return result
        state,model,maps = loaded
        report = state["report"]
        result.update(model=report,model_id=report["experiment_id"])
        if not retrospective and dt(report["trained_through"])>at:
            result["status"]="checkpoint_from_future"; return result
        if not fresh(observations(rows),at):
            result["status"]="observation_unknown"; return result
        timeline = Timeline(observations(rows),text_model=self.config.get("text_model") or report.get("text_model"))
        identity = timeline.encoder.identity if timeline.encoder else "character-ngram-v1"
        if identity != report["text_identity"]:
            result["status"]="text_encoder_changed"; return result
        import torch
        from .train import batch
        from .discovery_learning import annotation_digest, source_rows
        with torch.no_grad():
            out = model(batch([timeline.sample(at)],"cpu",torch.tensor(state["mean"]),torch.tensor(state["std"])))
        result["status"]="ready"
        nodes,aliases = catalog(rows,known_at)
        unchanged = report["annotation_digest"]==annotation_digest(source_rows(rows,known_at))
        for a,wire in (("place","place"),("behavior","task")):
            prediction = maps[a].predict(out[a][0].numpy())
            key = prediction["candidate_id"]
            node = maps[a].nodes.get(key,{})
            validation = report["validation_axes"][a]
            accepted = (unchanged and key is not None and node.get("confirmed",False)
                        and len(node.get("confirmed_days",[]))>=2 and validation["n"]>=2
                        and (validation["accuracy"] or 0)>=0.5 and prediction["margin"] is not None
                        and prediction["margin"]>=0.08)
            resolved = resolve(aliases[wire],key) if key else None
            prediction.update(id=resolved if accepted else None,
                              candidate_id=resolved, status="estimated" if accepted else prediction["status"],
                              name=nodes[wire].get(resolved,{}).get("name"),source="model",
                              model_id=report["experiment_id"],note="similarity is not a probability")
            result[a]=prediction
        result["task"]=copy.deepcopy(result["behavior"])
        for scope in ("current","daily"):
            result[scope]={}
            for i,name in enumerate(SCALES):
                valid=report["validation"][scope][name]
                supported=report["support"][scope][i]>=10 and valid["n"]>=3
                result[scope][name]=dict(value=float(out[scope][0,i]) if supported else None,
                                         status="estimated" if supported else "insufficient_labels",
                                         validation=valid,source="discovery",provisional=False,baseline_method=None)
        return result


class DiscoveryContext(ContextEngine):
    def __init__(self, runtime):
        super().__init__(runtime.store,runtime.config)
        self.runtime=runtime

    def directory(self,p,mode="real"):
        return self.runtime.directory(p,mode)

    def load(self,p,mode="real"):
        return self.runtime.load(p,mode)

    def train(self,p,rows,at,mode="real"):
        raise RuntimeError("Discovery uses Engine.train only; no independent context trainer")

    def infer(self,p,at,rows=None,mode="real",retrospective=False,available_at=None):
        rows=self.store.rows(p) if rows is None else rows
        known_at=dt(available_at or at)
        result=self.runtime.infer(p,at,mode,rows,retrospective,available_at)
        nodes,_=catalog(rows,known_at)
        labels=annotations(rows,known_at)
        for wire in ("place","task"):
            label=label_at(labels,wire,dt(at))
            if label:
                key=label["class_id"]
                result[wire]=dict(id=None if key in ("unknown","moving") else key,
                                  status=key if key in ("unknown","moving") else "confirmed",
                                  source="self_report",name=nodes[wire].get(key,{}).get("name"))
        result["behavior"]=copy.deepcopy(result["task"])
        return result

    def snapshot(self,p,at,rows=None,*,include_details=True):
        rows=self.store.rows(p) if rows is None else rows
        compatible=[r for r in rows if r["kind"] not in ("context_inference","context_review")
                    or r["payload"].get("architecture")=="discovery"]
        result=super().snapshot(p,at,compatible,include_details=include_details)
        loaded=self.runtime.load(p)
        result.update(version=VERSION,architecture="discovery",task_role="behavior_wire_alias",
                      activity_role="posthoc_only",discovery_maps={})
        if loaded and dt(loaded[0]["report"]["trained_through"])<=dt(at):
            result["discovery_maps"]={a:m.to_dict() for a,m in loaded[2].items()}
            result["map_model_id"]=loaded[0]["report"]["experiment_id"]
        return result

    def review(self,p,day,at):
        at=dt(at)
        start=dt(day+"T00:00:00+09:00"); end=start+timedelta(days=1)
        if end>at:
            raise ValueError("Review requires a completed JST day")
        rows=self.store.rows(p)
        times=[t for t in cutoffs(rows,end) if start<=t<end]
        times.extend(l["end"] for l in annotations(rows,at) if start<=l["end"]<end)
        episodes=[self.infer(p,t,rows,retrospective=True,available_at=at) for t in sorted(set(times))]
        live=[r["payload"] for r in reversed(rows) if r["kind"]=="context_inference"
              and r["payload"].get("architecture")=="discovery" and start<=dt(r["at"])<end]
        result=dict(day=day,created_at=at.isoformat(),architecture="discovery",version=VERSION,
                    episodes=episodes,original_live=live,edges=transitions(episodes),
                    note="Retrospective reanalysis; not an online accuracy evaluation")
        self.store.save("context_review",str(uuid4()),p,at,result)
        return result


class DiscoveryInsights:
    """Use the learned behavior map, not the legacy raw-feature clustering."""
    def __init__(self,runtime):
        self.runtime=runtime
        self.trend_cache={}
        from .discovery_support import support_mode, settings
        self.support_mode=support_mode(runtime.config)
        settings(runtime.config)

    def record_support(self, p, at, prediction):
        """Worker-only real-data audit, once per completed five-minute bucket."""
        from .discovery_support import bucket, BUCKET, decision_with_actions, effects
        from .sample_registry import is_sample_participant
        if self.support_mode == "off" or is_sample_participant(p):
            return
        if prediction.get("data_mode") != "real" or prediction.get("contains_sample", True):
            return
        at=dt(at)
        key=(bucket(at)-BUCKET).isoformat()
        store=self.runtime.store
        rows=store.rows(p)
        if any(r["kind"]=="support_decision" and r["id"]==key for r in rows):
            return
        active=prediction.get("model_id") if prediction.get("status")=="ready" else None
        decision=decision_with_actions(rows, at, "real", self.runtime.config, active)
        store.save("support_decision", key, p, at, dict(decision, evaluated_at=at.isoformat()))
        for offer, effect in effects(rows, at):
            store.save("support_effect", offer, p, at, effect)

    def snapshot(self,p,at,prediction=None,rows=None,data_mode="real"):
        from .baseline import conditional, weekday_trends
        from .topology import typing_windows
        from .insights import typing_change
        at=dt(at)
        rows=self.runtime.engine.rows(p,data_mode) if rows is None else rows
        source=available_rows(rows,at)
        key=(p,data_mode,at.date().isoformat())
        trends=self.trend_cache.get(key)
        if trends is None:
            trends=weekday_trends(rows,at)
            self.trend_cache={key:trends}
        patterns=dict(nodes=[],edges=[],current=None,latest=None,aliases={},
                      status="untrained",version=VERSION,at_capacity=False)
        loaded=self.runtime.load(p,data_mode)
        if loaded and dt(loaded[0]["report"]["trained_through"])<=at:
            mapping=loaded[2]["behavior"].to_dict()
            patterns.update(
                nodes=[dict(id=n["id"],count=n["count"],days=len(n["days"]),
                            status="confirmed" if n["confirmed"] else "emerging") for n in mapping["nodes"]],
                edges=mapping["edges"],status="training_map",at_capacity=mapping["overflow"]>0,
                model_id=loaded[0]["report"]["experiment_id"])
        from .discovery_support import decision_with_actions, preview_sample_rows
        support_rows=rows
        if data_mode=="sample" and loaded and self.support_mode!="off":
            support_rows=preview_sample_rows(self.runtime,p,rows,at)
        active=loaded[0]["report"]["experiment_id"] if loaded else None
        if data_mode=="real" and not fresh(observations(source),at):
            active=None
        support=decision_with_actions(support_rows,at,data_mode,self.runtime.config,active)
        support["retrospective_preview"] = data_mode=="sample"
        return dict(version=VERSION,architecture="discovery",at=at.isoformat(),patterns=patterns,
                    support=support,
                    baseline=conditional(rows,at),trends=trends,
                    typing=typing_change(typing_windows(rows,at),at),
                    places=dict(nodes=[],edges=[],current=None,latest=None,
                                destination=dict(status="gps_disabled",places=[]),anomaly=None,time_anomaly=None),
                    weather=[dict(at=r["at"],weather=r["payload"]["payload"]["weather"],source="self_report")
                             for r in source if r["kind"]=="research_event" and r["payload"].get("kind")=="weather_mood"][-200:],
                    exercise=dict(status="disabled",low=False,reason="discovery_state_baseline_not_validated"),
                    fatigue=dict(status="insufficient_data",acute=False,high=False,
                                 reason="discovery_delivery_not_validated"))
