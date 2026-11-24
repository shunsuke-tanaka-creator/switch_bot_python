"""API integration; runs on Mac without torch, on Ubuntu with optional ML worker."""
import asyncio
from collections import Counter
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
from uuid import uuid4
import httpx
from fastapi import HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from .data import SCALES, ACTIVITIES, Timeline, dt, monday
from .data_modes import (
    DATA_MODES,
    DEFAULT_SAMPLE_PARTICIPANT,
    data_mode,
    participants_for_mode,
    resolved_sample_participant,
    sample_manifest_identity,
)
from .api_cache import RecordView, ResearchCache, row_order
from .events import ResearchEvent
from .context_service import ContextEngine
from .insights import InsightEngine
from .questions import QuestionQueue, settings as question_settings


def utc(): return datetime.now(timezone.utc)
def iso(t): return t.isoformat(timespec='milliseconds')
def identifier(p):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',p):raise HTTPException(422,'Invalid participant')
    return p
def mode(value):
    try:return data_mode(value)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


INSIGHT_KINDS=('session','narrative','correction','research_event','context_inference',
               'step_snapshot','state_step_snapshot','inference','switchbot','switchbot_observation',
               'question_issued')  # 追加: discoveryの有用性補正に必要な質問履歴を同じsnapshotへ含める。


class Heartbeat(BaseModel):
    event_id: str = Field(min_length=1,max_length=200)
    participant_id: str
    occurred_at: datetime
    source: str = Field(max_length=50)
    logging_enabled: bool | None = None
    questions_enabled: bool = True
    interventions_enabled: bool = True
    @model_validator(mode='after')
    def validate_time(self):
        identifier(self.participant_id)
        if self.occurred_at.tzinfo is None or self.occurred_at>utc()+timedelta(minutes=5):raise ValueError('Invalid time')
        return self


class Correction(BaseModel):
    event_id: str=Field(min_length=1,max_length=200)
    start: datetime
    end: datetime
    activity: str | None=None
    scores: dict[str,int] | None=None
    scope: str='current'
    @model_validator(mode='after')
    def valid(self):
        if self.start.tzinfo is None or self.end.tzinfo is None or not self.start<self.end<=utc():raise ValueError('Invalid interval')
        if self.activity is not None and self.activity not in ACTIVITIES:raise ValueError('Unknown activity')
        if self.scope not in ('current','daily'):raise ValueError('Unknown scope')
        if self.scores and any(k not in SCALES or not 1<=v<=5 for k,v in self.scores.items()):raise ValueError('Invalid scores')
        if not self.scores and not self.activity:raise ValueError('Empty correction')
        return self


class Feedback(BaseModel):
    id: str=Field(min_length=1,max_length=200)
    outcome: str
    @model_validator(mode='after')
    def valid(self):
        if self.outcome not in ('accepted','declined','deferred','stop','resume'):raise ValueError('Invalid outcome')
        return self


class TrainRequest(BaseModel):
    data_mode: str = "real"
    force: bool = False
    @model_validator(mode='after')
    def valid(self):
        data_mode(self.data_mode)
        return self


def export_page(store,p,after=0,snapshot=None,limit=500):
    with store.lock:
        maximum=store.db.execute('SELECT COALESCE(MAX(seq),0) FROM records WHERE participant=?',(p,)).fetchone()[0]
        snapshot=maximum if snapshot is None else min(snapshot,maximum)
        rows=store.db.execute('SELECT * FROM records WHERE participant=? AND seq>? AND seq<=? ORDER BY seq LIMIT ?', (p,after,snapshot,limit)).fetchall()
        records=[dict(r,payload=json.loads(r['payload'])) for r in rows]
        devices=[dict(r) for r in store.db.execute('SELECT * FROM devices WHERE participant=?',(p,)).fetchall()]
        names=[dict(r) for r in store.db.execute('SELECT * FROM device_names WHERE participant=?',(p,)).fetchall()]
    cursor=records[-1]['seq'] if records else after
    return dict(format='lifelog-all-v1',participant=p,snapshot=snapshot,records=records,devices=devices,device_names=names,
                next_cursor=cursor if cursor<snapshot else None,generated_at=iso(utc()))


def ingest_page(store,page):
    if page.get('format')!='lifelog-all-v1':raise ValueError('Unknown export format')
    p=identifier(page['participant']);inserted=0
    for r in page['records']:
        if r['participant']!=p:raise ValueError('Participant mismatch')
        inserted+=store.save(r['kind'],r['id'],p,dt(r['at']),r['payload'])
    with store.lock,store.db:
        for table in ('devices','device_names'):
            for r in page.get(table,[]):
                if r['participant']!=p:raise ValueError('Participant mismatch')
                cols=('participant','id','payload') if table=='devices' else ('participant','id','name')
                store.db.execute(f'INSERT OR REPLACE INTO {table} VALUES(?,?,?)',tuple(r[c] for c in cols))
    return inserted


def pmtt_projection(graph):
    """Keep nodes with full-graph degree >= 2 and edges between them for display."""
    degree = Counter()
    for edge in graph["edges"]:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    # Compute degrees once: retained nodes may have degree <= 1 after projection.
    nodes = [node for node in graph["nodes"] if degree[node["id"]] >= 2]
    retained = {node["id"] for node in nodes}
    edges = [
        edge for edge in graph["edges"]
        if edge["source"] in retained and edge["target"] in retained
    ]
    return dict(nodes=nodes, edges=edges)


class Engine:
    def __init__(self,store,config):
        self.store=store;self.config=config
        self.api_cache=ResearchCache(store)
        self.root=Path(config.get('research_models','data/models'));self.root.mkdir(parents=True,exist_ok=True)
        self.sample_participant=config.get('research_sample_participant',DEFAULT_SAMPLE_PARTICIPANT)
        self.sample_manifest=Path(config.get('research_sample_manifest',f'data/sample/{self.sample_participant}/manifest.json'))
        self.architecture=config.get('research_architecture','legacy')
        if self.architecture not in ('legacy','discovery'):
            raise ValueError('research_architecture must be legacy or discovery')
        self.discovery=None
        if self.architecture=='discovery':
            from .discovery_service import DiscoveryRuntime, DiscoveryContext, DiscoveryInsights
            self.discovery=DiscoveryRuntime(self)
            self.context=DiscoveryContext(self.discovery)
            self.insights=DiscoveryInsights(self.discovery)
        else:
            self.context=ContextEngine(store,config)
            self.insights=InsightEngine(store,config)
        self.questions=QuestionQueue(store,self.insights)

    def record_view(self,p,data_mode_='real',kinds=None):
        p=identifier(p);data_mode_=data_mode(data_mode_)
        selected=(resolved_sample_participant(self.store,p,self.sample_participant)
                  if data_mode_!='real' else self.sample_participant)
        sources=participants_for_mode(p,data_mode_,selected)
        views=[self.api_cache.records(source,kinds) for source in sources]
        return self._merge_views(p,data_mode_,kinds,dict(zip(sources,views)))

    # 追加: 同一snapshotから取得したviewにも既存の結合キャッシュを共用する。
    def _merge_views(self,p,data_mode_,kinds,views):
        if len(views)==1:return next(iter(views.values()))
        from heapq import merge
        signature=tuple((source,view.fingerprint) for source,view in views.items())
        return self.api_cache.compute(
            ('rows',p,data_mode_,tuple(sorted(kinds)) if kinds is not None else None),
            signature,
            lambda:RecordView(list(merge(*(v.rows for v in views.values()),key=row_order,reverse=True)),signature),
        )

    def rows(self,p,data_mode_='real'):
        # Snapshots are read-only; copy an individual payload before enriching it.
        return self.record_view(p,data_mode_).rows

    def insight_snapshot(self,p,at,prediction,data_mode_):
        selected=(resolved_sample_participant(self.store,p,self.sample_participant)
                  if data_mode_!='real' else self.sample_participant)
        fingerprints={}
        # 追加: combinedのheartbeatを維持し、履歴・検出器・回答を同じsnapshotから取り出す。
        history_kinds=None if data_mode_=='combined' else INSIGHT_KINDS
        histories={}
        kinds=('session','narrative','correction','research_event',
               'context_inference','step_snapshot','state_step_snapshot')
        for owner in participants_for_mode(p,data_mode_,selected):
            history,view,labels=self.api_cache.record_views(
                owner,(history_kinds,kinds,('narrative','correction')),
            )
            histories[owner]=history
            fingerprints[owner]=( (view.fingerprint,view.time_key(at)),
                                  (labels.fingerprint,labels.time_key(at)) )
        # Sample mode uses the requesting participant as its existing cache slot.
        if data_mode_=='sample':fingerprints[p]=fingerprints[selected]
        records=self._merge_views(p,data_mode_,history_kinds,histories).rows
        return self.insights.snapshot(p,at,prediction,records,data_mode_,fingerprints=fingerprints)

    def file_revision(self,path):
        path=Path(path)
        try:
            stat=path.stat()
            return str(path),stat.st_mtime_ns,stat.st_ctime_ns,stat.st_size,stat.st_ino
        except FileNotFoundError:
            return str(path),None

    def context_snapshot(self,p,at,*,include_details=True):
        kinds=('context_candidate','research_event','context_inference','context_review')
        view=self.record_view(p,'real',kinds)
        model_time=None
        if self.discovery is not None:
            loaded=self.discovery.load(p)
            model_time=bool(loaded and dt(loaded[0]['report']['trained_through'])<=dt(at))
        signature=(view.fingerprint,view.time_key(at),model_time,
                   self.file_revision(self.context.directory(p)/'status.json'),
                   self.file_revision(self.context.directory(p)/'active.pt'))
        result=self.api_cache.compute(
            ('context',p),signature,lambda:self.context.snapshot(p,at,view.rows),
        )
        return {k:v for k,v in result.items()
                if include_details or k not in ('history','reviews')}
    def preview_at(self,p,data_mode_='real',at=None):
        if at is not None or data_mode_=='real':
            return dt(at or utc())
        from .sample_registry import active_manifest
        manifest=active_manifest(self.store,p)
        if manifest and manifest.get('preview_at'):
            if data_mode_=='sample':
                return dt(manifest['preview_at'])
            observed=[dt(r['payload'].get('received_at',r['at'])) for r in self.record_view(p,data_mode_,('session',)).rows
                      if r['kind']=='session' and not r['participant'].endswith('-sample')]
            return max([dt(manifest['preview_at']),*observed])
        return dt(utc())
    def directory(self,p,data_mode_='real'):
        return self.root/identifier(p)/data_mode(data_mode_)
    def legacy_directory(self,p):return self.root/identifier(p)
    def read_path(self,p,data_mode_,name):
        path=self.directory(p,data_mode_)/name
        if data_mode_=='real' and not path.exists():
            legacy=self.legacy_directory(p)/name
            if legacy.exists():return legacy
        return path

    def status(self,p,data_mode_='real'):
        data_mode_=data_mode(data_mode_)
        if self.discovery is not None:
            return self.discovery.status(p,data_mode_)
        path=self.read_path(p,data_mode_,'status.json')
        result=json.loads(path.read_text()) if path.exists() else dict(status='untrained')
        result.setdefault('data_mode',data_mode_)
        result.setdefault('contains_sample',data_mode_!='real' or p==self.sample_participant)
        if data_mode_!='real':
            from .sample_registry import active_generation
            selected=active_generation(self.store,p,self.sample_participant)
            result['sample_generation']=selected
            result['preview_support']=True
            manifest_hash=sample_manifest_identity(self.store,data_mode_,selected,self.sample_manifest)
            if result.get('sample_manifest_hash')!=manifest_hash and result.get('status')=='trained':
                result['status']='generation_changed'
        error=self.read_path(p,data_mode_,'error.json')
        if error.exists():result['last_error']=json.loads(error.read_text())
        return result

    def infer(self,p,at=None,data_mode_='real'):
        if self.discovery is not None:
            return self.discovery.infer(identifier(p),at,data_mode(data_mode_))
        from .inference import infer_state
        return infer_state(self,p,at,data_mode_)

    def train(self,p,data_mode_='real',force=False):
        if self.discovery is not None:
            return self.discovery.train(identifier(p),data_mode(data_mode_),force)
        result=self._train(p,data_mode_,force)
        from .delivery import evaluate_delivery
        evaluate_delivery(self,p,data_mode_)
        return result

    def _train(self,p,data_mode_='real',force=False):
        data_mode_=data_mode(data_mode_)
        from .train import train
        if self.config.get('research_require_cuda'):
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check the GPU driver and CUDA-enabled PyTorch')
        from .data_modes import mode_snapshot
        rows,identity=mode_snapshot(self.store,p,data_mode_,self.sample_participant,self.sample_manifest)
        count=len(Timeline(rows).labels)
        status=self.status(p,data_mode_)
        from .provenance import experiment_id
        from .text import encoder_identity
        from .data import VERSION
        text_path = self.config.get('text_model')
        run_id = experiment_id(identity['fingerprint'], dict(version=VERSION,epochs=self.config.get('research_epochs',20),text_identity=encoder_identity(text_path) if text_path else 'character-ngram-v1'))
        if not force and status.get('experiment_id')==run_id and (self.directory(p,data_mode_)/'active.pt').exists():
            return dict(status='no_new_labels',data_mode=data_mode_,contains_sample=data_mode_!='real' or p==self.sample_participant,
                        fingerprint=identity['fingerprint'],source_max_seq=identity['source_max_seq'],
                        sample_manifest_hash=identity['sample_manifest_hash'])
        result=train(rows,self.directory(p,data_mode_),epochs=self.config.get('research_epochs',20),text_model=self.config.get('text_model'),
                     data_mode=data_mode_,contains_sample=data_mode_!='real' or p==self.sample_participant,
                     source_fingerprint=identity['fingerprint'],source_max_seq=identity['source_max_seq'],
                     sample_manifest_hash=identity['sample_manifest_hash'])
        result['experiment_id']=run_id
        result['training_record_count']=count
        error=self.directory(p,data_mode_)/'error.json'
        if error.exists():error.unlink()
        from .io import atomic_json
        atomic_json(self.directory(p,data_mode_)/'status.json',result)
        return result

    def pending(self,p,at=None):
        from .sample_registry import is_sample_participant
        if is_sample_participant(p):return None
        return self.questions.pending(p,dt(at or utc()))

    def proposal_preview(self,p,data_mode_='real',at=None):
        """Preview queue selection without reserving or consuming any budget.

        Mode rows drive insights and answer-dependent trigger chains, while the
        participant's real rows exclusively drive pending/budget/gap queue state.
        """
        data_mode_=data_mode(data_mode_);at=self.preview_at(p,data_mode_,at)
        records=self.rows(p,data_mode_)
        prediction=None if data_mode_=='real' else self.infer(p,at,data_mode_)
        insight=self.insights.snapshot(p,at,prediction,records,data_mode_)
        selected=self.questions.preview(p,at,records,insight,self.rows(p,'real'))
        # 追加: 既存preview契約を維持し、歩数検出器の段階・閾値・抑止理由を確認可能にする。
        return dict(preview=True,participant=p,data_mode=data_mode_,
                    contains_sample=data_mode_!='real' or p==self.sample_participant,
                    at=iso(at),question=selected['question'],gates=selected['gates'],
                    exercise=insight.get('exercise'),support=insight.get('support'))

    def pmtt(self,p,data_mode_='real'):
        kinds=('session','narrative','switchbot','correction','inference',
               'step_snapshot','state_step_snapshot','question_issued',
               'context_candidate','context_inference','research_event')
        view=self.record_view(p,data_mode_,kinds)
        at=self.preview_at(p,data_mode_)
        selected=(resolved_sample_participant(self.store,p,self.sample_participant)
                  if data_mode_!='real' else self.sample_participant)
        newest=[(self.api_cache.newest(owner),owner)
                for owner in participants_for_mode(p,data_mode_,selected)]
        owner_order=tuple(owner for latest,owner in sorted(
            ((latest,owner) for latest,owner in newest if latest is not None), reverse=True))
        return self.api_cache.compute(
            ('pmtt',p,data_mode_),(view.fingerprint,view.time_key(at),owner_order),
            lambda:self._pmtt(p,data_mode_,view.rows,at,owner_order),
        )

    def _pmtt(self,p,data_mode_,rows,at,owner_order=None):
        # Explicit provenance graph: observations, work episodes, days, fixed weeks, and corrections.
        nodes={};edges=[]
        for r in rows:
            if r['kind'] not in {'session','narrative','switchbot','correction','inference','step_snapshot','state_step_snapshot'}:continue
            prefix=r['participant']+':' if data_mode_!='real' else ''
            provenance=(dict(participant=r['participant'],data_source='synthetic' if r['participant'].endswith('-sample') else 'real')
                        if data_mode_!='real' else {})
            observed_at=dt(r['at']);week=prefix+'week:'+monday(observed_at).date().isoformat();day=prefix+'day:'+observed_at.date().isoformat();work=prefix+'work:'+observed_at.replace(minute=0,second=0,microsecond=0).isoformat()
            for key,kind in [(week,'Pattern'),(day,'Day'),(work,'StateEpisode')]:nodes[key]=dict(id=key,kind=kind,**provenance)
            key=prefix+r['kind']+':'+r['id'];nodes[key]=dict(id=key,kind='Observation',source=r['kind'],at=r['at'],**provenance)
            edges.extend([dict(source=key,target=work,kind='provenance'),dict(source=work,target=day,kind='part_of'),dict(source=day,target=week,kind='part_of')])
            if r['kind']=='correction':nodes[key]['interval']=[r['payload']['start'],r['payload']['end']]
        # 追加: 運動提案の状態・通常歩数・不足・METs候補をPMTTへ残す。
        for r in rows:
            if r['kind']!='question_issued' or r['payload'].get('kind')!='exerciseProposal':continue
            prefix=r['participant']+':' if data_mode_!='real' else ''
            key=prefix+'proposal:'+r['id'];evidence=r['payload']['pmtt']
            nodes[key]=dict(id=key,kind='ExerciseProposal',at=r['at'],evidence=evidence)
            source=prefix+'state_step_snapshot:'+evidence['evidence_id']
            if source not in nodes:source=prefix+'session:'+evidence['evidence_id']
            if source in nodes:edges.append(dict(source=source,target=key,kind='supports'))
        from .context_data import catalog
        from .context_data import resolve
        groups={owner:[] for owner in owner_order or ()}
        for r in rows:groups.setdefault(r['participant'],[]).append(r)
        for owner,group in groups.items():
            classes,aliases=catalog(group,at)
            prefix=owner+':' if data_mode_!='real' else ''
            provenance=dict(participant=owner,data_source='synthetic' if owner.endswith('-sample') else 'real') if data_mode_!='real' else {}
            for axis in ('place','task'):
                for item in classes[axis].values():
                    key=prefix+item['id']
                    nodes[key]=dict(id=key,kind='ContextPlace' if axis=='place' else 'ContextTask',name=item['name'],confirmed=item['confirmed'],**provenance)
            for r in group:
                if r['kind']!='context_inference':continue
                key=prefix+'context:'+r['id'];nodes[key]=dict(id=key,kind='ContextEpisode',at=r['at'],source='synthetic' if owner.endswith('-sample') else 'live_inference',architecture=r['payload'].get('architecture','legacy'),**provenance)
                for axis in ('place','task'):
                    value=r['payload'].get(axis) or {};target=value.get('id')
                    if target:
                        target=prefix+resolve(aliases[axis],target)
                        nodes.setdefault(target,dict(id=target,kind='ContextPlace' if axis=='place' else 'ContextTask',**provenance))
                        edges.append(dict(source=key,target=target,kind=axis,provenance=value.get('source')))
        unique={json.dumps(e,sort_keys=True):e for e in edges}
        return dict(nodes=list(nodes.values()),edges=list(unique.values()))


def install(app,store,config):
    engine=Engine(store,config);app.state.research=engine
    upstream=config.get('research_upstream','').rstrip('/')

    async def proxy(path,method='GET',body=None):
        if not upstream:return None
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r=await client.request(method,upstream+path,json=body)
            return Response(content=r.content,status_code=r.status_code,media_type=r.headers.get('content-type','application/json'))
        except httpx.HTTPError:raise HTTPException(503,'Ubuntu research server unavailable')

    @app.get('/api/export/{participant}')
    def export(participant:str,after:int=Query(0,ge=0),snapshot:int|None=Query(None,ge=0),limit:int=Query(500,ge=1,le=1000)):
        return export_page(store,identifier(participant),after,snapshot,limit)

    @app.get('/api/research/{participant}/export')
    async def research_export(participant:str,after:int=Query(0,ge=0),snapshot:int|None=Query(None,ge=0),limit:int=Query(500,ge=1,le=1000)):
        p=identifier(participant)
        query=f'?after={after}&limit={limit}'+(f'&snapshot={snapshot}' if snapshot is not None else '')
        forwarded=await proxy(f'/api/export/{p}'+query)
        return forwarded or export_page(store,p,after,snapshot,limit)

    @app.post('/api/heartbeats')
    def heartbeat(payload:Heartbeat):
        return dict(inserted=store.save('heartbeat',payload.event_id,payload.participant_id,payload.occurred_at,dict(payload.model_dump(mode='json'),received_at=iso(utc()))))

    @app.post('/api/research/{participant}/events')
    async def research_event(participant:str,payload:ResearchEvent):
        p=identifier(participant)
        if payload.participant_id!=p:raise HTTPException(422,'Participant mismatch')
        body=payload.model_dump(mode='json')
        if payload.kind=='place_visit':
            # Old phones/outboxes receive an ACK without adding new GPS-derived observations.
            return dict(inserted=False,ignored='gps_disabled')

        # Keep the source Mac export complete; forward immediately for queue consistency.
        qid=payload.payload.get('question_id')
        if qid and not upstream:
            q=next((r['payload'] for r in engine.rows(p) if r['kind']=='question_issued' and r['id']==qid),None)
            if q is None:raise HTTPException(422,'Unknown question')
            expected={'weather_mood':'weatherMood','typing_reason':'typingReason','intervention_outcome':'breakCheck','context_label':('placeCheck' if payload.payload.get('axis')=='place' else 'taskCheck')}.get(payload.kind)
            if expected and q['kind']!=expected:raise HTTPException(422,'Question type mismatch')
            if payload.kind=='context_label':
                if payload.payload['class_id'] not in [o['value'] for o in q['options']]:raise HTTPException(422,'Identity was not offered')
                if dt(payload.payload['end'])!=dt(q['detectedAt']) or dt(payload.payload['start'])!=dt(q['detectedAt'])-timedelta(minutes=5):raise HTTPException(422,'Answer must label the question observation window')

            if payload.kind=='intervention_outcome' and not any(o['value']=='accept:'+payload.payload['action'] for o in q['options']):
                raise HTTPException(422,'Action was not offered')
        if payload.kind in ('context_name','context_merge') and not upstream:
            from .context_data import catalog
            nodes,_=catalog(engine.rows(p),utc());axis=payload.payload['axis']
            ids=[payload.payload['class_id']] if payload.kind=='context_name' else [payload.payload['source_id'],payload.payload['target_id']]
            if any(key not in nodes[axis] for key in ids):raise HTTPException(422,'Unknown identity for this participant')
        inserted=store.save('research_event',payload.event_id,p,payload.occurred_at,dict(body,received_at=iso(utc())))
        forwarded=await proxy(f'/api/research/{p}/events','POST',body)
        return forwarded or dict(inserted=inserted)

    @app.get('/api/research/{participant}/insights')
    async def insights(participant:str,data_mode_:str=Query('real',alias='data_mode')):
        p=identifier(participant);data_mode_=mode(data_mode_)
        forwarded=await proxy(f'/api/research/{p}/insights?data_mode={data_mode_}')
        if forwarded:return forwarded
        at=engine.preview_at(p,data_mode_)
        prediction=None if data_mode_=='real' else await asyncio.to_thread(engine.infer,p,at,data_mode_)
        result=await asyncio.to_thread(engine.insight_snapshot,p,at,prediction,data_mode_)
        result['context']=await asyncio.to_thread(engine.context_snapshot,p,utc(),include_details=False) if data_mode_=='real' else dict(status='real_only')
        pref_view=engine.record_view(p,'real',('heartbeat','intervention_feedback','research_event'))
        result['preferences']=engine.api_cache.compute(('preferences',p),pref_view.fingerprint,
                                                      lambda:question_settings(pref_view.rows))
        result['data_mode']=data_mode_;result['contains_sample']=data_mode_!='real' or p==engine.sample_participant
        if data_mode_!='real':
            result['sample_generation']=engine.status(p,data_mode_).get('sample_generation')
            result['preview_support']=True
        return result

    @app.get('/research/{participant}/context',response_class=HTMLResponse)
    async def context_page(participant:str):
        p=identifier(participant)
        return HTMLResponse((Path(__file__).parent/'context.html').read_text().replace('__PARTICIPANT__',p),headers={'Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'"})

    @app.get('/api/research/{participant}/context')
    async def context_snapshot(participant:str):
        p=identifier(participant)
        forwarded=await proxy(f'/api/research/{p}/context')
        return forwarded or await asyncio.to_thread(engine.context_snapshot,p,utc())

    @app.post('/api/research/{participant}/context/review/{day}')
    async def context_review(participant:str,day:str):
        p=identifier(participant)
        try: datetime.strptime(day,'%Y-%m-%d')
        except ValueError:raise HTTPException(422,'Expected YYYY-MM-DD')
        forwarded=await proxy(f'/api/research/{p}/context/review/{day}','POST')
        if forwarded:return forwarded
        try:return await asyncio.to_thread(engine.context.review,p,day,utc())
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc

    @app.get('/api/research/{participant}/proposal-preview')
    async def proposal_preview(participant:str,data_mode_:str=Query('real',alias='data_mode')):
        p=identifier(participant)
        try:data_mode_=data_mode(data_mode_)
        except ValueError as exc:raise HTTPException(400,str(exc)) from exc
        forwarded=await proxy(f'/api/research/{p}/proposal-preview?data_mode={data_mode_}')
        return forwarded or await asyncio.to_thread(engine.proposal_preview,p,data_mode_)

    @app.get('/api/research/{participant}/overview')
    async def overview(participant:str,data_mode_:str=Query('real',alias='data_mode')):
        p=identifier(participant);data_mode_=mode(data_mode_)
        forwarded=await proxy(f'/api/research/{p}/overview?data_mode={data_mode_}')
        if forwarded:return forwarded
        # Inference is persisted by the worker, not by every dashboard poll.
        records=engine.record_view(p,data_mode_,('inference','narrative','correction')).rows
        if data_mode_=='real':
            latest=next((r['payload'] for r in records if r['kind']=='inference' and r['payload'].get('data_mode','real')=='real'
                         and r['payload'].get('architecture','legacy')==engine.architecture),None)
            stale=latest is None or (dt(utc())-dt(latest['at'])).total_seconds()>180
            prediction=deepcopy(latest) if not stale else dict(status='worker_not_running_or_stale',current=None,daily=None,data_mode='real',contains_sample=False)
        else:
            prediction=await asyncio.to_thread(engine.infer,p,None,data_mode_)
        # Older worker records contain only LSTM estimates; add provenance for
        # readers without rewriting persisted history.
        for scope in ('current','daily'):
            for item in (prediction.get(scope) or {}).values():
                item.setdefault('source','lstm')
                item.setdefault('provisional',False)
                item.setdefault('baseline_method',None)
        answers=[dict(at=r['at'],kind=r['payload'].get('question',{}).get('kind'),answer=r['payload'].get('answer'),
                      **(dict(participant=r['participant'],source='synthetic' if r['participant'].endswith('-sample') else 'real')
                         if data_mode_!='real' else {})) for r in records if r['kind']=='narrative']
        return dict(prediction=prediction,model=engine.status(p,data_mode_),answers=answers[:500],activity_catalog=ACTIVITIES,
                    architecture=engine.architecture,activity_role='posthoc_only' if engine.discovery is not None else 'auxiliary',
                    corrections=[r['payload'] for r in records if r['kind']=='correction'][:100],
                    data_mode=data_mode_,contains_sample=data_mode_!='real' or p==engine.sample_participant)

    @app.get('/api/research/{participant}/pmtt')
    async def pmtt(participant:str,data_mode_:str=Query('real',alias='data_mode')):
        p=identifier(participant);data_mode_=mode(data_mode_)
        forwarded=await proxy(f'/api/research/{p}/pmtt?data_mode={data_mode_}')
        # The upstream API already projects its full graph; do not prune twice.
        if forwarded:return forwarded
        return pmtt_projection(engine.pmtt(p,data_mode_))

    @app.post('/api/research/{participant}/train')
    async def train_model(participant:str,payload:TrainRequest):
        p=identifier(participant)
        body=payload.model_dump()
        forwarded=await proxy(f'/api/research/{p}/train','POST',body)
        return forwarded or await asyncio.to_thread(engine.train,p,payload.data_mode,payload.force)

    @app.post('/api/research/{participant}/corrections')
    async def correct(participant:str,payload:Correction):
        p=identifier(participant);forwarded=await proxy(f'/api/research/{p}/corrections','POST',payload.model_dump(mode='json'))
        if forwarded:return forwarded
        return dict(inserted=store.save('correction',payload.event_id,p,utc(),payload.model_dump(mode='json')))

    @app.post('/api/research/{participant}/feedback')
    async def feedback(participant:str,payload:Feedback):
        p=identifier(participant);forwarded=await proxy(f'/api/research/{p}/feedback','POST',payload.model_dump())
        if forwarded:return forwarded
        return dict(inserted=store.save('intervention_feedback',payload.id+':'+payload.outcome,p,utc(),payload.model_dump()))

    @app.get('/research/{participant}',response_class=HTMLResponse)
    def dashboard(participant:str):
        p=identifier(participant)
        return HTMLResponse((Path(__file__).parent/'dashboard.html').read_text().replace('__PARTICIPANT__',p),headers={'Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:"})

    @app.get('/research/{participant}/insights',response_class=HTMLResponse)
    def insights_dashboard(participant:str):
        p=identifier(participant)
        return HTMLResponse((Path(__file__).parent/'insights.html').read_text().replace('__PARTICIPANT__',p),headers={'Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:"})

    # Preserve original lifespan, including scanner shutdown / DB ownership.
    original=app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            tasks=[]
            if config.get('research_enabled') and not upstream:
                async def inference_loop():
                    while True:
                        try:
                            for p in participants(store,engine.sample_participant):
                                result=await asyncio.to_thread(engine.infer,p,None,'real')
                                t=utc();store.save('inference',str(uuid4()),p,t,result)
                                if engine.discovery is not None:
                                    await asyncio.to_thread(engine.insights.record_support,p,t,result)
                        except Exception:logging.exception('Research inference failed')
                        await asyncio.sleep(60)
                async def training_loop():
                    while True:
                        try:
                            modes=config.get('research_training_modes',['real'])
                            if not isinstance(modes,list) or any(x not in DATA_MODES for x in modes):
                                raise ValueError('research_training_modes must contain real, sample, or combined')
                            for p in participants(store,engine.sample_participant):
                                for data_mode_ in modes:await asyncio.to_thread(engine.train,p,data_mode_)
                        except Exception as exc:
                            from .io import atomic_json
                            for p in participants(store,engine.sample_participant):
                                for data_mode_ in config.get('research_training_modes',['real']):
                                    directory=engine.discovery.directory(p,data_mode_) if engine.discovery is not None else engine.directory(p,data_mode_)
                                    atomic_json(directory/'error.json',dict(at=iso(utc()),error=str(exc)))
                            logging.exception('Research training failed; previous checkpoint preserved')
                        await asyncio.sleep(3600)
                async def context_training_loop():
                    while True:
                        for p in participants(store,engine.sample_participant):
                            try:await asyncio.to_thread(engine.context.train,p,engine.rows(p),utc())
                            except Exception:logging.exception('Context training failed; prior checkpoint retained')
                        await asyncio.sleep(max(60,config.get('context_train_interval_seconds',3600)))
                async def context_loop():
                    reviewed={}
                    while True:
                        for p in participants(store,engine.sample_participant):
                            try:
                                now=utc()
                                result=await asyncio.to_thread(engine.context.infer,p,now)
                                store.save('context_inference',str(uuid4()),p,now,result)
                                # Legacy state IDs must not be attached to discovery embeddings.
                                if engine.discovery is None:
                                    from .insights import persist_state_step_snapshots
                                    persist_state_step_snapshots(store,p,now)
                                day=(dt(now)-timedelta(days=1)).date().isoformat()
                                if reviewed.get(p)!=day:
                                    if not any(r['kind']=='context_review' and r['payload']['day']==day
                                               and r['payload'].get('architecture','legacy')==engine.architecture for r in engine.rows(p)):
                                        await asyncio.to_thread(engine.context.review,p,day,now)
                                    reviewed[p]=day
                            except Exception:logging.exception('Behavior context worker failed; original records retained')
                        await asyncio.sleep(60)
                tasks=[asyncio.create_task(inference_loop()),asyncio.create_task(training_loop()),asyncio.create_task(context_loop())]
                if engine.discovery is None:
                    tasks.append(asyncio.create_task(context_training_loop()))
            try:yield
            finally:
                for task in tasks:task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError):await task
    app.router.lifespan_context=lifespan
    return engine


def participants(store,sample_participant=DEFAULT_SAMPLE_PARTICIPANT):
    from .sample_registry import real_participants
    return real_participants(store,sample_participant)
