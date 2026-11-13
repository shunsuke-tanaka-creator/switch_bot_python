"""API integration; runs on Mac without torch, on Ubuntu with optional ML worker."""
import asyncio
from contextlib import asynccontextmanager, suppress
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
    rows_for_mode,
    source_identity,
)
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


class Engine:
    def __init__(self,store,config):
        self.store=store;self.config=config
        self.root=Path(config.get('research_models','data/models'));self.root.mkdir(parents=True,exist_ok=True)
        self.sample_participant=config.get('research_sample_participant',DEFAULT_SAMPLE_PARTICIPANT)
        self.sample_manifest=Path(config.get('research_sample_manifest',f'data/sample/{self.sample_participant}/manifest.json'))
        self.context=ContextEngine(store,config)
        self.insights=InsightEngine(store,config)
        self.questions=QuestionQueue(store,self.insights)

    def rows(self,p,data_mode_='real'):
        return rows_for_mode(self.store,identifier(p),data_mode(data_mode_),self.sample_participant)
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
        path=self.read_path(p,data_mode_,'status.json')
        result=json.loads(path.read_text()) if path.exists() else dict(status='untrained')
        result.setdefault('data_mode',data_mode_)
        result.setdefault('contains_sample',data_mode_!='real' or p==self.sample_participant)
        error=self.read_path(p,data_mode_,'error.json')
        if error.exists():result['last_error']=json.loads(error.read_text())
        return result

    def infer(self,p,at=None,data_mode_='real'):
        from .inference import infer_state
        return infer_state(self,p,at,data_mode_)

    def train(self,p,data_mode_='real',force=False):
        data_mode_=data_mode(data_mode_)
        from .train import train
        if self.config.get('research_require_cuda'):
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check the GPU driver and CUDA-enabled PyTorch')
        rows=self.rows(p,data_mode_)
        count=len(Timeline(rows).labels)
        identity=source_identity(self.store,p,data_mode_,self.sample_participant,self.sample_manifest)
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
        if p==self.sample_participant:return None
        return self.questions.pending(p,dt(at or utc()))

    def proposal_preview(self,p,data_mode_='real',at=None):
        """Preview queue selection without reserving or consuming any budget.

        Mode rows drive insights and answer-dependent trigger chains, while the
        participant's real rows exclusively drive pending/budget/gap queue state.
        """
        data_mode_=data_mode(data_mode_);at=dt(at or utc())
        records=self.rows(p,data_mode_)
        prediction=None if data_mode_=='real' else self.infer(p,at,data_mode_)
        insight=self.insights.snapshot(p,at,prediction,records,data_mode_)
        selected=self.questions.preview(p,at,records,insight,self.rows(p,'real'))
        return dict(preview=True,participant=p,data_mode=data_mode_,
                    contains_sample=data_mode_!='real' or p==self.sample_participant,
                    at=iso(at),question=selected['question'],gates=selected['gates'])

    def pmtt(self,p,data_mode_='real'):
        # Explicit provenance graph: observations, work episodes, days, fixed weeks, and corrections.
        rows=self.rows(p,data_mode_);nodes={};edges=[]
        for r in rows:
            if r['kind'] not in {'session','narrative','switchbot','correction','inference'}:continue
            at=dt(r['at']);week='week:'+monday(at).date().isoformat();day='day:'+at.date().isoformat();work='work:'+at.replace(minute=0,second=0,microsecond=0).isoformat()
            for key,kind in [(week,'Pattern'),(day,'Day'),(work,'StateEpisode')]:nodes[key]=dict(id=key,kind=kind)
            key=r['kind']+':'+r['id'];nodes[key]=dict(id=key,kind='Observation',source=r['kind'],at=r['at'])
            edges.extend([dict(source=key,target=work,kind='provenance'),dict(source=work,target=day,kind='part_of'),dict(source=day,target=week,kind='part_of')])
            if r['kind']=='correction':nodes[key]['interval']=[r['payload']['start'],r['payload']['end']]
        from .context_data import catalog
        classes,aliases=catalog(rows,dt(utc()))
        from .context_data import resolve
        for axis in ('place','task'):
            for item in classes[axis].values():
                nodes[item['id']]=dict(id=item['id'],kind='ContextPlace' if axis=='place' else 'ContextTask',name=item['name'],confirmed=item['confirmed'])
        for r in rows:
            if r['kind']!='context_inference':continue
            key='context:'+r['id'];nodes[key]=dict(id=key,kind='ContextEpisode',at=r['at'],source='live_inference')
            for axis in ('place','task'):
                value=r['payload'].get(axis) or {};target=value.get('id')
                if target:
                    target=resolve(aliases[axis],target)
                    nodes.setdefault(target,dict(id=target,kind='ContextPlace' if axis=='place' else 'ContextTask'))
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
        records=engine.rows(p,data_mode_)
        prediction=None if data_mode_=='real' else await asyncio.to_thread(engine.infer,p,None,data_mode_)
        result=await asyncio.to_thread(engine.insights.snapshot,p,utc(),prediction,records,data_mode_)
        result['context']=engine.context.snapshot(p,utc(),records) if data_mode_=='real' else dict(status='real_only')
        result['preferences']=question_settings(engine.rows(p,'real'))
        result['data_mode']=data_mode_;result['contains_sample']=data_mode_!='real' or p==engine.sample_participant
        return result

    @app.get('/research/{participant}/context',response_class=HTMLResponse)
    async def context_page(participant:str):
        p=identifier(participant)
        return HTMLResponse((Path(__file__).parent/'context.html').read_text().replace('__PARTICIPANT__',p),headers={'Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'"})

    @app.get('/api/research/{participant}/context')
    async def context_snapshot(participant:str):
        p=identifier(participant)
        forwarded=await proxy(f'/api/research/{p}/context')
        return forwarded or await asyncio.to_thread(engine.context.snapshot,p,utc())

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
        records=engine.rows(p,data_mode_)
        if data_mode_=='real':
            latest=next((r['payload'] for r in records if r['kind']=='inference' and r['payload'].get('data_mode','real')=='real'),None)
            stale=latest is None or (dt(utc())-dt(latest['at'])).total_seconds()>180
            prediction=latest if not stale else dict(status='worker_not_running_or_stale',current=None,daily=None,data_mode='real',contains_sample=False)
        else:
            prediction=await asyncio.to_thread(engine.infer,p,None,data_mode_)
        answers=[dict(at=r['at'],kind=r['payload'].get('question',{}).get('kind'),answer=r['payload'].get('answer')) for r in records if r['kind']=='narrative']
        return dict(prediction=prediction,model=engine.status(p,data_mode_),answers=answers[:500],activity_catalog=ACTIVITIES,
                    corrections=[r['payload'] for r in records if r['kind']=='correction'][:100],
                    data_mode=data_mode_,contains_sample=data_mode_!='real' or p==engine.sample_participant)

    @app.get('/api/research/{participant}/pmtt')
    async def pmtt(participant:str,data_mode_:str=Query('real',alias='data_mode')):
        p=identifier(participant);data_mode_=mode(data_mode_)
        forwarded=await proxy(f'/api/research/{p}/pmtt?data_mode={data_mode_}')
        return forwarded or engine.pmtt(p,data_mode_)

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
                                    atomic_json(engine.directory(p,data_mode_)/'error.json',dict(at=iso(utc()),error=str(exc)))
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
                                day=(dt(now)-timedelta(days=1)).date().isoformat()
                                if reviewed.get(p)!=day:
                                    if not any(r['kind']=='context_review' and r['payload']['day']==day for r in engine.rows(p)):
                                        await asyncio.to_thread(engine.context.review,p,day,now)
                                    reviewed[p]=day
                            except Exception:logging.exception('Behavior context worker failed; original records retained')
                        await asyncio.sleep(60)
                tasks=[asyncio.create_task(inference_loop()),asyncio.create_task(training_loop()),asyncio.create_task(context_loop()),asyncio.create_task(context_training_loop())]
            try:yield
            finally:
                for task in tasks:task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError):await task
    app.router.lifespan_context=lifespan
    return engine


def participants(store,sample_participant=DEFAULT_SAMPLE_PARTICIPANT):
    with store.lock:
        return [r[0] for r in store.db.execute('SELECT DISTINCT participant FROM records') if r[0]!=sample_participant]
