"""API integration; runs on Mac without torch, on Ubuntu with optional ML worker."""
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from uuid import uuid4
import httpx
from fastapi import HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
from .data import SCALES, ACTIVITIES, Timeline, dt, monday


def utc(): return datetime.now(timezone.utc)
def iso(t): return t.isoformat(timespec='milliseconds')
def identifier(p):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',p):raise HTTPException(422,'Invalid participant')
    return p


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

    def rows(self,p):return self.store.rows(p)
    def directory(self,p):return self.root/identifier(p)

    def status(self,p):
        path=self.directory(p)/'status.json'
        result=json.loads(path.read_text()) if path.exists() else dict(status='untrained')
        error=self.directory(p)/'error.json'
        if error.exists():result['last_error']=json.loads(error.read_text())
        return result

    def infer(self,p,at=None):
        at=dt(at or utc());rows=self.rows(p);timeline=None
        status=self.status(p)
        response=dict(at=iso(at),source='model',status='untrained',current=None,daily=None,comparisons={},activity=None,model=status)
        checkpoint=self.directory(p)/'active.pt'
        if not checkpoint.exists():return response
        import numpy as np
        import torch
        from .model import HierarchicalLSTM
        from .train import batch
        from .data import VERSION, FEATURES
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state.get('version')!=VERSION or state.get('features')!=FEATURES:
            response['status']='incompatible_checkpoint';return response
        timeline=Timeline(rows,text_model=state['report'].get('text_model'))
        identity=timeline.encoder.identity if timeline.encoder else 'character-ngram-v1'
        if identity!=state['report'].get('text_identity',identity):
            response['status']='text_encoder_changed';return response
        model=HierarchicalLSTM(**state['settings']);model.load_state_dict(state['weights']);model.eval()
        # Do not display unvalidated heads, stale data, or synthetic/random outputs as measured states.
        events=[e for e in timeline.events if e[0]<=at and e[1] in ('key','session','switchbot','switchbot_observation','heartbeat')]
        if not events or (at-events[-1][0]).total_seconds()>3600:
            response['status']='observation_unknown';return response
        with torch.no_grad():
            outputs={}
            for scope in ('current','daily'):
                result=model(batch([timeline.sample(at,scope)],'cpu',torch.tensor(state['mean']),torch.tensor(state['std'])))
                outputs[scope]=result[scope][0].numpy()
                if scope=='current':activity_prob=torch.softmax(result['activity'][0],dim=-1)
            if state['report'].get('activity_train_n',0)>=10 and state['report']['validation']['activity_n']>=3 and activity_prob.max().item()>=.6:
                response['activity']=dict(label=ACTIVITIES[activity_prob.argmax().item()],source='model',score=float(activity_prob.max()),note='uncalibrated class score')
        response['status']='ready';response['model']=state['report']
        for scope,pred in outputs.items():
            response[scope]={}
            for i,k in enumerate(SCALES):
                valid=state['report']['validation'][scope][k]
                supported=state['report']['support'][scope][i]>=10 and valid['n']>=3
                response[scope][k]=dict(value=float(pred[i]) if supported else None,status='estimated' if supported else 'insufficient_labels',validation=valid)
        if all(v['value'] is None for scope in ('current','daily') for v in response[scope].values()):response['status']='insufficient_labels'
        past=[l for l in timeline.labels if l.scope=='current' and l.available<at]
        activity=timeline.activity_at(at)
        groups={'all':past,'time':[l for l in past if l.at.hour==at.hour],
                'weekday':[l for l in past if l.at.weekday()==at.weekday()],
                'activity':[l for l in past if activity>=0 and l.activity==activity]}
        for name,group in groups.items():
            result={}
            for i,k in enumerate(SCALES):
                values=[float(l.values[i]) for l in group if np.isfinite(l.values[i])]
                value=response['current'][k]['value']
                enough=len(values)>=5 and value is not None
                result[k]=dict(n=len(values),mean=float(np.mean(values)) if values else None,
                               difference=value-float(np.mean(values)) if enough else None,
                               z=(value-float(np.mean(values)))/max(float(np.std(values)),.5) if enough else None,
                               status='available' if enough else 'insufficient_data')
            response['comparisons'][name]=result
        return response

    def train(self,p):
        from .train import train
        if self.config.get('research_require_cuda'):
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check the GPU driver and CUDA-enabled PyTorch')
        rows=self.rows(p)
        count=sum(r['kind'] in ('narrative','correction') for r in rows)
        status=self.status(p)
        if status.get('training_record_count')==count:return dict(status='no_new_labels')
        result=train(rows,self.directory(p),epochs=self.config.get('research_epochs',20),text_model=self.config.get('text_model'))
        result['training_record_count']=count
        error=self.directory(p)/'error.json'
        if error.exists():error.unlink()
        from .train import atomic_json
        atomic_json(self.directory(p)/'status.json',result)
        return result

    def pending(self,p,at=None):
        at=dt(at or utc());rows=[r for r in self.rows(p) if dt(r['at'])<=at]
        heart=next((r['payload'] for r in rows if r['kind']=='heartbeat'),{})
        if heart.get('questions_enabled') is False:return None
        kind='nightReflection' if at.hour>=22 else 'currentState'
        if at.hour<8:return None
        narrative=[r for r in rows if r['kind']=='narrative' and r['payload'].get('question',{}).get('kind')==kind]
        if kind=='nightReflection':
            if any(dt(r['at']).date()==at.date() for r in narrative):return self.rest_question(p,at,rows)
            bucket=str(at.date());prompt='今日全体の状態を振り返りますか？';value='start_daily_reflection'
        else:
            issued=[r for r in rows if r['kind']=='question_issued' and r['payload'].get('kind')==kind]
            last=max([dt(r['at']) for r in narrative+issued],default=None)
            if last and (at-last).total_seconds()<10800:
                recent=next((r['payload'] for r in issued if dt(r['at'])==last),None)
                if recent and not any(r['id']==recent['id'] for r in narrative):return recent
                return self.rest_question(p,at,rows)
            bucket=iso(at);prompt='今の状態（5項目・各5段階）を教えてください';value='start_current_state'
        q=dict(id=hashlib.sha256(f'{p}:{kind}:{bucket}'.encode()).hexdigest(),participantID=p,kind=kind,prompt=prompt,
               options=[dict(label='回答を始める',value=value)],allowFreeText=False,eventType='state_assessment',
               detectedAt=iso(at),questionShownAt=None,snoozeCount=0,
               context=dict(time_of_day='night' if at.hour>=22 else 'afternoon',typing_active=False,calendar_busy=False,external_trigger='server'))
        self.store.save('question_issued',q['id'],p,at,q)
        return q

    def rest_question(self,p,at,rows):
        if not 8<=at.hour<22:return None
        feedback=[r for r in rows if r['kind']=='intervention_feedback' or (r['kind']=='narrative' and r['payload'].get('event_type')=='break_suggestion')]
        feedback=[dict(r,payload=dict(outcome=r['payload'].get('outcome') or r['payload'].get('answer',{}).get('label'))) for r in feedback]
        if feedback and feedback[0]['payload']['outcome']=='stop':return None
        heart=next((r['payload'] for r in rows if r['kind']=='heartbeat'),{})
        if heart.get('interventions_enabled') is False:return None
        offers=[r for r in rows if r['kind']=='intervention' and dt(r['at']).date()==at.date()]
        if len(offers)>=2 or any((at-dt(r['at'])).total_seconds()<3600 for r in offers):return None
        state=self.infer(p,at);fatigue=(state.get('current') or {}).get('fatigue',{}).get('value')
        z=state.get('comparisons',{}).get('all',{}).get('fatigue',{}).get('z')
        if fatigue is None or z is None or fatigue<4 or z<1:return None
        q=dict(id=str(uuid4()),participantID=p,kind='breakCheck',prompt='必要であれば、少し休憩しませんか？',
               options=[dict(label=x,value=y) for x,y in [('休憩する','accepted'),('今はしない','declined'),('後で','deferred'),('提案を停止','stop')]],
               allowFreeText=False,eventType='break_suggestion',detectedAt=iso(at),questionShownAt=None,snoozeCount=0,
               context=dict(time_of_day='afternoon',typing_active=False,calendar_busy=False,external_trigger='model'))
        self.store.save('intervention',q['id'],p,at,dict(question=q,evidence=state))
        return q

    def pmtt(self,p):
        # Explicit provenance graph: observations, work episodes, days, fixed weeks, and corrections.
        rows=self.rows(p);nodes={};edges=[]
        for r in rows:
            if r['kind'] not in {'session','narrative','switchbot','correction','inference'}:continue
            at=dt(r['at']);week='week:'+monday(at).date().isoformat();day='day:'+at.date().isoformat();work='work:'+at.replace(minute=0,second=0,microsecond=0).isoformat()
            for key,kind in [(week,'Pattern'),(day,'Day'),(work,'StateEpisode')]:nodes[key]=dict(id=key,kind=kind)
            key=r['kind']+':'+r['id'];nodes[key]=dict(id=key,kind='Observation',source=r['kind'],at=r['at'])
            edges.extend([dict(source=key,target=work,kind='provenance'),dict(source=work,target=day,kind='part_of'),dict(source=day,target=week,kind='part_of')])
            if r['kind']=='correction':nodes[key]['interval']=[r['payload']['start'],r['payload']['end']]
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
        return dict(inserted=store.save('heartbeat',payload.event_id,payload.participant_id,payload.occurred_at,payload.model_dump(mode='json')))

    @app.get('/api/research/{participant}/overview')
    async def overview(participant:str):
        p=identifier(participant)
        forwarded=await proxy(f'/api/research/{p}/overview')
        if forwarded:return forwarded
        # Inference is persisted by the worker, not by every dashboard poll.
        records=engine.rows(p);latest=next((r['payload'] for r in records if r['kind']=='inference'),None)
        stale=latest is None or (dt(utc())-dt(latest['at'])).total_seconds()>180
        prediction=latest if not stale else dict(status='worker_not_running_or_stale',current=None,daily=None)
        answers=[dict(at=r['at'],kind=r['payload'].get('question',{}).get('kind'),answer=r['payload'].get('answer')) for r in records if r['kind']=='narrative']
        return dict(prediction=prediction,model=engine.status(p),answers=answers[:500],activity_catalog=ACTIVITIES,
                    corrections=[r['payload'] for r in records if r['kind']=='correction'][:100])

    @app.get('/api/research/{participant}/pmtt')
    async def pmtt(participant:str):
        p=identifier(participant);forwarded=await proxy(f'/api/research/{p}/pmtt')
        return forwarded or engine.pmtt(p)

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

    # Preserve original lifespan, including scanner shutdown / DB ownership.
    original=app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            tasks=[]
            if config.get('research_enabled'):
                async def inference_loop():
                    while True:
                        try:
                            for p in participants(store):
                                result=await asyncio.to_thread(engine.infer,p)
                                t=utc();store.save('inference',str(uuid4()),p,t,result)
                        except Exception:logging.exception('Research inference failed')
                        await asyncio.sleep(60)
                async def training_loop():
                    while True:
                        try:
                            for p in participants(store):await asyncio.to_thread(engine.train,p)
                        except Exception as exc:
                            from .train import atomic_json
                            for p in participants(store):atomic_json(engine.directory(p)/'error.json',dict(at=iso(utc()),error=str(exc)))
                            logging.exception('Research training failed; previous checkpoint preserved')
                        await asyncio.sleep(3600)
                tasks=[asyncio.create_task(inference_loop()),asyncio.create_task(training_loop())]
            try:yield
            finally:
                for task in tasks:task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError):await task
    app.router.lifespan_context=lifespan
    return engine


def participants(store):
    with store.lock:return [r[0] for r in store.db.execute('SELECT DISTINCT participant FROM records')]
