"""python -m research.cli --help. Exported private data must stay outside git."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse
import httpx
from lifelog_server import Store
from .service import export_page, ingest_page, identifier
from .data import dt
from .data_modes import DATA_MODES, DEFAULT_SAMPLE_PARTICIPANT, rows_for_mode, source_identity


def export_all(store, participant, path):
    # Single SQLite read transaction represented as one page: includes every record and device metadata.
    with store.lock:
        page=export_page(store,participant,limit=2**31-1)
    Path(path).write_text(json.dumps(page,ensure_ascii=False),encoding='utf-8')
    return dict(records=len(page['records']),path=str(path))


def import_file(store,path,participant):
    text=Path(path).read_text(encoding='utf-8')
    try:
        value=json.loads(text)
        if isinstance(value,dict) and value.get('format')=='lifelog-all-v1':return ingest_page(store,value)
    except json.JSONDecodeError:pass
    # Old iPhone session-export JSONL contains file/type/time/data, not full research uploads.
    groups={}
    for n,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        r=json.loads(line)
        if not all(k in r for k in ('file','type','time','data')):raise ValueError(f'Unsupported JSONL line {n}')
        dt(r['time']);groups.setdefault(r['file'],[]).append(r)
    inserted=0
    for name,events in groups.items():
        events.sort(key=lambda e:dt(e['time']))
        sid='legacy-'+hashlib.sha256((participant+name).encode()).hexdigest()
        payload=dict(session_id=sid,participant_id=participant,device_id='legacy-import',started_at=events[0]['time'],ended_at=events[-1]['time'],
                     close_reason='legacy_export',events=[{k:e[k] for k in ('type','time','data')} for e in events],typing={},context=None,provenance='legacy_event_export')
        inserted+=store.save('session',sid,participant,dt(payload['ended_at']),payload)
    return inserted


def sync(store,source,participant,state_path,client=None):
    # Persist cursor only after inserts commit. Retry is idempotent using original record IDs.
    from .train import atomic_json
    if urlparse(source).scheme not in ('http','https'):raise ValueError('HTTP(S) source required')
    state_path=Path(state_path)
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    key=source.rstrip('/')+'|'+participant
    cursor=state.get(key,0);snapshot=None;inserted=0
    own=client is None;client=client or httpx.Client(timeout=60)
    try:
        while True:
            params=dict(after=cursor,limit=500)
            if snapshot is not None:params['snapshot']=snapshot
            result=client.get(source.rstrip('/')+f'/api/export/{participant}',params=params);result.raise_for_status();page=result.json()
            if page.get('participant')!=participant:raise ValueError('Participant mismatch')
            if page['snapshot']<cursor:raise ValueError('Source database rewound; explicitly reset replication state after checking source')
            snapshot=page['snapshot'];inserted+=ingest_page(store,page)
            cursor=page['next_cursor'] if page['next_cursor'] is not None else snapshot
            state[key]=cursor;atomic_json(state_path,state)
            if page['next_cursor'] is None:break
    finally:
        if own:client.close()
    return dict(inserted=inserted,cursor=cursor)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',default='data/lifelog.sqlite3');p.add_argument('--participant',default='p001')
    sub=p.add_subparsers(dest='command',required=True)
    ex=sub.add_parser('export');ex.add_argument('path')
    im=sub.add_parser('import');im.add_argument('path')
    sy=sub.add_parser('sync');sy.add_argument('--source',required=True);sy.add_argument('--state',default='data/replication.json');sy.add_argument('--watch',action='store_true')
    for name in ('train','compare','personalize'):
        tr=sub.add_parser(name);tr.add_argument('--models',default='data/models');tr.add_argument('--epochs',type=int,default=20);tr.add_argument('--text-model',default=None)
        tr.add_argument('--data-mode',choices=DATA_MODES,default='real');tr.add_argument('--sample-participant',default=DEFAULT_SAMPLE_PARTICIPANT)
        tr.add_argument('--sample-manifest',default=None)
    sample=sub.add_parser('generate-sample');sample.add_argument('--calendar',required=True);sample.add_argument('--sample-participant',default=DEFAULT_SAMPLE_PARTICIPANT)
    sample.add_argument('--manifest',default=None);sample.add_argument('--dry-run',action='store_true');sample.add_argument('--include-today',action='store_true')
    args=p.parse_args();identifier(args.participant)
    if args.command=='generate-sample' and args.dry_run:
        from .sample_data import generate_sample
        result=generate_sample(args.calendar,args.sample_participant,args.participant,manifest_path=args.manifest,
                               dry_run=True,include_today=args.include_today)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False));return
    store=Store(args.db)
    try:
        if args.command=='export':result=export_all(store,args.participant,args.path)
        elif args.command=='import':result=dict(inserted=import_file(store,args.path,args.participant))
        elif args.command=='sync':
            while True:
                try:result=sync(store,args.source,args.participant,args.state);print(json.dumps(result),flush=True)
                except Exception:
                    if not args.watch:raise
                    import logging;logging.exception('Replication failed; cursor retained')
                if not args.watch:return
                time.sleep(60)
        elif args.command=='generate-sample':
            from .sample_data import generate_sample
            result=generate_sample(args.calendar,args.sample_participant,args.participant,store=store,manifest_path=args.manifest,
                                   include_today=args.include_today)
        else:
            from .train import train,ablations,personalization_evaluation
            action={'compare':ablations,'train':train,'personalize':personalization_evaluation}[args.command]
            manifest=args.sample_manifest or f'data/sample/{args.sample_participant}/manifest.json'
            rows=rows_for_mode(store,args.participant,args.data_mode,args.sample_participant)
            directory=Path(args.models)/args.participant/args.data_mode
            identity=source_identity(store,args.participant,args.data_mode,args.sample_participant,manifest)
            metadata=dict(data_mode=args.data_mode,contains_sample=args.data_mode!='real' or args.participant==args.sample_participant,
                          source_fingerprint=identity['fingerprint'],source_max_seq=identity['source_max_seq'],
                          sample_manifest_hash=identity['sample_manifest_hash'])
            if args.command=='train':
                result=action(rows,directory,epochs=args.epochs,text_model=args.text_model,**metadata)
            else:
                result=action(rows,directory/args.command,epochs=args.epochs,text_model=args.text_model,**metadata)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    finally:store.close()

if __name__=='__main__':main()
