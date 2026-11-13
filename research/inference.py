"""State inference from a compatible checkpoint; no training or HTTP side effects."""
from datetime import datetime, timezone
from .data import Timeline, dt, ACTIVITIES, SCALES
from .data_modes import data_mode


def infer_state(engine,p,at=None,data_mode_='real'):
    data_mode_=data_mode(data_mode_)
    at=dt(at or datetime.now(timezone.utc));rows=engine.rows(p,data_mode_)
    status=engine.status(p,data_mode_)
    response=dict(at=at.isoformat(timespec='milliseconds'),source='model',status='untrained',current=None,daily=None,comparisons={},activity=None,
                  data_mode=data_mode_,contains_sample=data_mode_!='real' or p==engine.sample_participant,model=status)
    checkpoint=engine.read_path(p,data_mode_,'active.pt')
    if not checkpoint.exists():return response
    import numpy as np
    import torch
    from .model import HierarchicalLSTM
    from .train import batch
    from .data import VERSION, FEATURES
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('version')!=VERSION or state.get('features')!=FEATURES:
        response['status']='incompatible_checkpoint';return response
    timeline=Timeline(rows,text_model=engine.config.get('text_model') or state['report'].get('text_model'))
    identity=timeline.encoder.identity if timeline.encoder else 'character-ngram-v1'
    if identity!=state['report'].get('text_identity',identity):
        response['status']='text_encoder_changed';return response
    model=HierarchicalLSTM(**state['settings']);model.load_state_dict(state['weights']);model.eval()
    # Do not display unvalidated heads, stale data, or synthetic/random outputs as measured states.
    events=[e for e in timeline.events if e[0]<=at and dt(e[2].get('_available_at', e[2].get('received_at', e[0])))<=at and e[1] in ('key','session','switchbot','switchbot_observation','heartbeat')]
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

