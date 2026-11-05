"""Versioned features and labels. No questionnaire/model output is a sensor feature."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache
from bisect import bisect_right
import hashlib
import json
import math
import numpy as np

JST = ZoneInfo('Asia/Tokyo')
SCALES = ('mood', 'fatigue', 'stress', 'concentration', 'sleepiness')
ACTIVITIES = 'desk_work meeting study cooking cleaning laundry shopping meal bath toilet dressing relax nap sleep_prep video game reading sns walk commute errand family friends call'.split()
ACTIVITY_NAMES = dict(zip('デスクワーク 会議・打合せ 勉強・研究 料理 掃除 洗濯 買い物 食事 入浴 トイレ 身支度 くつろぎ 仮眠 就寝準備 テレビ・動画 ゲーム 読書 スマホ・SNS 散歩 通勤・通学 用事 家族と 友人と 通話'.split(), ACTIVITIES))
# Text: local signed character 1/2/3-gram hashing, no external service or downloads.
TEXT_DIM = 32
BASE = ['key_count','delete_count','commit_chars','pause_count','mean_interval','max_interval',
        'observed_session','observation_age','heartbeat_age','collection_enabled',
        'door_open','door_age','room_open','room_age',
        'steps','distance','flights','energy','speed','stride','asymmetry','health_age',
        'calendar_busy','calendar_count','battery','charging','context_age',
        'hour_sin','hour_cos','weekday_sin','weekday_cos','elapsed','text_chars']
# Every measurement has a presence channel. Missing is never interpreted as a measured zero.
FEATURES = BASE + [f'{x}_known' for x in BASE] + [f'text_{i}' for i in range(TEXT_DIM)]
F = len(FEATURES)
VERSION = 'hierarchical-v1-5min-1h-jst-week'


def dt(s):
    t = datetime.fromisoformat(s.replace('Z', '+00:00')) if isinstance(s, str) else s
    if t.tzinfo is None: raise ValueError('timezone required')
    return t.astimezone(JST)


def midnight(t): return dt(t).replace(hour=0, minute=0, second=0, microsecond=0)
def monday(t): return midnight(t) - timedelta(days=dt(t).weekday())

def text_vector(text):
    out = np.zeros(TEXT_DIM, np.float32)
    text = text[-4096:]
    for size in (1, 2, 3):
        for i in range(max(0, len(text)-size+1)):
            h = int.from_bytes(hashlib.blake2b(text[i:i+size].encode(), digest_size=8).digest(), 'little')
            out[h % TEXT_DIM] += 1 if h & 256 else -1
    norm = np.linalg.norm(out)
    return out / norm if norm else out


@dataclass
class Label:
    at: datetime                 # prediction cutoff (question shown, not answer completed)
    available: datetime          # label was not available until this time
    scope: str
    values: np.ndarray
    activity: int = -1
    id: str = ''


class Timeline:
    def __init__(self, rows, text_model=None):
        self.encoder = None
        if text_model:
            from .text import local_encoder
            self.encoder = local_encoder(text_model)
        self.rows = sorted(rows, key=lambda r: (dt(r['at']), r.get('seq', 0)))
        self.events, self.labels, self.corrections = [], [], []
        for r in self.rows:
            p, kind, at = r['payload'], r['kind'], dt(r['at'])
            if kind == 'session':
                start, end = dt(p['started_at']), dt(p['ended_at'])
                self.events.append((end, 'session', p))
                for e in p.get('events', []):
                    if e.get('data',{}).get('purpose')=='assessment': continue
                    if e.get('type') in {'key','delete','commit','context','kana','conversion'}:
                        self.events.append((dt(e['time']), e['type'], dict(e.get('data', {}), _available_at=p.get('received_at') or p['ended_at'])))
            elif kind in {'switchbot','switchbot_observation','heartbeat','sensor'}:
                self.events.append((at, kind, p))
            elif kind == 'correction':
                self.corrections.append((at, p))
                if p.get('activity') in ACTIVITIES:
                    self.labels.append(Label(dt(p['end']),at,'activity',np.full(5,np.nan,np.float32),ACTIVITIES.index(p['activity']),r['id']+':activity'))
                if p.get('scores'):
                    self.labels.append(Label(dt(p['end']), at, p.get('scope','current'), self.scores(p['scores']), id=r['id']))
            elif kind == 'narrative':
                answer = p.get('answer', {})
                question = p.get('question', {}).get('kind')
                cutoff = dt(p.get('question_shown_at') or p.get('detected_at') or r['at'])
                if cutoff > at: continue
                if question in {'currentState','nightReflection'}:
                    try:
                        scores = json.loads(answer.get('free_text') or '{}')
                        values = self.scores(scores)
                        if np.isfinite(values).any():
                            self.labels.append(Label(cutoff, at, 'daily' if question == 'nightReflection' else 'current', values, id=r['id']))
                    except (ValueError, TypeError): pass
                elif question == 'activityCheck':
                    activity=answer.get('label') or ACTIVITY_NAMES.get(answer.get('free_text'))
                    if activity in ACTIVITIES:
                        self.labels.append(Label(cutoff, at, 'activity', np.full(5, np.nan, np.float32), ACTIVITIES.index(activity), r['id']))
        self.events.sort(key=lambda e: e[0])
        self.features = lru_cache(maxsize=12000)(self.features)
        self.labels.sort(key=lambda l: l.at)
        self.event_times = [e[0] for e in self.events]
        for l in self.labels:
            if l.scope in ('current','daily'): l.activity = self.activity_at(l.at)

    @staticmethod
    def scores(d):
        if not isinstance(d, dict): raise ValueError('scores must be object')
        return np.array([float(d[k]) if type(d.get(k)) in (int,float) and 1 <= d[k] <= 5 else np.nan for k in SCALES], np.float32)

    def baseline(self, at, scope, activity=-1):
        past = [l for l in self.labels if l.scope == scope and l.available < at and l.at < at]
        vectors = []
        # All history / same hour / same weekday / same activity. Sparse groups remain masked.
        for group in [past, [l for l in past if l.at.hour == at.hour],
                      [l for l in past if l.at.weekday() == at.weekday()],
                      [l for l in past if activity >= 0 and l.activity == activity]]:
            if group:
                vals = np.stack([l.values for l in group]); count = np.isfinite(vals).sum(0)
                avg = np.nansum(vals, 0) / np.maximum(count, 1)
                vectors.extend(np.where(count >= 3, avg, 3).tolist())
                vectors.extend((count >= 3).astype(float).tolist())
            else: vectors.extend([3.]*5 + [0.]*5)
        return np.array(vectors, np.float32)

    def activity_at(self, at):
        for available, p in reversed(self.corrections):
            if available <= at and dt(p['start']) <= at < dt(p['end']) and p.get('activity') in ACTIVITIES:
                return ACTIVITIES.index(p['activity'])
        prior = [l for l in self.labels if l.scope == 'activity' and l.available <= at and 0 <= (at-l.at).total_seconds() <= 1800]
        return prior[-1].activity if prior else -1

    def features(self, start, end):
        vals = {}; stop=bisect_right(self.event_times,end); begin=bisect_right(self.event_times,start)
        recent=[e for e in self.events[:stop] if dt(e[2].get('_available_at',e[2].get('received_at',e[0])))<=end]
        window=[e for e in recent if start<e[0]<=end]
        # Without coverage, no key events is unknown; observed non-use is represented by presence mask.
        heart = next((e for e in reversed(recent) if e[1] == 'heartbeat'), None)
        if heart:
            age = (end-heart[0]).total_seconds()
            vals['heartbeat_age'] = min(age/3600, 24)
            if age <= 300:
                enabled = heart[2].get('logging_enabled')
                if enabled is not None: vals['collection_enabled'] = float(enabled)
        sessions = [e for e in window if e[1] == 'session']
        latest_session = next((e for e in reversed(recent) if e[1] == 'session'), None)
        if latest_session: vals['observation_age'] = min((end-latest_session[0]).total_seconds()/3600, 24)
        keys = [e[0] for e in window if e[1] == 'key']
        coverage = bool(sessions or keys or (heart and (end-heart[0]).total_seconds() <= 300 and heart[2].get('logging_enabled')))
        if coverage:
            vals.update(key_count=math.log1p(len(keys)), delete_count=math.log1p(sum(e[1]=='delete' for e in window)),
                        commit_chars=math.log1p(sum(len(e[2].get('text','')) for e in window if e[1]=='commit')),
                        observed_session=float(bool(sessions)))
        gaps = np.diff([k.timestamp() for k in keys])
        if len(gaps): vals.update(pause_count=math.log1p(sum(gaps>2)), mean_interval=float(np.mean(gaps))/60, max_interval=float(max(gaps))/60)
        # Exclude context.before/after from text features: overlapping text would duplicate history.
        text = ''.join(e[2].get('text','') for e in window if e[1]=='commit')
        if text: vals['text_chars'] = math.log1p(len(text))
        if latest_session:
            c = latest_session[2].get('context') or {}
            updated = c.get('updated_at')
            if updated and dt(updated) <= end:
                age = (end-dt(updated)).total_seconds(); vals['context_age'] = min(age/3600,24)
                if age <= 7200:
                    mapping = {'step_count':'steps','walking_distance_m':'distance','flights_climbed':'flights',
                               'active_energy_kcal':'energy','walking_speed_mps':'speed','walking_step_length_m':'stride',
                               'walking_asymmetry_pct':'asymmetry','battery_level':'battery'}
                    for src, dest in mapping.items():
                        field_time=(c.get('measurement_updated_at') or {}).get(src)
                        fresh=field_time and 0<=(end-dt(field_time)).total_seconds()<=7200
                        if fresh and isinstance(c.get(src),(int,float)) and math.isfinite(c[src]): vals[dest] = math.log1p(max(0,c[src]))
                    vals['health_age'] = age/3600
                    if (c.get('measurement_updated_at') or {}).get('battery_level') and c.get('battery_state') not in (None,'unknown'): vals['charging'] = float(c['battery_state'] in ('charging','full'))
                    if (c.get('measurement_updated_at') or {}).get('calendar_events') and c.get('calendar_events') is not None:
                        ev = c['calendar_events']; vals['calendar_count'] = math.log1p(len(ev))
                        vals['calendar_busy'] = float(any(dt(x['start']) <= end < dt(x['end']) for x in ev))
        for prefix, word in [('door','玄関'),('room','寝室')]:
            sensor = next((e for e in reversed(recent) if e[1] in ('switchbot','switchbot_observation') and word in e[2].get('name','')), None)
            if sensor:
                age = (end-sensor[0]).total_seconds(); vals[prefix+'_age'] = min(age/3600,24)
                # Requires periodic BLE observations, not just door transitions.
                if age <= 120 and sensor[2].get('state') in ('open','close','timeOutNotClose'):
                    vals[prefix+'_open'] = float(sensor[2]['state']!='close')
        hour = end.hour + end.minute/60
        vals.update(hour_sin=math.sin(hour*math.pi/12),hour_cos=math.cos(hour*math.pi/12),
                    weekday_sin=math.sin(end.weekday()*2*math.pi/7), weekday_cos=math.cos(end.weekday()*2*math.pi/7),
                    elapsed=(end-start).total_seconds()/300)
        base = np.array([vals.get(k,0) for k in BASE],np.float32)
        return np.concatenate([base, np.array([float(k in vals) for k in BASE],np.float32), self.encoder.encode(text) if self.encoder else text_vector(text)])

    def sample(self, at, scope='current', max_units=48):
        """Current fixed JST week, causal 5-minute bins inside <=1h work units.
        Confirmed activity changes split units; gaps still produce explicitly masked bins.
        """
        at = dt(at); week = monday(at)
        x = np.zeros((7,max_units,12,F),np.float32)
        lengths = np.zeros((7,max_units),np.int64); units = np.zeros(7,np.int64)
        for day in range(at.weekday()+1):
            begin = week + timedelta(days=day); stop = min(begin+timedelta(days=1),at)
            cuts = {begin,stop}
            tick = begin+timedelta(hours=1)
            while tick < stop: cuts.add(tick); tick += timedelta(hours=1)
            for l in self.labels:
                if l.scope=='activity' and l.available <= at and begin < l.at < stop:
                    cuts.add(l.at)
            for available,p in self.corrections:
                if available <= at:
                    for bound in ['start','end']:
                        t=dt(p[bound])
                        if begin < t < stop: cuts.add(t)
            cuts=sorted(cuts)
            # Preserve latest units; overflow is explicit and bounded, never a silent giant batch.
            if len(cuts)-1 > max_units: raise ValueError('Too many work boundaries in a day; increase max_units')
            for j,(a,b) in enumerate(zip(cuts,cuts[1:])):
                t=a; k=0
                while t < b and k < 12:
                    e=min(t+timedelta(minutes=5),b)
                    x[day,j,k]=self.features(t,e); t=e; k+=1
                lengths[day,j]=k
            units[day]=len(cuts)-1
        # At exactly midnight create one partial, empty bin so packing remains well defined.
        if units[at.weekday()] == 0:
            units[at.weekday()]=1; lengths[at.weekday(),0]=1
            x[at.weekday(),0,0]=self.features(at,at)
        # Previous four weeks of sensor-only daily features, all strictly before current week.
        history=np.stack([self.features(week-timedelta(days=i+1),week-timedelta(days=i)) for i in reversed(range(28))])
        activity=self.activity_at(at)
        return dict(x=x,lengths=lengths,units=units,days=np.int64(at.weekday()+1),history=history,
                    personal=self.baseline(at,scope,activity),activity=np.int64(activity))

    def examples(self):
        return [(l,self.sample(l.at,l.scope)) for l in self.labels if l.scope in ('current','daily','activity')]
