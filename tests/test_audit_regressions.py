"""CPU-only regression checks: python -m unittest discover -s tests -p test_audit_regressions.py."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fastapi.testclient import TestClient
from lifelog_server import Store, create_app
from research.data import Timeline, BASE, SCALES, dt, LazySample, committed_text
from research.baseline import period
from research.evaluation import online_updates
from research.sample_data import generate_sample
from research.text import encoder_identity
from research.provenance import experiment_id
from research.questions import QuestionQueue
from research.data_modes import source_identity
from research.io import atomic_json


def row(kind, key, at, payload):
    return dict(kind=kind,id=key,at=at.isoformat(),payload=payload,participant='p001')


def answer(key, at, received):
    return row('narrative',key,at+timedelta(seconds=10),dict(
        question_shown_at=at.isoformat(),received_at=received.isoformat(),
        question={'kind':'currentState'},answer={'free_text':json.dumps(dict.fromkeys(SCALES,5))}))


class AuditRegressionTests(unittest.TestCase):
    def test_delayed_answer_not_known_until_received(self):
        at=dt('2026-11-01T10:00:00+09:00')
        timeline=Timeline([answer(str(i),at+timedelta(minutes=i),at+timedelta(days=7)) for i in range(3)])
        np.testing.assert_array_equal(timeline.baseline(at+timedelta(days=1),'current')[:10], [3]*5+[0]*5)
        self.assertEqual(timeline.baseline(at+timedelta(days=8),'current')[0],5)

    def test_delayed_key_recovered_at_prediction_cutoff(self):
        start=dt('2026-11-01T10:00:00+09:00'); end=start+timedelta(minutes=5,seconds=20)
        timeline=Timeline([row('session','s',end,dict(started_at=start.isoformat(),ended_at=end.isoformat(),received_at=(end+timedelta(seconds=1)).isoformat(),events=[dict(type='key',time=(start+timedelta(minutes=4,seconds=45)).isoformat(),data={})]))])
        before=timeline.features(start,start+timedelta(minutes=5))
        after=timeline.features(start,start+timedelta(minutes=5),start+timedelta(minutes=10))
        self.assertEqual(before[BASE.index('key_count')],0)
        self.assertAlmostEqual(after[BASE.index('key_count')],np.log(2),places=6)
        self.assertGreater(timeline.sample(start+timedelta(minutes=10))['x'][6,:,:,BASE.index('key_count')].sum(),0)

    def test_online_uses_prior_same_day_but_not_target_or_late_answer(self):
        at=dt('2026-11-01T10:00:00+09:00')
        examples=Timeline([answer('early',at,at+timedelta(minutes=1)),answer('target',at+timedelta(hours=1),at+timedelta(hours=1,minutes=1)),answer('late',at,at+timedelta(days=1))]).examples()
        self.assertEqual([x[0].id for x in online_updates(examples,set(),at+timedelta(hours=1),'2026-11-01')],['early'])
        self.assertTrue(all(isinstance(s,LazySample) for _,s in examples))

    def test_time_period_boundaries(self):
        t=dt('2026-11-01T00:00:00+09:00')
        for hour,expected in [(0,'night'),(4,'night'),(5,'morning'),(12,'afternoon'),(17,'evening'),(22,'night')]:
            self.assertEqual(period(t.replace(hour=hour)),expected)

    def test_sample_id_cannot_target_real_participant(self):
        with self.assertRaisesRegex(ValueError,'must differ'):
            generate_sample('/nonexistent.json',sample_participant='p001',source_participant='p001',dry_run=True)

    def test_text_identity_includes_bin_and_is_path_independent(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            for directory in (a,b):Path(directory,'pytorch_model.bin').write_bytes(b'weights-1')
            first=encoder_identity(a);self.assertEqual(first,encoder_identity(b))
            Path(a,'pytorch_model.bin').write_bytes(b'weights-2')
            self.assertNotEqual(first,encoder_identity(a))

    def test_experiment_identity_includes_settings(self):
        self.assertNotEqual(experiment_id('data',{'epochs':1}),experiment_id('data',{'epochs':2}))

    def test_source_identity_includes_content_not_only_max_seq(self):
        store=Store(':memory:');at=dt('2026-11-01T10:00:00+09:00')
        record=answer('a',at,at);store.save(record['kind'],record['id'],'p001',at,record['payload'])
        first=source_identity(store,'p001')['fingerprint']
        payload=dict(record['payload'],received_at=(at+timedelta(days=1)).isoformat())
        store.db.execute('UPDATE records SET payload=?',(json.dumps(payload),));store.db.commit()
        self.assertNotEqual(first,source_identity(store,'p001')['fingerprint']);store.close()

    def test_receipt_time_is_server_owned_and_no_worker_by_default(self):
        store=Store(':memory:')
        with tempfile.TemporaryDirectory() as directory:
            app=create_app(dict(ble_enabled=False,allowed_clients=['testclient'],research_models=directory),store)
            with TestClient(app) as client:
                response=client.post('/api/narrative-events',json=dict(event_id='r',participant_id='p001',detected_at='2026-11-01T10:00:00+09:00',answered_at='2026-11-01T10:00:10+09:00',question={'kind':'currentState','prompt':'state'},answer={'free_text':'{}'},received_at='2000-01-01T00:00:00+00:00'))
                self.assertEqual(response.status_code,200,response.text)
                self.assertGreater(dt(store.rows('p001')[0]['payload']['received_at']),dt('2026-11-01T10:00:00+09:00'))
        store.close()

    def test_four_daily_slots_include_current_and_alternate_context(self):
        class Insights:
            def snapshot(self,*args):return dict(fatigue={},typing={},places={})
        store=Store(':memory:');queue=QuestionQueue(store,Insights());first=[]
        for day in range(2):
            kinds=[]
            for hour in [8,10,12,14,16,18,20,22]:
                t=dt('2026-11-01T00:00:00+09:00')+timedelta(days=day,hours=hour)
                store.save('session',str(t),'p001',t,dict(started_at=(t-timedelta(minutes=1)).isoformat(),ended_at=t.isoformat(),events=[]))
                q=queue.pending('p001',t)
                if q:
                    kinds.append(q['kind']);store.save('narrative',q['id'],'p001',t+timedelta(seconds=10),dict(question={'kind':q['kind']},answer={'label':'answered'}))
            self.assertEqual(len(kinds),4,kinds)
            self.assertIn('currentState',kinds);self.assertIn('nightReflection',kinds);self.assertIn('weatherMood',kinds)
            first.append(kinds[0])
        self.assertEqual(first,['placeCheck','taskCheck']);store.close()

    def test_commit_contract_reconstructs_candidate_without_duplicating_prefix(self):
        t=dt('2026-11-01T10:00:00+09:00')
        events=[(t,'commit',{'text':x}) for x in ['h','e']]
        events += [(t,'kana',{'text':'未確定'}),(t,'context',{'before':'別の文'}),
                   (t,'commit',{'text':'hello','replaces':'he'}),(t,'commit',{'text':'!12'})]
        self.assertEqual(committed_text(events),'hello!12')
        self.assertEqual(committed_text(events[-2:]),'hello!12')

    def test_invalid_commit_is_rejected_before_storage(self):
        from lifelog_server import Session
        from pydantic import ValidationError
        for field in ('text','replaces'):
            payload=dict(session_id='s',participant_id='p001',started_at='2026-11-01T10:00:00+09:00',ended_at='2026-11-01T10:01:00+09:00',
                         events=[dict(time='2026-11-01T10:00:30+09:00',type='commit',data={field:False})])
            with self.assertRaises(ValidationError):Session(**payload)

    def test_store_compatibility_import_and_idempotent_reopen(self):
        from lifelog.store import Store as ExtractedStore
        self.assertIs(Store,ExtractedStore)
        at=dt('2026-11-01T10:00:00+09:00')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'records.sqlite3')
            store=Store(path);self.assertTrue(store.save('heartbeat','id','p001',at,{}));store.close()
            store=Store(path);self.assertFalse(store.save('heartbeat','id','p001',at,{}))
            self.assertEqual(len(store.rows('p001')),1);store.close()

    def test_fixed_four_slots_do_not_get_consumed_by_typing_anomaly(self):
        class Insights:
            def snapshot(self,*args):return dict(fatigue={},typing=dict(unusual=True,evidence_id='window'),places={})
        at=dt('2026-11-01T09:00:00+09:00');store=Store(':memory:');queue=QuestionQueue(store,Insights())
        self.assertEqual(queue.pending('p001',at)['kind'],'weatherMood')
        store.close()
        store=Store(':memory:');queue=QuestionQueue(store,Insights())
        store.save('research_event','prefs','p001',at-timedelta(days=1),dict(kind='preferences',payload=dict(daily_limit=6)))
        self.assertEqual(queue.pending('p001',at)['kind'],'typingReason');store.close()

    def test_atomic_json_can_be_used_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'status.json');atomic_json(path,{'valid':True});self.assertEqual(json.loads(path.read_text()),{'valid':True})
            self.assertEqual(len(list(Path(directory).iterdir())),1)

if __name__=='__main__':unittest.main()
