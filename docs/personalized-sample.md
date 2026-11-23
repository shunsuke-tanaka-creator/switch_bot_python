Personalized sample generations use aggregate inputs only. The shipped profile is
`research/profiles/p001.json`, transcribed from the approved 2026-11-24 brief.
Measured values and assumptions have separate fields. The scale means/SDs describe
condition-weighted daily baselines with an SD floor of 0.5; they do not identify
the raw answer marginals. Smoothed aggregate value counts take precedence when
available; otherwise the generator assumes a maximum-entropy distribution with
the supplied bounded moments. The validation report includes the actual production
seven-day conditional baseline as well as raw answer distributions.

Run from the repository root in the normal server/ML Python environment:

```bash
python3 -m research.cli --db data/lifelog.sqlite3 --participant p001 generate-sample \
  --tendencies research/profiles/p001.json --weeks 8 --seed 17 --require-gates
python3 -m research.cli --db data/lifelog.sqlite3 --participant p001 validate-sample \
  --generation p001-g<DIGEST>-sample
python3 -m research.cli --db data/lifelog.sqlite3 --participant p001 activate-sample \
  --generation p001-g<DIGEST>-sample
python3 -m research.cli --db data/lifelog.sqlite3 --participant p001 train \
  --data-mode sample --epochs 20
```

Use the `sample_participant` returned by generation for `<DIGEST>`. Generating
does not activate. Activation and its manifest identity are published in one
SQLite transaction under the Store lock and a database write lock. Checkpoints
from another active generation are rejected. The generation ID incorporates the
profile, generator/consumer code hashes, seed, duration and scenario; modifying
any of those creates a new generation. Existing generations cannot be overwritten.
The original `p001-sample` remains available as a registered legacy generation.
Registered generation rows reject further writes, including through Store.save.

The default period starts on the profile's assumed `start_date`. Edit that
aggregate profile before generating a different preview period. Every generation
has a fixed clock and identity; it never appends a moving “today” session.
Sample views use the immutable generation's last session upload with same-day
step measurements as their preview clock (midnight uploads carrying the preceding
day's steps are excluded). Combined views use the latest source session and expose the selected
sensor source. Inference also accepts an explicit observation time via
`Engine.infer(participant, at, "sample")`; stale explicit cutoffs can correctly
return `observation_unknown`. Generation never issues production questions.

The manifest, `validation.json` and `opportunities.json` are saved beside the DB
under `sample/<generation>/`, or beside an explicitly supplied `--manifest`.
The immutable manifest/report/opportunities are also stored in SQLite, so
activation and validation do not depend on a local filesystem copy. For an
in-memory DB only SQLite copies are saved unless a manifest path is supplied.

`--dry-run` includes mandatory production-parser preflight without writing rows.
`--require-gates` extends the period one week at a time, retaining the profile
and master seed, until every current/daily item has train ≥10 and validation ≥3
and training can start. It stops at 52 weeks. No seed is selected by accuracy.
`--scenario gate-test` fixes the period at seven dense days and requires
training start plus at least one display item: one daily reflection/day makes
daily train ≥10 impossible in seven days. Additional scenarios are `boundary`
(more missing days), `low-steps` and `shuffled-labels` (weak-signal control).

Combined training builds each example, sensor history and personal baseline
inside its source participant's timeline, then pools examples for fitting.
Reports contain per-source support and validation metrics. Combined inference
uses one source's history, preferring fresh real observations, and exposes that
sensor source. Combined support is synthetic preview support.
The overview answers and PMTT nodes carry source identity, and the model panel
shows train/validation support by source. Combined insight snapshots are computed
per source before selecting the displayed source.

Typing events target the production topology features: `log1p(inserts)/8`,
deletion-burst/insertion rate divided by two, and the fraction of insertion
intervals exceeding two seconds. Five synthetic place/action states can create
more than five production nodes because the production map also conditions on
time of day. The report exposes this difference. Steps are cumulative per device
and JST day; state bindings are replayed through the production functions.
Adjacent ten-minute sessions within each state provide the fresh step deltas
required by the production device-carry check. A cached replay is compared
against the production uncached replay in the test suite.

The flags `research_personalized_sample_enabled`,
`research_sample_auto_regenerate`, `research_sample_context_preview_enabled`,
and `research_real_sample_overlay_enabled` default to false. CLI generation and
activation are explicit operations; no automatic regeneration or overlay runs.
Automatic training retains `research_training_modes: ["real"]`. Reserved
participants ending in `-sample` are excluded from question issuance and worker
enumeration. Real selection, fingerprints, gates and questions remain isolated.
The optional context/real-overlay UI extensions are reserved for a later change.

After training, pass `--training-report data/models/p001/sample/status.json` to
`validate-sample` to include validation metrics and warnings for MAE below 0.15.
Synthetic validation is a functional test and does not establish prediction
accuracy on real data.

```bash
python3 -m unittest discover -s tests -p test_personalized_sample.py -v
pytest -q tests/test_personalized_sample.py tests/test_research.py tests/test_insights.py \
  tests/test_baseline_delivery.py tests/test_discovery.py
```

The unittest core needs numpy. The acceptance and discovery tests skip explicitly
when torch or server dependencies are unavailable.
