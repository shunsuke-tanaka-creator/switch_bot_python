# Research API computation

Production baselines supplied for this task: overview real/sample 5.3/18.0s,
insights 23.1/15.3s, context 6.4s, PMTT 6.6s. These are first-request measurements
with 243k real records and a 14.8k-record sample; production was not accessible
from this sandbox.

## Bottlenecks and changes

- Every endpoint loaded, JSON-decoded and sorted complete histories through
  `lifelog/store.py:62` and `research/data_modes.py:40`. Insights additionally
  loaded preferences; sample inference loaded mode/real rows again.
  `research/service.py:161` now shares bounded parsed snapshots through
  `research/api_cache.py:89`. Verified appends decode only new records. Overview,
  context and single-source insights select the kinds they actually consume.
- Sample status/checkpoint checks previously computed `source_identity` merely
  to inspect its manifest hash, rereading and hashing complete training histories
  (`research/data_modes.py:80`). The identical manifest component is now available
  separately (`research/data_modes.py:68`) for display/inference checks. Training
  still uses the full, unchanged identity.
- `Timeline` assigned activity to every state label by rescanning all labels.
  It now indexes activity times and searches the causal 30-minute interval
  (`research/data.py:130`, `research/data.py:157`), preserving correction precedence.
  Delivery/comparisons construct label-only timelines instead of expanding
  sensor events a second time (`research/delivery.py:146`,
  `research/inference.py:20`).
- Weekday trends rescanned entire target/reference cells for each selected week
  in each of 2,000 bootstrap draws. Precomputed week blocks preserve seeded RNG
  calls, sample order and floating-point operations (`research/baseline.py:128`).
- Typing/topology replay sorted unrelated records; it now filters first
  (`research/topology.py:32`, `research/topology.py:245`). Existing insight caches
  now include content, replacing sequence-only and date-only keys that missed
  edited/same-day answers (`research/insights.py:596`,
  `research/discovery_service.py:243`).
- Context catalog/profile replay is shared by insights and context
  (`research/service.py:203`); complete PMTT graph reconstruction is cached
  (`research/service.py:327`). Display projections, limits, provenance and list
  order remain intact. Overview copies persisted estimates before enrichment.

## Freshness and invalidation

The cache mirrors training's **content** fingerprint principle, rather than
using only `MAX(seq)`. SHA-256 streams include participant, sequence, kind, ID,
timestamp and full stored payload, grouped by kind. Each computation selects its
dependencies. No TTL hides new records.

Under the store lock, `PRAGMA data_version`, `total_changes` and the number of
new sequence rows verify same-connection append-only changes. Unchanged
databases reuse snapshots; verified appends extend fingerprints. Updates,
deletes, registry writes or external-connection changes conservatively rescan
content, including edits that leave the maximum sequence unchanged. Active
sample selection is resolved on every access; source identity is part of
combined-mode keys. Context status/checkpoint file revisions also invalidate.
Cached rows are read-only, and sources/artifact slots use bounded LRU eviction.

Heartbeats have a separate content dependency. They update preferences and
inference inputs, and remain available to combined sensor-source selection.
They do **not** invalidate context or baseline/topology artifacts. PMTT normally
stays cached too; it rebuilds if a heartbeat changes combined source-group order,
because that order affects JSON arrays.

Responses are assembled live. Reception keys distinguish `< at` label evidence
from `<= at` visible records; context expiry remains inclusive at 180 seconds.
Overview staleness, fatigue, exercise and inference remain live. Existing
five-minute insight and daily trend reuse retain their time bounds, with added
content/reception invalidation. Sample preview times and real/sample/combined
semantics are unchanged.

## Validation and expected effect

Dependency-light tests cover cached/uncached JSON equality, new answers/names,
heartbeat reuse and live settings, in-place edits, deletes, external writes,
incremental decoding, modes, reception/expiry boundaries and bounded eviction.
API regressions additionally cover all four endpoints in both architectures and
all modes, including combined PMTT ordering and status changes.

Local synthetic timings: 6,000-label timeline initialization 1.86s → 0.19s
(about 10×); a 77-day, single-period trend fixture 8.10s → 0.56s (about 14.5×).
Reference comparisons produced identical payloads. These are computation
benchmarks, not projected production endpoint latencies.

Cold insights should benefit from the algorithm fixes; subsequent endpoint
requests share parsing, and unchanged context/PMTT avoid replay entirely.
Appending still merges ordered row references; external writes take the slower
rescan path. The append query explicitly uses the sequence primary-key range;
SQLite otherwise chose the participant index and rescanned its history. One
parsed history per cached source increases resident memory. Checkpoint
loading/model execution and JSON serialization remain.
Rebenchmark production first requests and repeated polls after deployment.

The sandbox ran 20 unittest cache/algorithm/lazy-sample checks, the exact bootstrap
oracle, compilation and whitespace checks. An additional AST-isolated comparison
against HEAD passed 274 method/endpoint payload checks, including active sample
generations; it substituted same-thread execution for the async worker wrapper.
Full pytest/FastAPI/HTTPX/PyTorch and production measurements could not
run: dependencies were absent and network installation was blocked. `.git` is
read-only, so the changes are uncommitted; nothing was pushed.
