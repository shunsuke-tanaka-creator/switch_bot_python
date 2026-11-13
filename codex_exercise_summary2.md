# 運動提案のトポロジカル状態化 実装サマリー

## 1. 状態表現と履歴の調査結果

- 既存の `GrowingMap` / `behavior_map` は、保存済みsessionから10分窓を再構成し、入力量、修正率、休止率、時刻のsin/cosを使う増殖型プロトタイプだった。ノードと遷移はAPI応答時に決定論的に再計算され、各観測のノード割当ては永続化されていなかった。
- 場所とタスクはGrowingMapとは別のContextLSTMで推定され、ライブ結果は `context_inference` として永続化されていた。歩数はsession内の日累積値と、前回変更で追加した `step_snapshot` に保存されていた。
- 既存レコードには「その時点のGrowingMap状態ID」と歩数の結合がない。後から地図全体を再計算するとノード統合やContext推定の利用可能時刻が変わり得るため、導入前履歴から信頼できる状態別歩数基準を遡及再構成しない設計とした。
- 追加: GrowingMapノードは場所ID・タスクID（提案上のアクション）が異なる観測を厳密に別状態として扱い、ノードに場所・アクションのラベルと観測割当てを出力する。トポロジー版は `observation-topology-v2-state-context` とした。
- 追加: 鮮度付き歩数と直近状態を `state_step_snapshot` に冪等保存する。状態ID、場所、アクション、状態中心特徴、歩数、測定時刻、元session IDを保持し、通常のrecords export/ingestで移送される。参加者の入力本文は追加レコードへ保存しない。

## 2. 新しい検出器の定義

- 文脈: 現在または直近15分以内の `state_step_snapshot` が示すGrowingMap状態。状態は同じ場所ID・同じアクションIDを持つノードである。
- 運動量: 同一端末・同一日の連続した日累積歩数スナップショットの差分を使う。観測間隔も15分以内に限定し、長い欠測中の歩数を直近状態へ誤帰属させない。差分は後側スナップショットに永続化された状態へ帰属し、状態ごと・日ごとに合計する。
- 基準: 同一参加者・同一端末・同一状態の過去日別状態内歩数。中央値を通常値、25パーセンタイルを下限とする。異なる状態、参加者、端末、トポロジー版は混ぜない。
- 閾値: 今日の状態内歩数が25パーセンタイル未満で、中央値との差が500歩以上、かつ中央値の25%以上の場合を不足とする。
- ゲート: 過去3出現日で暫定、5出現日で正式。現在の状態結合歩数がない、歩数またはsessionが15分より古い、当日の正の歩数差分がない、現在状態の差分を計算できない場合は提案しない。
- METs表、100歩/分、`活動分数 = 歩行相当分数 × 3.0 / METs`、5分単位切上げ、最大4候補は維持した。
- QuestionQueueの08:00–24:00、共通上限、間隔、延期・停止、同日同状態クールダウン、`exercise_proposals_enabled` 既定無効を維持した。
- PMTTには状態ID・版、場所・アクション、通常値、25パーセンタイル、今日値、不足歩数、基準日数・回数、閾値、METs候補、根拠レコードIDを保存する。
- 提案文とproposal-preview表示から運動提案用の曜日・同時刻表現を除き、場所・行動で普段より歩数が少ないことを示す文面へ変更した。
- `research/data.py` の特徴量一覧、モデル入力、階層モデルの版は変更していない。

## 3. 変更ファイル

- `research/topology.py`
- `research/insights.py`
- `research/questions.py`
- `research/service.py`
- `research/insights.html`
- `lifelog_server.py`
- `tests/test_insights.py`
- `docs/otenki-workflow.md`
- `codex_exercise_summary2.md`

## 4. 実行できたチェック

- 指定venv `/Users/tanakashunsuke/Desktop/python/python01` を使用した。
- `python -B -m compileall -q lifelog_server.py research tests test_lifelog_server.py`: 成功。
- `python -B -m unittest discover -s tests -p test_audit_regressions.py -v`: 15件成功。
- pytest非依存の合成統合確認: 同一状態基準、異状態分離、履歴不足ゲート、METs候補4件、カレンダー表現なし、状態結合保存が成功。
- FastAPI TestClientによるsession保存とproposal-preview確認: 成功。
- `node --check` による `research/insights.html` 内JavaScript確認: 成功。
- `git diff --check`: 成功。
- `python -m pytest -q`: venvにpytestがないため `No module named pytest`。ライブラリは追加していない。

## 5. 残課題

- 実参加者データで、3日/5日、15分、500歩、25%の閾値と提案頻度を校正する必要がある。
- `state_step_snapshot` は、十分な入力窓、鮮度付き歩数、IDが確定した場所・タスク推定が近接した場合だけ蓄積される。観測が疎な環境では基準形成に時間がかかる。
- 導入前の曜日・拠点・時刻基準は意図的に利用しないため、新方式の履歴が3出現日に達するまで運動提案は抑止される。
- pytest全体と実機iOS表示は、pytestを備えた別環境と開発者側ビルドで確認する必要がある。
- 性能面では、未結合の新しい歩数がある時にGrowingMapを再生する処理が増えた。未結合歩数がない周期は地図を再生しない。
- 新しい外部入力面や認証変更はない。追加保存する状態ID・中心特徴・場所・アクションは既存研究recordsと同じ保持・export対象になるため、既存データと同様に非公開で管理する。
- 作業リポジトリの `.git` が環境上読み取り専用で直接コミットできなかったため、一時クローンの同ブランチでコミットを作成した。GitHubへのpushは `Could not resolve host: github.com` で失敗したため、最新コミットを含む `feat-exercise-proposal-mets.bundle` をリポジトリ直下に残した。
