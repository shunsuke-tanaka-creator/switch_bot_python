# お天気分・自動トポロジー研究版

## 構成

主手法の `HierarchicalLSTM.micro / meso / macro`、98次元入力、48次元の各表現、状態5尺度と24活動の出力は維持する。新しい機能は同じ観測記録を利用する補助モジュールであり、三階層の代替モデルではない。標準の対象者は従来どおり `p001`。

| ファイル | 担当 |
| --- | --- |
| `research/baseline.py` | 本人回答による曜日・時間帯・活動の条件付き基準。日を均等化。曜日×時間帯の探索的比較 |
| `research/topology.py` | 入力窓、欠測に対応した増殖型プロトタイプ、遷移辺・近接ノード統合・活性減衰、場所の移動確率 |
| `research/insights.py` | 疲労上昇・入力修正増加の検出、ノードと場所の関連、確認済み対処結果による推薦 |
| `research/questions.py` | 永続質問ID、期限、後回し、停止、表示履歴と予約を使う共通予算 |
| `research/events.py` | iPhone/Web共通の `research-v2` 入力検証 |
| `research/insights.html` | 天気4分類、傾向、場所・生活パターン、質問・設定の画面 |

実装は既存のGWR原著・Graph Transformerそのものの再現ではない。修論の自己成長メモリを参考に、観測の固定定義による増殖型プロトタイプを実装した。原著GWRやGraph Transformerへの置き換えは、この研究版の必須条件ではない。外部LLM/APIキーや追加の課金サービスは使わない。

## 起動

サーバーのリポジトリのルートで実行する。既存環境を更新する前に `data/lifelog.sqlite3` と `data/models/` をバックアップする。新規の `research_event` は既存records表に追記するので、DBの破壊的マイグレーションは不要。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install 'torch>=2.6,<3'
.venv/bin/python lifelog_server.py --config server-config.json
```

期待：uvicornが設定のhost/portで起動する。`/health` が `status: ok`。`http://100.106.185.3:8000/research/p001/insights` を開ける。

意図しない出力：`ModuleNotFoundError`、`address already in use`、`This device is not allowed`。それぞれvenv、既存プロセス、設定の `allowed_clients` を確認する。最初の `insufficient_labels` は異常ではなく、本人回答が不足している状態。

`server-config.json` はGit対象外。既存の設定に以下を加える。`research_models` の場所は現行設定を維持する。

```json
{
  "research_enabled": true,
  "research_models": "data/models",
  "fatigue_delta": 1.0,
  "fatigue_z": 2.0,
  "typing_min_inserts": 100,
  "pattern_radius": 0.35,
  "trend_min_weeks": 6
}
```

Ubuntuへ同期する構成では、これまでどおりMacで収集し `research_upstream` をUbuntuに向ける。新イベントはMacにも保存し、Ubuntuにも即時転送する。転送失敗は503として返し、iPhoneのoutboxから再送する。全種類export/replicationの既存recordsに含まれるので、新イベントだけの別バックアップは不要。場所の座標はサーバーに届かないため、iPhoneのアプリデータも別途保持する。

## APIとデータの区別

`POST /api/research/p001/events`：`schema_version`, `event_id`, `participant_id`, `occurred_at`, `kind`, `payload`。

| kind | payload | 用途 |
| --- | --- | --- |
| `weather_mood` | `weather`: happy / normal / sad / angry、任意のquestion_id | 晴れ・くもり・雨・雷。疲労や気分1〜5へ換算しない |
| `place_visit` | place_id, phase: arrival/departure, accuracy_m, 任意のdevice_id | 座標のない滞在クラス。低精度はマップへ反映しない |
| `typing_reason` | question_id, reason: mistype/rewrite/conversion/unknown | 修正の意味の本人回答 |
| `intervention_outcome` | question_id（元提案）, action: short_break/refresh/switch_task, executed, helpfulness | 実行と役立ちを分離。未実行をhelpfulにできない |
| `question_state` | question_id, state: shown/scheduled/deferred/declined/stop/resume | 通知・親アプリ・キーボードの制御 |
| `preferences` | questions_enabled, interventions_enabled, daily_limit, intervention_limit, gap_minutesの一部 | 既定は合計4回・内提案2回・60分間隔 |

POSTはevent_idで冪等。別のparticipantや未発行質問の参照、未提示行動の効果、座標の混入、不正な天気・数値は422。`received_at` はサーバーが付与し、端末の発生時刻とは区別する。イベントは単独送信できるため、外出中にキーボードを使わなくても後で同期できる。

`GET /api/research/p001/insights`：baseline, fatigue, typing, trends, weather, places, patterns, preferencesを返す。

従来の `GET /api/questions/pending/p001` は新しい質問も返す。`currentState`, `weatherMood`, `typingReason`, `supportOutcome` を追加したiOS版と組み合わせる。質問のcamelCaseキー、contextのsnake_caseキー、ミリ秒付きISO8601日時は維持する。`expiresAt` と任意の `relatedID` を追加。

## アルゴリズムと初期値

- 状態の正解は現在状態・夜の振り返りの本人回答だけ。未回答を0点にせず、天気・推定・クリックから状態の正解を作らない。質問中の操作は `purpose=assessment` で観測特徴から除外する。
- 条件付き疲労基準は直近56日の本人回答を使用。曜日・時間帯・活動に重みを付け、日ごとに均等化して希薄な条件を全体へ縮約。最低5日未満は不足。
- 急上昇は普段との差1点以上・標準化差2以上が、観測のある別々の5分窓2つで連続した場合。異なるモデルfingerprintの出力を持続の裏付けとして混ぜない。絶対値4以上の高疲労フラグは別。
- 曜日傾向は最大84日を使用し、最低6週間・対象4日・比較8日。同じ活動区分内で比較し、週単位ブートストラップと固定28比較の補正を行う。探索的な関連であり曜日の因果効果ではない。
- 入力は非重複10分窓。文字入力の操作とユーザーの削除のみを使い、同じ端末・入力モード、100操作以上、過去20窓・5日以上を要求。修正頻度が基準99パーセンタイルを2窓連続で超えると候補。削除と誤入力は同一視しない。誤入力/推敲などを本人に確認する。
- 生活ノードは固定の観測特徴（入力量・修正頻度・ポーズ・周期時刻）から自動作成。観測不足は保留、近いものを更新、遠いものを追加。近づいた重心は統合し、参照活性は30日スケールで減衰。異なる3日に現れると「定着」。承認は不要。256ノードの上限を画面・APIで明示し、上限時に遠い観測を無理に既存クラスへ押し込まない。原記録は減衰を理由に削除しない。
- LSTMの48次元表現を生活クラスの固定座標として蓄積しないため、毎時の再学習からマップが独立する。原記録から同じ版のマップを再構築でき、サーバー再起動・replicationで同じIDが得られる。
- 場所はiPhoneで匿名クラスに集約。サーバーは場所間の遷移、曜日・時刻の出現数、平滑化付き行き先分布を作る。十分な日数がなければ予測しない。最終位置が古い場合は現在地不明とし、時刻だけの行き先予測は観測した到着と区別する。
- 休憩などの推薦は単一値後件の重み付きルールで順位付けし、似た検出理由で本人が実行・評価した結果を加味する。「受け入れた」だけでは成功体験にしない。
- 予定の更新時刻が新しく、現在が予定内なら自動質問を保留。昼間は夜の振り返り用の1枠を残す。質問GETは同一IDを予約し、表示済み履歴＋有効な予約で複数画面からの上限超過を防ぐ。既定では8時前は質問しない。

## 検証

```bash
.venv/bin/python -m pytest -q
```

期待：すべてのテストが `passed`。三階層すべてへの勾配、学習・時系列評価、未学習の抑制、未来データ除外、新イベントのAPI・冪等性・再構築・質問予算を検証する。

意図しない出力：`FAILED`、`ERROR`、import失敗。モデルや閾値を実データで最適化する前に修正する。合成データの通過を研究上の有効性と扱わない。

対になるiOSリポジトリの `docs/otenki-workflow.md` にXcode・実機確認手順を記載。iOSビルドと位置・通知の実機試験は、このLinux環境では実行できない。
