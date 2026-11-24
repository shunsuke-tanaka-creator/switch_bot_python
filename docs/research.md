# 個人向け三階層LSTM — 実装・運用・未検証事項



## 構成

Macが既存のBLE受信とiPhoneのログ受信を継続する。Ubuntuの同期CLIはMacの全種類エクスポートAPIを差分取得し、各周回の処理後に60秒待機する。Macの `research_upstream` をUbuntuへ設定すると、質問と研究画面のAPIをUbuntuへ中継する。iPhoneの接続先はMacのままでよい。

研究構成の既定は `research_architecture=legacy`。以下の固定24活動の補助分類と回答履歴による個人条件付けはLegacyの仕様である。Discoveryでは活動回答を事後注釈に用い、共有モデルからPlace・Behavior・現在/当日全体の5尺度を推定する。暫定baselineからの項目別配信切替にも接続しない。

- **Micro**: 5分ビンを最大12個まとめるLSTM。日本時間の毎正時で最大1時間に区切り、確認済みの活動変更・訂正の境界でも分割する。タイピングの有無、削除・打ち直しの手がかり、文章、観測状態、健康・予定・開閉を入力する。
- **Meso**: 1日分のMicro表現を順序付きで統合するLSTM。日付境界はAsia/Tokyo。1日最大48区間を超えた場合は明示エラーとし、黙って切り捨てない。
- **Macro**: 月曜0時〜翌月曜0時の固定週。過去4週間の日別特徴と、今週のその時点までのMeso表現を入力するLSTM。過去の日別特徴は同じMicro/Mesoネットワークを再計算したものではなく、日別の観測特徴を射影したもの。
- **上位→下位**: Macro表現で当日のMeso表現を条件付けし、その両方をMicroの状態推定へ渡す。3つのLSTMと融合・出力層を同時に学習する。
- **出力**: 現在と当日全体の5項目を別の出力層で1〜5に推定する。Legacyでは固定24活動の分類を補助学習する。活動回答を状態得点に流用しない。
- **個人適応（Legacy）**: その予測時刻より前に回答済みの履歴から、全体/同時刻帯/同曜日/同活動の基準と存在マスクを作り、個人条件として入力する。推定と本人回答は別に保存する。
- **非使用と欠測**: 非使用の経過時間、heartbeatの鮮度、記録ON/OFF、各特徴の存在マスクを入力。iPhone上の全アプリ使用状況は取得できない。heartbeatが途絶えたことを非使用・睡眠・不在と断定しない。
- HealthKit等は項目別の取得時刻を使用する。旧版に項目別時刻がない値は鮮度を確認できないためモデル特徴に使わない。HealthKitは現在のアプリが取得する日累積/直近1時間集計であり、歩数から作業中の精密な動作を復元するものではない。
- 文章は確定本文だけを特徴化する。前後文脈は保管/エクスポートするが、重複や他者情報の混入を避けるためモデル特徴には使用しない。`text_model` 指定時はローカル多言語SentenceTransformerの意味表現を固定射影して32次元化。未指定時は文字n-gram比較用モードで、文章の意味理解として評価しない。

## セットアップ

### Mac（収集）

既存サーバーを停止してDBをバックアップしてから、変更を導入する。既存仮想環境にも新しい依存を入れる。

```sh
.venv/bin/python -m pip install -r requirements.txt
```

`server-config.json` に `"research_upstream": "http://YOUR_UBUNTU_TAILSCALE_IP:8000"` を追加する。`allowed_clients` にUbuntuのTailscale IPを追加する。Macの `research_enabled` はfalse/未指定のまま（Macでは学習しない）。既存の起動方法で再起動する。Tailscaleの接続制限は維持する。インターネットへAPIを公開しない。

### Ubuntu（RTX 5060 Ti 16GB / RAM 32GB想定）

Python 3.12を基準とする環境と、GPUに対応するNVIDIAドライバー/CUDA版PyTorchを用意する。Blackwell対応のビルドをPyTorch公式のインストール選択画面で選ぶ： https://pytorch.org/get-started/locally/

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
```

CUDA動作を実際に確認する。`deploy/ubuntu.example.json` を `server-config.json` へコピーし、IP/保存先を実環境へ置換する。`research_require_cuda: true` はGPUが利用できなければ学習を停止し、エラーを記録する。

文章モデルは `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` を採用する。固定revisionの配置、学習CLI、サーバー設定、再学習と文字n-gram比較は [文章エンコーダの導入](sentence-encoder.md) を参照。基本・文脈・SSL・部分プーリングへ同じモデルを渡す。実データでの精度改善は未評価。

手動起動例：

```sh
.venv/bin/python lifelog_server.py --config server-config.json --no-ble
# 別ターミナル
.venv/bin/python -m research.cli --participant p001 sync --source http://YOUR_MAC_TAILSCALE_IP:8000 --watch
```

常駐用のsystemd例を `deploy/` に用意した。配置先 `/opt/switch_bot_python` と実行ユーザー `research` は環境に合わせて変更する。`replication.env` には `MAC_SOURCE=http://YOUR_MAC_TAILSCALE_IP:8000` を記入する。自動起動の登録や本番サーバーへのデプロイは、この作業環境から実施していない。

## 全種類のエクスポート・インポート

```sh
# Macで実行。全recordsとdevices/device_namesを含むJSON。入力本文も含む。
.venv/bin/python -m research.cli --participant p001 export /PRIVATE/PATH/lifelog-all-p001.json
# Ubuntuで実行
.venv/bin/python -m research.cli --participant p001 import /PRIVATE/PATH/lifelog-all-p001.json
# 旧iPhoneの file/type/time/data JSONL も同じimportコマンドで取り込める。
```

`GET /api/export/{participant}?after=0&limit=500` はページ形式。最初の `snapshot` を後続ページへ渡す。recordsのseqを固定し、元のIDを保持した冪等取り込みを行う。カーソルはコミット後に永続化する。機器の最新値と別名は各ページで再同期するため、recordsと完全に同一時刻のDB全体スナップショットではない。DB巻き戻りはエラーにし、カーソルを勝手にリセットしない。

「全種類」はDB内の収集記録・回答・訂正・観測状態・推論・質問・提案などと機器メタデータを指す。iPhoneにしか残らない未送信ログ、OSのログ、モデル重みのファイルは含まない。モデルは `data/models/{participant}/{data_mode}/` を別途バックアップする。旧版の `data/models/{participant}/active.pt` と `status.json` は `real` の互換パスとして読み込む。エクスポート・DB・モデル重みはGit対象外。

## サンプルデータとdata_mode

サンプルは実participantへ書かず、`p001-sample` に保存する。投入前の確認は次のコマンドで行う。`--dry-run` はDBとmanifestを変更しない。`--include-today` は推論の1時間鮮度ゲート用に当日セッションを追加する。

```sh
.venv/bin/python -m research.cli --participant p001 generate-sample \
  --calendar data/private/p001-calendar-tags.json \
  --sample-participant p001-sample --dry-run --include-today
```

実投入は収集サーバーを停止してDBをバックアップした後、`--dry-run` を外して実行する。同一入力・seed・オプションは冪等で、既存manifestと入力hashまたは生成条件が異なる場合は中断する。manifestの既定先は `data/sample/p001-sample/manifest.json`。

`data_mode` は `real`（対象participantのみ、既定）、`sample`（`p001-sample`のみ）、`combined`（両方を時刻順に結合）の3種類。学習例：

```sh
.venv/bin/python -m research.cli --participant p001 train --data-mode real
.venv/bin/python -m research.cli --participant p001 train --data-mode sample
.venv/bin/python -m research.cli --participant p001 train --data-mode combined
```

HTTPでは `GET /api/research/{participant}/overview?data_mode=...`、`GET /api/research/{participant}/insights?data_mode=...`、`POST /api/research/{participant}/train` のJSON `{"data_mode":"..."}` を使う。画面はreal以外を常時「サンプル込み」と表示し、回答・訂正・設定変更を無効にする。自動学習モードは `research_training_modes`（既定 `["real"]`）。自動質問と自動推論のparticipant列挙から `p001-sample` を除外する。

## 学習・比較評価

以下の `research.cli train/compare/personalize` と60/20/20分割・活動accuracyの説明はLegacyを対象とする。Discoveryの学習・表示条件は[生活パターン発見](life-pattern-discovery.md)を参照する。

```sh
.venv/bin/python -m research.cli --participant p001 train --data-mode real --epochs 20 --text-model /opt/models/multilingual-sentence-model
.venv/bin/python -m research.cli --participant p001 compare --epochs 20 --text-model /opt/models/multilingual-sentence-model
.venv/bin/python -m research.cli --participant p001 personalize --epochs 20 --text-model /opt/models/multilingual-sentence-model
```

- `train`: 日単位の時系列60/20/20分割。正規化は学習部分のみ。ラベルの回答完了時刻とサーバー受信時刻の遅い方が分割境界前であることを確認する。学習とvalidationでcheckpointを選び、常駐再学習ではtestを評価しない。validationが改善した候補をatomic replaceで採用する。
- `compare`: Microのみ、Micro+Meso、三階層、個人条件なし、文章なしを同じ日付分割・seedで評価し `comparison.json` に保存する。文字n-gramモードと意味表現モードを混同しない。
- `personalize`: 固定モデルと、過去に回答済みのラベルで順次重みを更新するモデルを比較する。予測前に受信済みの同日過去回答も使うイベント単位のオンライン評価（prequential）とし、予測結果と項目別MAE/nを `personalization.json` に保存する。
- 現在/1日全体それぞれの5項目のMAEと件数、活動のaccuracyを出力する。指標が改善したことは実際の比較結果で確認する。合成データの学習成功を研究成果と扱わない。
- Legacyは20状態回答・5日以上を学習開始の下限とする（統計的十分性の保証ではない）。未達は `insufficient_labels`。Discoveryは最低12観測時点・3日、および訓練4例以上と検証例を要求する。尺度の表示には、両構成とも別途、項目ごと学習回答10件・validation回答3件以上の教師支持が必要。曜日差などは十分な週数と各条件の回答数を別途確認する。
- 日内変化の正解はcurrentState、当日全体の正解はnightReflection。夜の回答を各作業の正解へ配布しない。質問表示時刻を予測のcutoffにする。未完了/後から到着したセッションを過去時点で既知の入力として扱わない。旧インポートには元の受信時刻がないため、セッション終了時刻を利用可能時刻とするという制限がある。
- `research_enabled=true` の非upstreamサーバーでは、学習の各周回処理後に3600秒、推論の各周回処理後に60秒待機する。実行間隔には処理時間が加わる。既定では自動研究workerを起動しない。LegacyとDiscoveryでは再学習の判定が異なる（下記「再現性・checkpoint互換性」参照）。ファイルロックで同時起動を防ぐ。データがないときにランダムな状態値は表示しない。
- 比較実験は最終評価として人が実行する。繰り返しtestを見てモデル選択すると評価が楽観的になるため、最終の将来期間を保持する。

## 初日からの暫定推定と項目別のLSTM切替（Legacy限定）

この節は `research_architecture=legacy` の仕様です。参加者の `research_event`（`kind: "preferences"`）で `baseline_estimates_enabled: true` を明示した場合のみ有効です。既定はfalseで、無効時は従来のLSTM表示条件を維持します。質問のゲート・枠数、Legacyの学習開始の20状態回答/5日、モデル全体のvalidationによる昇格規則は変更しません。設定はrealの参加者履歴から読み、推定と切替状態はparticipant/data_modeごとに分離します。

- 現在/1日全体×5項目について、予測時刻より**厳密に前に受信済み**の回答が1件あれば暫定推定できます。個人平均はその項目の利用可能な過去回答すべての平均、直前値は予測cutoff順の最後の回答です。後着回答・将来回答・当該予測の正解を先取りしません。未回答の項目は値を出しません。
- 各周回処理後に60秒待機する既存workerの推論記録に、項目別の `source`（`baseline`/`lstm`）、`provisional`、`baseline_method`（`mean`/`last`/null）を付加します。`baselines.mean` / `baselines.last` は各値と、既に回答された保存済み予測に対する累積MAE/件数を保持します。MAEの小さい方法を暫定値に採用し、同点・履歴なしは直前値です。LSTMへ切り替えた後も両ベースラインを併記し、`lstm_value` と `model_id` に実際のモデル予測を残します。
- 予測と回答の対応は質問表示時刻以前の最新の保存済み推定を用い、保存時刻も回答受信前であることを確認します。同じ推論記録を同じ項目で重複採点しません。未回答・回答後に保存された推定をMAEへ含めず、過去の予測を回答から作り直しません。
- 学習/昇格の後、active LSTMがある場合に項目別の切替を評価します。学習回答10件以上、同じ時系列validation窓にその項目の回答10件以上かつ3日以上が必要です。activeの重みをその窓で再評価し、個人平均/直前値も各予測cutoffで過去の受信済み回答だけを順次参照します。LSTMのMAEが良い方のベースラインMAEより0.1以上小さく、かつその90%以下の場合に合格します。モデルの学習に使った回答やtest窓を切替評価に混ぜません。
- 合格が3つの異なる評価日に続いた項目だけをLSTM配信へ切り替えます。各加算には未使用のvalidation回答が必要で、同じデータの繰り返し再評価・同日再評価は連続回数を増やしません。量・品質の不合格で連続回数をリセットします。モデル全体の昇格と、この配信フラグは独立です。
- 切替後は実際に配信・保存した直近10件の回答済みLSTM予測を、同じ10点の両ベースラインと比較します。LSTMが良い方のベースラインより0.1以上悪い状態が、新しい回答を含む2つの異なる評価日に続けばbaseline配信へ戻し、切替の合格連続回数をリセットします。
- 暫定値はfatigue insight / breakCheckのモデル推定として使用しません。疲労の既存の個人基準・閾値・観測の持続条件は維持します。overview APIの既存の項目に `source` / `provisional` があるので、画面側で暫定値を識別できます。旧推論記録に由来がない場合は従来のLSTM記録として扱います。
- overviewの `validation` はLSTMの検証MAEです。暫定baselineを表示している場合も、その表示値の誤差を表す指標ではありません。画面では表示値の算出元とLSTM検証MAEを別列に表示します。

状態は `data/models/{participant}/{data_mode}/delivery-state.json` にatomic replaceで保存し、評価ごとの10項目の監査は `delivery-audit.jsonl` へappend-onlyで追記します。監査にはactive checkpointの内容SHA256、train/validationの正確なID・cutoff・受信時刻とsplit identity、件数/日数、3方法のvalidation MAE、固定閾値、合格/劣化連続回数、配信元の前後、hold/switch/revertと理由・評価時刻を残します。復帰判断ではliveの同一点MAEと回答ID・モデルIDも記録します。ファイルロックで同時更新を直列化し、監査追記後に状態置換が中断した場合は次回評価で監査から状態を復元します。DBとともにこの2ファイルもバックアップ対象です。

## 質問・訂正・支援・可視化

- 既定4枠はcurrent/night/weather各1とplace/task交互1。currentは8〜21時台、nightは22〜23時台。上限5以上の場合のみイベント提案を優先する。API取得時に質問IDを永続化し、最新1件を返す。iPhoneの表示時刻/回答時刻も保持する。睡眠中に通知で起こす仕組みは追加しない。
- 活動は既存の7大分類/24詳細活動を使用。訂正は開始/終了と活動、任意の5項目をappend-onlyで保存する。訂正の入力時刻より前には、その訂正を既知情報として使わない。現在の実装では訂正の状態得点は終了時点の追加ラベルとして扱う。
- 4種類の比較はLegacyの機能で、その時点より前の本人回答のみを使う。5件未満は比較不足。平均との差と標準化差を表示する。Discoveryのoverviewではこの4比較を計算しない。推定を新たな正解として基準へ入れない。
- 支援は観測に裏付けられた疲労・入力修正の変化を候補化する。1日2回まで/間隔1時間以上、閉じる・拒否・後回し・停止が可能。これは検証前の設定可能な研究ルールであり、最適な介入方策を学習したものではない。実物の家電/ロボットは操作しない。
- PMTTは観測→作業時間枠→日→固定週を関連付け、推定・本人回答・訂正の由来を区別する。現状は時間枠に沿ったグラフ構築であり、未知の生活パターンのトポロジーを自動発見する学習器ではない。
- `GET /api/research/{participant}/pmtt` は表示用の `{nodes, edges}` を返す。完全なグラフの全edgeでsource/target両方の次数を数え、次数2以上のノードと、その両端が残るedgeだけを一度抽出する。抽出後の次数では再判定しない。保存済みレコードと内部の `Engine.pmtt()` は完全なままで、CLI・評価はこの表示用APIを使用しない。
- iPhoneの「研究」画面はローカル配信のWeb画面。本人回答のグラフ、状態推定、4基準比較、訂正、エクスポート、PMTT、モデル検証状況を表示する。入力本文をグラフ表示する必要はない。

## 再現性・checkpoint互換性

この節の `run.json` と環境・Git情報の記録はLegacyおよび該当する比較実験の仕様です。`run.json` はseed、epochs、optimizer設定、Python/依存版、Git SHA/dirty状態、train/validation/testの正確なID・予測時刻・利用可能時刻を保存します。Discoveryはseed、設定、文章モデル内容ID、分割境界、例時刻等をreportへ保存しますが、`run.json` は生成せず、同じ記録形式・情報量ではありません。同じseedでも異なるCPU/GPU・ライブラリ間のビット一致は保証しません。厳密な中断resumeは実装しません。

Legacyの自動再学習の判定にはデータ内容、特徴版、研究コード、実行環境、学習設定、文章モデル内容の実験IDを使用します。成果物が欠けた場合はskipしません。Discoveryの実行指紋は観測・設定・文章モデルに加えて時間単位の時刻（`rolling_hour`）を含むため、データが同じでも時間単位が変われば再学習し得ます。状態APIのtrain JSONには `force: true` を指定できます。LegacyのCLI `train` は毎回学習を試み、`context-train --force` はLegacyのcontextのskipを解除します。

v2特徴はevent timeで5分binへ配置し、**予測時刻までに収集サーバーへ届いたイベント**だけを利用します。ラベルの利用可能時刻は回答/訂正時刻と受信時刻の遅い方です。Mac→Ubuntu同期は元の収集サーバー受信時刻を保持します。従って、この評価は収集サーバーの情報集合を測り、Ubuntu転送遅延を含むエンドツーエンド評価ではありません。旧データには真の受信時刻がなく、回答/セッション終了時刻へfallbackする限界があります。

Legacyは `hierarchical-v2-received-cutoff` と `behavior-context-v2-received-cutoff` に更新しています。v1の入力意味と混ぜず、旧checkpointはエラー状態を返します。昇格前の `active.pt` を `previous.pt` に保持し、失敗した学習でactiveを置換しません。Legacyの `run.json` / `status.json` は直近候補、checkpoint内のreportはその重みの条件です。長期の全実験履歴は別の `--models` ディレクトリで保持してください。

外部文章モデルは設定側のパスを優先し、非hiddenファイル（`.bin`を含む）の内容IDを検証します。同じ内容を別のディレクトリへ移せます。実行時に重みをダウンロードしません。

入力は日本語確定・英数字・記号を `commit.text` で扱います。英語候補は `replaces` に観測済みの接頭辞を記録し、同一窓内の重複を除きます。これは入力イベント特徴であり、編集後の全文を復元する機能ではありません。前後文脈、未確定kana、回答操作は文章特徴に入りません。

## 検証範囲

2026-11-12にLinux CPUの専用venvで、軽量回帰を含む88テストと構文/設定チェックが成功しました。合成データでの短い学習・評価・Checkpoint復元・オンライン個人化を含みます。実データ学習、推定精度、SentenceTransformer実運用重み、CUDA/MPS、Mac BLE、iOSとの実接続は未実行です。[依存](dependencies.md)、[変更一覧](audit-fixes.md)を参照してください。精度・速度・メモリの改善量は未測定です。

```sh
python -B -m unittest discover -s tests -p test_audit_regressions.py -v
python -m pytest -q
```

期待：`OK` / 全テスト `passed`。意図しない出力：`FAILED` / `ERROR` / 依存のimport失敗。

## 明示的に有効化するモデル高度化実験

[評価・SSL・部分プーリング](model-experiments.md)に、参加者単位のMAE_macro、4ベースライン、厳密な学習/SSLカットオフ、共有Encoder、5次元個人バイアス、3条件prequentialを記載しています。新しい `experiment` コマンドは本番の学習・推論・既存 `personalize` から分離され、全機能フラグは既定無効です。
