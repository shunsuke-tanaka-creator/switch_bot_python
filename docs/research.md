# 個人向け三階層LSTM — 実装・運用・未検証事項



## 構成

Macが既存のBLE受信とiPhoneのログ受信を継続する。UbuntuがMacの全種類エクスポートAPIを60秒ごとに差分取得し、学習・推論する。Macの `research_upstream` をUbuntuへ設定すると、質問と研究画面のAPIをUbuntuへ中継する。iPhoneの接続先はMacのままでよい。

- **Micro**: 5分ビンを最大12個まとめるLSTM。日本時間の毎正時で最大1時間に区切り、確認済みの活動変更・訂正の境界でも分割する。タイピングの有無、削除・打ち直しの手がかり、文章、観測状態、健康・予定・開閉を入力する。
- **Meso**: 1日分のMicro表現を順序付きで統合するLSTM。日付境界はAsia/Tokyo。1日最大48区間を超えた場合は明示エラーとし、黙って切り捨てない。
- **Macro**: 月曜0時〜翌月曜0時の固定週。過去4週間の日別特徴と、今週のその時点までのMeso表現を入力するLSTM。過去の日別特徴は同じMicro/Mesoネットワークを再計算したものではなく、日別の観測特徴を射影したもの。
- **上位→下位**: Macro表現で当日のMeso表現を条件付けし、その両方をMicroの状態推定へ渡す。3つのLSTMと融合・出力層を同時に学習する。
- **出力**: 現在と当日全体の5項目を別の出力層で1〜5に推定する。活動分類は補助学習。活動回答を状態得点に流用しない。
- **個人適応**: その予測時刻より前に回答済みの履歴から、全体/同時刻帯/同曜日/同活動の基準と存在マスクを作り、個人条件として入力する。推定と本人回答は別に保存する。
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

文章の意味を利用する場合は、例えば `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` の信頼できるモデルを事前にダウンロードし、ローカル保存先を `text_model` に指定する。参加者の文章は外部へ送信しない。実行時は `local_files_only=True / trust_remote_code=False`。重みと設定のSHA256をcheckpointに記録し、推論時のモデル不一致は停止する。モデルファイルをGitへ入れない。

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

```sh
.venv/bin/python -m research.cli --participant p001 train --data-mode real --epochs 20 --text-model /opt/models/multilingual-sentence-model
.venv/bin/python -m research.cli --participant p001 compare --epochs 20 --text-model /opt/models/multilingual-sentence-model
.venv/bin/python -m research.cli --participant p001 personalize --epochs 20 --text-model /opt/models/multilingual-sentence-model
```

- `train`: 日単位の時系列60/20/20分割。正規化は学習部分のみ。ラベルの回答完了時刻とサーバー受信時刻の遅い方が分割境界前であることを確認する。学習とvalidationでcheckpointを選び、常駐再学習ではtestを評価しない。validationが改善した候補をatomic replaceで採用する。
- `compare`: Microのみ、Micro+Meso、三階層、個人条件なし、文章なしを同じ日付分割・seedで評価し `comparison.json` に保存する。文字n-gramモードと意味表現モードを混同しない。
- `personalize`: 固定モデルと、過去に回答済みのラベルで順次重みを更新するモデルを比較する。予測前に受信済みの同日過去回答も使うイベント単位のオンライン評価（prequential）とし、予測結果と項目別MAE/nを `personalization.json` に保存する。
- 現在/1日全体それぞれの5項目のMAEと件数、活動のaccuracyを出力する。指標が改善したことは実際の比較結果で確認する。合成データの学習成功を研究成果と扱わない。
- 全体で20状態回答・5日以上を学習開始の最低ゲートとする（統計的十分性の保証ではない）。表示はさらに項目ごと学習回答10件・validation回答3件以上が必要。未達は `insufficient_labels`。曜日差などは十分な週数と各条件の回答数を別途確認する。
- 日内変化の正解はcurrentState、当日全体の正解はnightReflection。夜の回答を各作業の正解へ配布しない。質問表示時刻を予測のcutoffにする。未完了/後から到着したセッションを過去時点で既知の入力として扱わない。旧インポートには元の受信時刻がないため、セッション終了時刻を利用可能時刻とするという制限がある。
- 1時間ごとに新規回答を確認して再学習する。同一データの再学習を避け、ファイルロックで同時起動を防ぐ。1分ごとに推論記録を作成する。データがないときにランダムな状態値は表示しない。
- 比較実験は最終評価として人が実行する。繰り返しtestを見てモデル選択すると評価が楽観的になるため、最終の将来期間を保持する。

## 質問・訂正・支援・可視化

- 既定4枠はcurrent/night/weather各1とplace/task交互1。currentは8〜21時台、nightは22〜23時台。上限5以上の場合のみイベント提案を優先する。API取得時に質問IDを永続化し、最新1件を返す。iPhoneの表示時刻/回答時刻も保持する。睡眠中に通知で起こす仕組みは追加しない。
- 活動は既存の7大分類/24詳細活動を使用。訂正は開始/終了と活動、任意の5項目をappend-onlyで保存する。訂正の入力時刻より前には、その訂正を既知情報として使わない。現在の実装では訂正の状態得点は終了時点の追加ラベルとして扱う。
- 4種類の比較はその時点より前の本人回答のみ。5件未満は比較不足。平均との差と標準化差を表示する。推定を新たな正解として基準へ入れない。
- 支援は観測に裏付けられた疲労・入力修正の変化を候補化する。1日2回まで/間隔1時間以上、閉じる・拒否・後回し・停止が可能。これは検証前の設定可能な研究ルールであり、最適な介入方策を学習したものではない。実物の家電/ロボットは操作しない。
- PMTTは観測→作業時間枠→日→固定週を関連付け、推定・本人回答・訂正の由来を区別する。現状は時間枠に沿ったグラフ構築であり、未知の生活パターンのトポロジーを自動発見する学習器ではない。
- iPhoneの「研究」画面はローカル配信のWeb画面。本人回答のグラフ、状態推定、4基準比較、訂正、エクスポート、PMTT、モデル検証状況を表示する。入力本文をグラフ表示する必要はない。

## 再現性・checkpoint互換性

`run.json` はseed、epochs、optimizer設定、Python/依存版、Git SHA/dirty状態、train/validation/testの正確なID・予測時刻・利用可能時刻を保存します。同じseedでも異なるCPU/GPU・ライブラリ間のビット一致は保証しません。厳密な中断resumeは実装しません。

自動再学習の判定にはデータ内容、特徴版、研究コード、実行環境、学習設定、文章モデル内容の実験IDを使用します。成果物が欠けた場合はskipしません。状態APIのtrain JSONには `force: true` を指定できます。CLIの `train` は毎回学習を試み、`context-train --force` はcontextのskipを解除します。

v2特徴はevent timeで5分binへ配置し、**予測時刻までに収集サーバーへ届いたイベント**だけを利用します。ラベルの利用可能時刻は回答/訂正時刻と受信時刻の遅い方です。Mac→Ubuntu同期は元の収集サーバー受信時刻を保持します。従って、この評価は収集サーバーの情報集合を測り、Ubuntu転送遅延を含むエンドツーエンド評価ではありません。旧データには真の受信時刻がなく、回答/セッション終了時刻へfallbackする限界があります。

`hierarchical-v2-received-cutoff` と `behavior-context-v2-received-cutoff` に更新しています。v1の入力意味と混ぜず、旧checkpointはエラー状態を返します。昇格前の `active.pt` を `previous.pt` に保持し、失敗した学習でactiveを置換しません。`run.json` / `status.json` は直近候補、checkpoint内のreportはその重みの条件です。長期の全実験履歴は別の `--models` ディレクトリで保持してください。

外部文章モデルは設定側のパスを優先し、非hiddenファイル（`.bin`を含む）の内容IDを検証します。同じ内容を別のディレクトリへ移せます。実行時に重みをダウンロードしません。

入力は日本語確定・英数字・記号を `commit.text` で扱います。英語候補は `replaces` に観測済みの接頭辞を記録し、同一窓内の重複を除きます。これは入力イベント特徴であり、編集後の全文を復元する機能ではありません。前後文脈、未確定kana、回答操作は文章特徴に入りません。

## 検証範囲

2026-11-12にLinux CPUの専用venvで、軽量回帰を含む88テストと構文/設定チェックが成功しました。合成データでの短い学習・評価・Checkpoint復元・オンライン個人化を含みます。実データ学習、推定精度、SentenceTransformer実運用重み、CUDA/MPS、Mac BLE、iOSとの実接続は未実行です。[依存](dependencies.md)、[変更一覧](audit-fixes.md)を参照してください。精度・速度・メモリの改善量は未測定です。

```sh
python -B -m unittest discover -s tests -p test_audit_regressions.py -v
python -m pytest -q
```

期待：`OK` / 全テスト `passed`。意図しない出力：`FAILED` / `ERROR` / 依存のimport失敗。
