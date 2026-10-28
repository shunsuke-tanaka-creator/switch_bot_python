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

Python 3.12以上と、GPUに対応するNVIDIAドライバー/CUDA版PyTorchを用意する。Blackwell対応のビルドをPyTorch公式のインストール選択画面で選ぶ： https://pytorch.org/get-started/locally/

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

「全種類」はDB内の収集記録・回答・訂正・観測状態・推論・質問・提案などと機器メタデータを指す。iPhoneにしか残らない未送信ログ、OSのログ、モデル重みのファイルは含まない。モデルは `data/models/{participant}/` を別途バックアップする。エクスポート・DB・モデル重みはGit対象外。

## 学習・比較評価

```sh
.venv/bin/python -m research.cli --participant p001 train --epochs 20 --text-model /opt/models/multilingual-sentence-model
.venv/bin/python -m research.cli --participant p001 compare --epochs 20 --text-model /opt/models/multilingual-sentence-model
.venv/bin/python -m research.cli --participant p001 personalize --epochs 20 --text-model /opt/models/multilingual-sentence-model
```

- `train`: 日単位の時系列60/20/20分割。正規化は学習部分のみ。ラベルの回答完了日も分割前であることを確認する。学習とvalidationでcheckpointを選び、常駐再学習ではtestを評価しない。validationが改善した候補をatomic replaceで採用する。
- `compare`: Microのみ、Micro+Meso、三階層、個人条件なし、文章なしを同じ日付分割・seedで評価し `comparison.json` に保存する。文字n-gramモードと意味表現モードを混同しない。
- `personalize`: 固定モデルと、過去に回答済みのラベルで順次重みを更新するモデルを比較する。評価日の予測にその日の回答を使わず、予測結果と項目別MAE/nを `personalization.json` に保存する。
- 現在/1日全体それぞれの5項目のMAEと件数、活動のaccuracyを出力する。指標が改善したことは実際の比較結果で確認する。合成データの学習成功を研究成果と扱わない。
- 全体で20状態回答・5日以上を学習開始の最低ゲートとする（統計的十分性の保証ではない）。表示はさらに項目ごと学習回答10件・validation回答3件以上が必要。未達は `insufficient_labels`。曜日差などは十分な週数と各条件の回答数を別途確認する。
- 日内変化の正解はcurrentState、当日全体の正解はnightReflection。夜の回答を各作業の正解へ配布しない。質問表示時刻を予測のcutoffにする。未完了/後から到着したセッションを過去時点で既知の入力として扱わない。旧インポートには元の受信時刻がないため、セッション終了時刻を利用可能時刻とするという制限がある。
- 1時間ごとに新規回答を確認して再学習する。同一データの再学習を避け、ファイルロックで同時起動を防ぐ。1分ごとに推論記録を作成する。データがないときにランダムな状態値は表示しない。
- 比較実験は最終評価として人が実行する。繰り返しtestを見てモデル選択すると評価が楽観的になるため、最終の将来期間を保持する。

## 質問・訂正・支援・可視化

- 8〜21時台は現在の5項目を3時間以上の間隔で質問する。22〜23時台は当日全体を優先する。API取得時に質問IDを永続化し、最新1件を返す。iPhoneの表示時刻/回答時刻も保持する。睡眠中に通知で起こす仕組みは追加しない。
- 活動は既存の7大分類/24詳細活動を使用。訂正は開始/終了と活動、任意の5項目をappend-onlyで保存する。訂正の入力時刻より前には、その訂正を既知情報として使わない。現在の実装では訂正の状態得点は終了時点の追加ラベルとして扱う。
- 4種類の比較はその時点より前の本人回答のみ。5件未満は比較不足。平均との差と標準化差を表示する。推定を新たな正解として基準へ入れない。
- 支援は推定疲労4以上かつ全体基準から1標準偏差以上の場合に候補化する。1日2回まで/間隔1時間以上、閉じる・拒否・後回し・停止が可能。これは検証前の設定可能な研究ルールであり、最適な介入方策を学習したものではない。実物の家電/ロボットは操作しない。
- PMTTは観測→作業時間枠→日→固定週を関連付け、推定・本人回答・訂正の由来を区別する。現状は時間枠に沿ったグラフ構築であり、未知の生活パターンのトポロジーを自動発見する学習器ではない。
- iPhoneの「研究」画面はローカル配信のWeb画面。本人回答のグラフ、状態推定、4基準比較、訂正、エクスポート、PMTT、モデル検証状況を表示する。入力本文をグラフ表示する必要はない。

## 検証済み / 未達・未検証

### この環境で確認済み

- Python CPU上で18件のテスト（既存API 8件＋研究パイプライン10件）に合格。三層への勾配、未来データ不使用、日付/週境界、欠測マスク、合成データの学習・checkpoint読込、5方式の比較、逐次個人適応比較、エクスポート/冪等同期、質問間隔を確認。
- 旧セッションJSONLの取り込みと、状態回答不足時の学習拒否を確認。本文そのものはリポジトリへ保存していない。

### 未達・未検証（完了扱いにしない）

1. **実データでの推定精度、三階層の優位性、個人適応の改善は未検証**。5項目の回答・活動・健康/センサーを含む全種類エクスポートが必要。週/曜日の効果には複数週が必要。
2. RTX 5060 TiでのCUDA実行、速度・VRAM消費、Ubuntu本番常駐、Mac BLE→Ubuntu→iPhoneの実接続は未検証。
3. この環境にXcode/iOS SDKがないため、iOSのコンパイル・XCTest・実機UI・WebKitダウンロードは未検証。Web画面はJSDOMでAPI応答との契約・DOM生成・停止操作を確認したが、実ブラウザーでの描画は未確認。アプリ側の手順を参照。
4. ローカル多言語SentenceTransformerの実重みでの意味特徴はオプションとして実装したが、本環境ではモデル未配置のため動作/精度未検証。未指定時は文字特徴として明示する。
5. 姿勢・振動・全アプリ利用状況・在室の確定は収集していない。2台のドアだけで本人の在室/外出を断定しない。
6. 休憩提案の因果的効果/無作為割付比較、学習による介入最適化、学習したトポロジーによるPMTTパターン発見は未実装。現状は制約付きのルール提案と履歴記録。
7. 履歴訂正で過去の全レコードを自動再解釈する仕組みは未実装。元記録を保持した追加ラベル/区間情報として扱う。
8. 利用者間の認証・権限分離を備える公開サービスではない。元のTailscale単独利用者構成を維持する。

## テスト

```sh
python -m pip install -r requirements.txt
# CPUテスト用PyTorch、または既に導入したCUDA版を使用
python -m pytest -q
```

API単体テストだけなら `pytest -q test_lifelog_server.py`。研究パイプラインのテストにはPyTorchが必要。
