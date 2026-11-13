# 生活ログ収集・三階層LSTM研究

SwitchBot開閉センサーとiOS [communication-keyboard](https://github.com/shunsuke-tanaka-creator/communication-keyboard) の入力・回答をSQLiteに収集し、状態5尺度と拠点・タスクを研究するPythonサーバーです。Macで収集、MacまたはUbuntuで学習・評価・推論します。研究日はAsia/Tokyo固定です。

**既定は収集のみ（自動学習OFF）です。** GPS分類・行き先予測はありません。BLEの開閉だけで在宅や睡眠を断定しません。

## 構成

| パス | 役割 |
| --- | --- |
| `lifelog_server.py` | FastAPI、入力検証、起動CLI。従来のimport/起動入口を維持 |
| `lifelog/` | SQLite保存、日時、BLE収集 |
| `switchbot_contact_ble.py` | 共用BLEパーサーと診断CLI |
| `research/data.py`, `model.py`, `train.py`, `inference.py`, `evaluation.py` | 特徴、モデル、学習、推論、オンライン評価 |
| `research/context_*.py` | GPSなしの拠点・タスク学習と推論 |
| `research/service.py`, `questions.py`, `insights.py`, `topology.py` | 研究API、質問予算、条件比較、生活パターン |
| `research/io.py`, `provenance.py` | atomic保存、実験ID・環境・分割の記録 |
| `research/cli.py`, `transfer.py`, `sample_data.py` | 同期、移行、隔離した合成データ生成 |
| `tests/`, `test_lifelog_server.py` | 回帰テスト・API/モデルテスト |
| `deploy/`, `docs/` | 常駐設定例と仕様・運用手順 |

小規模のため全面的な `src/` 移行は行っていません。モデルのparameter名・ネットワーク構成は維持します。

## 新規セットアップ

Python 3.12を基準に、用途別の依存を使用します。既存環境の更新はDB・モデルをバックアップしてから行ってください。[依存関係と検証範囲](docs/dependencies.md)も参照。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp server-config.example.json server-config.json
.venv/bin/python lifelog_server.py --config server-config.json --no-ble
```

期待：依存導入が完了し、APIが `127.0.0.1:8000` で起動します。
意図しない出力：`ERROR` / `ModuleNotFoundError` / `address already in use`。venv、Python版、既存プロセスを確認します。既存の `server-config.json` を例で上書きしないでください。

iPhoneへ接続する場合は、設定の `host` をMacのTailscale IP、`allowed_clients` を接続するiPhone/UbuntuのIPへ明示設定します。iPhoneのBackend URLも同じMacへ設定します。新規インストール時のURLは空欄です。公開インターネット向けの認証・利用者間権限分離はありません。転送ヘッダーのIPは信用しません。

BLE利用時は `--no-ble` を外し、MacのBluetoothとターミナルのBluetooth許可を有効にします。`devices` 未指定なら対応する非暗号化SwitchBot開閉センサーを検出します。機器IDを確認し、`devices` で対象を限定できます。macOSのIDはCoreBluetooth UUIDです。

既存の `run_server.sh` と `service.py`（macOS LaunchAgent install/status/start/stop/remove）は利用できます。手動起動と常駐を二重起動しないでください。新規環境ではLaunchAgentは未登録です。収集はログイン・起動中のみで、蓋閉じやスリープ中の継続は保証しません。

## API・保存

| API | 用途 |
| --- | --- |
| `GET /health` | APIとBLEの状態 |
| `POST /api/sessions` | 入力セッション。受信時刻をサーバーが付与 |
| `POST /api/narrative-events`, `/api/context-events` | 回答と文脈。イベントIDで冪等 |
| `POST /api/sensor-events`, `/api/activities` | 疑似センサーと活動回答 |
| `GET /api/questions/pending/{participant}` | 質問があれば200、なければ204。設定済みiOSは204/通信失敗でローカル質問を発生させない |
| `GET /api/dashboard/{participant}`, `/api/records/{participant}/{kind}/{id}` | 日別集計と詳細 |
| `PUT /api/devices/{id}/name` | 機器名 |
| `GET /api/export/{participant}` | ページ付き差分同期 |
| `/research/{participant}`, `/research/{participant}/insights`, `/research/{participant}/context` | 研究画面 |

既定保存先は `data/lifelog.sqlite3`、モデルは `data/models/{participant}/{data_mode}/`、合成manifestは `data/sample/{sample-participant}/manifest.json`。DB、入力本文、ログ、モデルはGit対象外です。入力ログは確定本文と前後文脈を含みますが、モデルの文章特徴には確定本文のみを使用します。外部AIサービスへの送信はありません。保持期限や自動削除はありません。

## 研究・互換性

- 状態：Micro（5分→最大1時間）・Meso（日）・Macro（固定週＋過去4週）のLSTM。現在/日全体の5尺度と補助活動分類。
- 質問：既定4件は current 1、night 1、weather 1、place/task日替わり1。条件・観測不足では4件未満です。イベント起点の追加提案は上限を明示的に5以上へ増やした場合に使います。
- 評価：**イベントごとのオンライン個人化（2=B）**。対象予測より前に受信済みの同日回答を利用できます。日単位凍結評価と混ぜません。
- 復元：**推論checkpointと同条件での再学習（10=A）**。optimizer/RNG途中状態の厳密resumeはありません。
- 特徴をv2へ更新したため旧v1 checkpointは `incompatible_checkpoint`。元ファイルとDBは削除せず、承認した環境で再学習・評価して切り替えます。既存の旧保存パスは引き続き探索します。

詳細：[研究・評価](docs/research.md) / [拠点・タスク](docs/behavior-context.md) / [質問・イベント契約](docs/otenki-workflow.md) / [今回の変更と未検証事項](docs/audit-fixes.md)。

## 検証

```sh
python -B -m unittest discover -s tests -p test_audit_regressions.py -v
python -m pytest -q
```

期待：軽量回帰テストは `OK`、全体テストは `passed`。前者はnumpy/FastAPI/httpx、後者は開発依存とPyTorchが必要です。
意図しない出力：`FAILED` / `ERROR` / import失敗。合成データのテスト通過を実データ精度の根拠にはしません。
