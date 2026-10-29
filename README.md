# MacBook 生活ログサーバー

SwitchBot開閉センサーのBLE受信と、iPhoneのキーボード・活動回答のログをSQLiteに保存します。iPhoneはTailscale経由で統合ダッシュボードを取得します。

- MacBook: `100.106.185.3:8000`
- iPhone: `100.124.206.1`
- 参加者ID: `p001`
- 想定設置場所: **玄関 / 寝室兼作業部屋**

## 起動

```sh
cd /Users/tanakashunsuke/Desktop/python/20261008
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
./run_server.sh
```

MacとiPhoneのTailscaleを有効にします。MacのBluetoothをONにして、Python/ターミナルへのBluetoothアクセスを許可してください。Macを電源につなぎ、蓋を開けた状態で使用します。スクリプトは `caffeinate` でアイドルスリープを抑止しますが、蓋閉じ・明示的スリープ・ログアウト中は収集できません。

このMacではLaunchAgentをインストール済みです。通常は手動の `run_server.sh` と二重起動しないでください。

```sh
.venv/bin/python service.py status   # 稼働状態
.venv/bin/python service.py stop     # 今回のログイン中は停止
.venv/bin/python service.py start    # 再開
.venv/bin/python service.py install  # 初回登録 / 設定の再登録
.venv/bin/python service.py remove   # 自動起動を解除（データは残す）
```

LaunchAgentはログイン時に起動し、異常終了時は再起動します。Tailscaleがまだ起動していない場合も再試行します。プログラムを変更した後は `stop` → `start` で反映します。ログは `data/server.stdout.log` / `data/server.stderr.log`、DBは `data/lifelog.sqlite3` です。

## iPhoneの設定

1. communication-keyboardの更新版をXcodeからiPhoneにインストールします。
2. 「くらしのログ」を開き、接続先が `http://100.106.185.3:8000`、参加者IDが `p001` であることを確認します。以前URLを空欄にしていた場合は「このMacBookに接続する」を押します。別URLが保存されている場合は歯車の設定画面で変更します。
3. キーボードをiOS設定で追加し、「フルアクセスを許可」をON、アプリのプライベートモードをOFFにすると入力ログを記録できます。
4. 玄関のドアだけを開閉し、変化した機器をタップして「玄関」と名付けます。もう一方を「寝室兼作業部屋」と名付けます。名前はMacのDBに保存されます。
5. 「活動を記録」で今の行動を入力できます。日付を選ぶとその日のまとめとタイムライン、各行を押すと保存されたログの詳細が開きます。

iPhoneへの着信サーバーは不要です。「疑似センサー受信」はOFFのままで構いません。iPhoneアプリが閉じていても、Macが起動していればSwitchBotのログは蓄積します。アプリは前面で15秒ごとに取得し、通信断のときは保存済みデータであることを表示します。

## 2台の固定登録（任意）

初期状態は対応する開閉センサーを自動検出します。機器IDを確認後、対象を2台に限定する場合は次のように設定します。

```sh
cp server-config.example.json server-config.json
```

`devices` を実際に表示されたIDで設定し、サーバーを再起動してください。

```json
"devices": {
  "玄関センサーのBLE-UUID": {"name": "玄関", "participant_id": "p001"},
  "部屋センサーのBLE-UUID": {"name": "寝室兼作業部屋", "participant_id": "p001"}
}
```

macOSではMACアドレスではなくCoreBluetoothのUUIDです。場所を特定せずにIDを自動で割り当てることはしません。設定済みでも未受信の機器は「不明」と表示します。対応は既存コードと同じ非暗号化のSwitchBot開閉センサーです。他機種や暗号化された広告は読み取りません。

## 表示・集計の意味

- 機器は最後の受信状態、電池残量、受信時刻を表示します。120秒受信しないと現在は「不明」です。初回観測は開閉回数に数えません。繰り返し届く同一状態はタイムラインに重複追加しません。
- 日付境界はAsia/Tokyoです。入力セッションは終了日に集計します。タイムラインは100件ずつ、同一スナップショットに固定して追加取得します。
- 現在の活動は、30分以内の本人の活動回答または2分以内の実際の打鍵のうち、新しい根拠を使います。回答には「本人の回答」、打鍵には「推定」と表示します。根拠が古ければ「不明」です。開閉だけで在宅・外出・睡眠を断定しません。
- キーボードは60秒ごと、無入力60秒、非表示のいずれかでセッションを確定し送信します。iOSによる拡張の強制終了やフルアクセス無効中は、未確定の入力がサーバーに届かないことがあります。
- 終了済み入力セッション・活動回答・文脈イベントはiPhoneの永続キューに残し、再送します。サーバーは参加者・イベント種別・IDで重複保存を防ぎます。以前の送信キューも移行します。旧版で送信キューに積まれなかったローカルログの全履歴を自動取り込みする機能はありません。
- 歩数等はキーボード送信時点の端末Contextです。値の取得時刻を表示します。iPhone上のすべてのアプリの利用状況は取得しません。

## 接続と保存範囲

サーバーはTailscale IPにのみbindし、実際の接続元IPが `allowed_clients` のものだけ受け付けます。リバースプロキシの転送ヘッダーは信用しません。現在はMac自身と指定iPhoneのみです。端末のTailscale IPが変わった場合は設定を更新してください。APIはこの1人用構成を想定し、複数利用者間の認証・権限分離を実装した公開サービスではありません。

**入力イベントには、既存アプリが記録する確定本文・入力欄の前後文脈も含まれます。** 活動回答・端末Context・センサー状態とともにMacへ保存します。クラウドのAIやSwitchBotクラウドには送信しません。iPhoneのBackend URLを空にすると新しい通信を停止できます（端末内の記録と既存の未送信キューは残ります）。

DBとログはgit対象外です。バックアップはサーバーを停止して `data/` をコピーしてください。削除する場合は停止後に必要なバックアップを取ってから行ってください。自動削除・自動保持期限はありません。

## 検証

```sh
.venv/bin/python -m pytest -q
curl http://100.106.185.3:8000/health
```

テストは一時DBとBLE無効のAPIを使います。重複再送、参加者別取得、日本時間の日付境界、ページ分割中の追加、状態鮮度、機器名永続化、活動推定の期限、不正入力、接続元制限、DB再起動、BLEパケットを確認します。BLEを使わない開発起動は `./run_server.sh --no-ble --host 127.0.0.1 --port 8001` です。

既存の診断CLIも使えます：`.venv/bin/python switchbot_contact_ble.py --seconds 30`。継続サーバーと同時にスキャンするより、サーバー停止中に実行してください。

## API

| API | 用途 |
| --- | --- |
| `GET /health` | API・BLEスキャナーの状態 |
| `POST /api/sessions` | 既存iOS入力セッション |
| `POST /api/narrative-events` | 回答1件 / 旧再送形式の配列 |
| `POST /api/context-events` | 文脈イベント |
| `POST /api/sensor-events` | 旧疑似センサーとの互換（実機BLEとは区別） |
| `POST /api/activities` | 単独クライアント向け活動記録 |
| `GET /api/questions/pending/{participant}` | 204。既存キーボードのローカル質問にフォールバック |
| `GET /api/dashboard/{participant}?day=YYYY-MM-DD` | 状態、集計、タイムライン |
| `GET /api/records/{participant}/{kind}/{id}` | 詳細ログ |
| `PUT /api/devices/{id}/name` | `participant_id` と `name` で設置場所を保存 |

開閉データの形式は [SwitchBot公式BLE仕様](https://github.com/OpenWonderLabs/SwitchBotAPI-BLE/blob/latest/devicetypes/contactsensor.md)、macOSの受信は [Bleak公式ドキュメント](https://bleak.readthedocs.io/en/latest/backends/macos.html) を参照しています。

## 三階層LSTM研究機能（追加）

Micro最大1時間・Meso1日・Macro固定週のモデル、現在/当日全体の5項目、個人基準比較、学習/比較評価、全種類エクスポート、Mac→Ubuntu差分同期、質問・可視化・訂正・休憩提案を追加した。Macでは従来の収集を継続し、Ubuntuで `research_enabled` を有効にする。**既定では自動学習しない。**

設定、学習コマンド、検証済み事項と未実装/未検証事項は [docs/research.md](docs/research.md) を参照。
