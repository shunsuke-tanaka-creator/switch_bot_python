# お天気分・質問・生活パターン

現在の入力契約は `research/events.py` の `research-v2` です。GPS分類、GPSの場所遷移・行き先予測は削除しました。拠点・タスクは[行動Context](behavior-context.md)で独立に扱います。

| kind | 主なpayload | 用途 |
| --- | --- | --- |
| `weather_mood` | weather、任意のquestion_id | happy/normal/sad/angry。5尺度の正解へ換算しない |
| `typing_reason` | question_id、reason | 誤入力・推敲・変換・不明を区別 |
| `context_label` | axis、class_id、start、end、question_id | GPSなしの拠点/タスク本人確認 |
| `context_name`, `context_merge` | 契約に沿うID/軸 | 命名・統合 |
| `intervention_outcome` | question_id、action、executed、helpfulness | 実行と役立ちを区別 |
| `question_state` | question_id、state | shown/scheduled/deferred/declined/stop/resume |
| `preferences` | questions_enabled、exercise_proposals_enabled、daily_limit、gap_minutes等 | 共通設定 |

`POST /api/research/{participant}/events` は参加者・イベントIDで冪等。`received_at` はサーバーが付与します。旧 `place_visit` には互換ACKを返すだけで新規保存せず、既存DB内のGPS行も学習・表示へ使いません。

既定は合計4件/日・60分間隔。current 1、night 1、weather 1、place/taskを日替わり1とし、夜枠を予約します。予定中・観測不足・8時前には出ない場合があります。typingReason/breakCheck/supportOutcomeの自動提案を使う場合はdaily_limitを明示的に5以上にします。その場合はイベント起点の提案が優先され、4定期枠の全件発行は保証しません。表示履歴と有効な予約で複数画面を調整します。

<!-- 追加: 運動提案の設定とデータゲート。 -->
運動提案は `exercise_proposals_enabled` の既定が無効で、歩数データ提供とは別に有効化する。既存のイベント提案と同様に `daily_limit` を5以上へ明示設定した場合にだけ発行する。同じ曜日・同じ拠点・同時刻までの累積歩数を優先し、拠点不明時だけ曜日・時刻へフォールバックする。当日中の歩数増加、鮮度15分以内、3日以上の基準（5日未満は暫定）、中央値との差500歩以上かつ25%以上、25パーセンタイル未満を満たす場合だけ、同日同文脈で1回まで提案する。

サーバーURL設定済みのiOSでは200の質問のみ表示し、204/通信失敗ではローカル質問へfallbackしません。URL未設定時のみローカルTriggerを利用します。

`/research/{participant}/insights` は天気・本人の条件付き基準・探索的曜日傾向・入力修正・生活パターンを表示します。生活パターンは入力操作と周期時刻から作る増殖型プロトタイプで、原著GWR/Graph Transformerの実装ではありません。旧クライアントのため `places` 応答枠は空で残します。

[README](../README.md)の検証手順を使用します。テストの通過は実データの有効性や医学的な診断性能を示しません。
