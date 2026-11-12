# 監査後の修正（2026-11-11）

採用判断：2=B（受信済み同日回答によるオンライン個人化）、10=A（推論復元＋再学習、厳密resumeなし）、GPS削除、他は推奨設定。

| 監査項目 | 修正 |
| --- | --- |
| S01/S03/S05 | サーバー受信時刻、予測時点cutoff、オンライン重み更新protocol |
| S02 | 実IDとの一致拒否、予約sample ID、manifestなし非空投入先の拒否 |
| S04/S11 | 4質問の配分、context軸交互、0〜4時night |
| S06 | runtime/dev/ML分離、numpy不足とDarwin marker修正。Linux CPUのclean venvで導入・pip check済み。専用固定版を追加 |
| S07/S08 | 文章モデル内容hashと移動、コード/設定/環境/データの実験ID、欠落成果物の再試行、force |
| S09/S10 | seed/split/依存/Git metadata、NaN/Inf停止、有効validation 0件の昇格禁止、候補集合の評価整合 |
| S12/S13 | JSON I/Oのtorch依存除去、特徴の遅延生成と訓練正規化の逐次集計 |
| X01 | 全入力モードのcommit契約、英語候補replacesの重複除外 |
| GPS/構成 | GPS地図/行き先処理を削除。Store/BLE/inferenceを小さく分離。従来CLI/importは維持 |

削除判断が不十分な外部公開APIやモデルheadは一律削除していません。旧サンプルCLIは処理を停止して新generatorを案内する互換入口とし、危険な旧DB直接書込をなくしました。チェックポイントのparameter keyを変える最適化は実測後の課題です。

2026-11-12追検証：承認後に専用venvへ収集・ML・テスト依存を導入。CPUで88テスト（軽量15件を含む）が成功しました。合成データによる短い学習、held-out評価、オンライン個人化、Checkpoint保存・復元、APIを含みます。追加したローカル文章モデルの保存・再読み込み・移動・キャッシュ失効も成功しました（外部重み取得なし）。Python AST、JSON/YAML、シェル構文、両repoの共通fixture照合、diff確認も実施。PRごとのCPU CIを追加しました。

未実施：実データ学習、実運用Checkpoint・事前学習済み文章モデルの互換性、GPU/MPS、Mac/iOS実接続。旧macOS固定依存の再解決は実施していません。

切替手順：既存DB/モデル/設定をバックアップ → 両repoの変更をレビュー → 承認した環境で全テスト・実機確認 → 新しいモデル保存先でv2学習・評価 → 推論結果を確認して切替。旧v1ファイルやDBは自動削除しません。収集先IPやBundle IDの既存設定は引き継ぎます。
