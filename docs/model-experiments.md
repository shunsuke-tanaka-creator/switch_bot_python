# 評価・SSL・部分プーリング（明示的に有効化するオフライン実験）

`python -m research.cli experiment` は、本番の `train` / `compare` / `personalize` と分離した実験入口です。
`ExperimentConfig` の機能フラグはすべて **False**。HTTP、自動学習、推論の設定には接続していません。
明示的な有効化がない実行は、DBを開いたり出力ディレクトリを作ったりする前にエラーになります。
既存モデルのパラメータ名・チェックポイント互換性・学習損失は維持します。

## リポジトリのサンプルだけで全経路を確認

既存のML/dev環境（[依存関係](dependencies.md)）で実行します。
`--sample-fixture` は `research/experiment_sample.py` の決定的な架空2参加者データを使い、DBを開きません。
入力原文、実参加者データ、外部モデルのダウンロードは不要です。

```sh
python -m research.cli experiment --sample-fixture \
  --enable-evaluation --enable-ridge --enable-ssl \
  --enable-partial-pooling --enable-prequential --upper-finetune \
  --epochs 1 --ssl-epochs 1 --finetune-epochs 1 --hidden 8 \
  --ssl-max-samples 4 --models /tmp/lifelog-sample-experiment
```

期待：`status: trained`、`production_promoted: false`、有限の `test_MAE_macro`。
`experiment.json` と `experiment.pt` が生成されます。
意図しない出力：`Traceback`、`Non-finite`、`No observed SSL future targets`、`disabled`。
フラグ・観測の有無・カットオフを確認してください。サンプルのMAEは動作確認であり、実データでの精度改善の証拠ではありません。

## フラグと保存先

| フラグ | 有効になる処理 | 既定 |
| --- | --- | --- |
| `--enable-evaluation` | 個人平均・直前回答・時間帯平均の比較 | 無効 |
| `--enable-ridge` | 学習データのみで標準化・推定するRidge（evaluationも必要） | 無効 |
| `--enable-ssl` | MFM＋未来特徴予測→Encoder凍結でHead学習 | 無効 |
| `--upper-finetune` | SSL後にMacro・上位融合とHeadを低学習率で微調整 | 無効 |
| `--enable-partial-pooling` | 参加者別Timeline＋共有Encoder＋5次元バイアス | 無効 |
| `--enable-prequential` | Frozen / Personal Bias / Full Fine-tuningの比較（partial-poolingも必要） | 無効 |

複数参加者は `--enable-partial-pooling` が必要です。共有モデルの各参加者の特徴・回答履歴・作業区間は別々に構築し、他人の系列を自分の系列に結合しません。評価だけを有効にする場合も、この新しい実験経路のモデルを学習します。既存の `personalize` の2条件比較はそのままです。

出力先の既定値は `data/experiments`。この経路は `active.pt` / `candidate.pt` / `status.json` を書き換えず、モデルを昇格させません。
実験チェックポイントは本番の `version` を持たず、誤配置されても既存推論の互換性検査で拒否されます。
再読込は `research.experiments.load_experiment(path)` を使い、`model_batch` に参加者IDを渡します。
同じ出力先での同時実行はロックで拒否します。同じ場所への再実行は前の実験ファイルを置き換えるので、比較ごとに別ディレクトリを指定します。

実データを使う別途の実験では `--participants ID1 ID2` で対象を明示し、既存DBを `--db`（サブコマンドより前）で指定します。
DBはread-onlyで開いてメモリへコピーし、そのスナップショットから読みます。自動列挙・収集・送信・原文の出力はしません。
`experiment.json` には参加者ID、ラベルID、時刻、予測値と集計結果が含まれるため、実データで実行した成果物はgitへコミットしないでください。

## 1. 評価とカットオフ

`research/experiment_data.py` が全参加者共通のJST日付順で60% / 20% / 20%に分割します。
参加者ごとの異なる分割境界は使いません。共有Encoderに別参加者の評価期間のデータが流入するのを防ぎます。
明示したい場合は、タイムゾーン付きISO時刻を指定します。

```sh
python -m research.cli experiment --sample-fixture \
  --enable-evaluation --enable-ridge --enable-partial-pooling \
  --train-before 2026-09-13T00:00:00+09:00 \
  --test-from 2026-09-15T00:00:00+09:00 \
  --epochs 1 --hidden 8 --models /tmp/lifelog-fixed-cutoffs
```

期待：上記の境界が `experiment.json` の `boundaries` に保存され、`status: trained`。
意図しない出力：`timezone required`、`Require pretrain_before <= train_before < test_from`、`each require state labels`。
日付のみの値や空の分割を避けてください。

| 用途 | 使える時刻 |
| --- | --- |
| 教師あり学習 | 質問時刻 `at < train_before` かつ回答受信時刻 `available < train_before` |
| 検証・モデル選択 | `train_before <= at < test_from` かつ `available < test_from` |
| テスト | `at >= test_from`。モデル・ハイパーパラメータ選択に使用しない |
| SSLの入力・復元対象・未来教師 | すべて `pretrain_before` より前。未来教師の区間終了時刻も **厳密に前** |
| SSLカットオフ | 既定は `train_before`。`--pretrain-before` はそれ以前だけ許可 |
| 標準化 | SSLなし：学習サンプルのみ。SSLあり：SSL入力サンプルのみ（未来教師は含めない） |
| 特徴の作成 | 各予測時刻までに観測・受信されたセンサー情報のみ。古い情報の後日受信も制限 |
| 個人の回答統計による条件付け | 予測時刻より前に質問され、かつ受信済みの本人回答のみ |

検証・テストでも、受信済みの過去回答は個人条件付けに使います。これは「重みは固定だが過去回答で条件付けする」評価です。
全評価期間の回答を一切見ないオープンループ予測とは異なります。
同じ回答の正解は自身の予測に使いません。後日受信した過去回答は、受信後の予測からのみ利用できます。
`run` にGit SHA、依存関係、seed、分割ごとのID/時刻/受信時刻を保存します。

`research/experiment_metrics.py` の主指標は次の順序で求める `MAE_macro` です。

1. 各参加者・各尺度で、観測済み正解についてMAEを計算する。
2. 各参加者について、評価可能な尺度のMAEを等重みで平均する。
3. 評価可能な参加者を等重みで平均する。

欠測尺度は分母から除き、0誤差にはしません。5尺度ごとの参加者平均・回答件数・参加者数、参加者別結果、current/daily別結果も出力します。
総合値ではcurrent/dailyの回答を尺度内で集約します。モデル選択は検証の `MAE_macro` のみです。

| ベースライン | 推定方法・フォールバック |
| --- | --- |
| `personal_mean` | 学習期間の本人・同じscopeの平均。未観測尺度は同scopeの学習集団平均→3 |
| `last_answer` | 予測より前に受信した本人・同じscopeの直近の有限値（尺度別）。未回答は個人平均へ |
| `time_of_day_mean` | 学習期間の本人・同じscope・同じJST時台（0〜23）の平均。未観測は個人平均へ |
| `ridge` | 直近の因果的特徴bin＋個人条件付け40次元。scope/尺度別Ridge、切片は正則化しない |

平均・時間帯平均・Ridgeは学習期間に固定します。直前回答だけは、評価中に実際に受信した過去回答も使います。
Ridgeの平均・標準偏差・係数は学習側だけで推定します。`--ridge-alpha` の既定値は1です。

## 2. 自己教師あり事前学習

`research/ssl.py` は質問票・訂正由来の活動境界を除いたセンサーTimelineを使い、ラベルのない時刻からも学習します。
既定の入力間隔は60分、最大256入力/参加者で期間全体から等間隔に選択します。
`--ssl-stride-minutes`（5分以上）、`--ssl-max-samples`、`--ssl-epochs` で調整できます。

- **MFM**：元のpresenceチャネルが観測済みの特徴だけをランダムに隠して復元。欠測・padding・presenceビット・決定的な時刻特徴は損失対象外です。テキストを隠す際はベクトル全体と対応する観測情報を隠します。Micro＋Meso＋Macroの表現から週内binを、Macroから過去28日の特徴を復元します。
- **Future Feature Prediction**：入力時刻tから、各hの `(t+h−5分, t+h]` の観測済み特徴を予測します。`--horizons 5 15 30` が既定。5〜30分の5分刻みを指定できます。未来の特徴を入力には加えません。
- SSLの損失は観測対象の二乗誤差。正規化はSSL入力だけで推定し、教師の未来時刻・受信時刻もカットオフを越えません。対応する観測がなければ成功扱いせずエラーにします。
- SSL後は復元・予測用Headを捨て、current/daily/activity Headと有効時の個人バイアスのみを教師あり学習します。Encoder、融合層、個人条件付け投影は凍結します。
- 任意の上層微調整ではMacro LSTM、history projection、top_day/top_workとHeadのみ更新。Micro/Mesoと個人投影は凍結を維持します。既定学習率は0.001→0.0001、低い値を必須とします。

`ssl` に損失、入力数、更新数、最大入力時刻・未来教師終了時刻を、`stages` に段階・epoch・学習率・検証スコアを記録します。
実験経路の教師あり5尺度損失はL2バイアスとの対応を明確にするためMSE、活動は交差エントロピーです。
既存本番経路のHuber損失は変えません。SSLを使わない場合は共有EncoderとHeadを最初から教師あり学習します。

## 3. 個人差の部分プーリングとprequential

`research/experimental_model.py` は共通Encoder/Headの5尺度予測に `b_p ∈ R^5` を加えます。
current/dailyで同じ5次元補正を共有し、活動分類には加えません。初期補正は0、未知の参加者にも0を使います。
バイアスは加算した値で評価し、1〜5へのクリップで誤差を隠しません。

尺度kの本人回答数をnとすると、観測誤差に `λ b[p,k]^2 / max(n,1)` を加えます。
同じ尺度のn回答を通じて合計すると `Σ(error²) + λ b²` というL2正則化になります。
共有予測を固定した場合の最適補正は `n/(n+λ) × 本人の平均残差` なので、回答が少ないほど0に縮小されます。
`--bias-l2` は正の値を必須とし、既定は10。学習時のnはカットオフ前の回答のみ、オンライン時は受信して更新可能になった回答だけで増やします。
欠測を含む教師ありバッチでは各回答の観測尺度数で損失を平均するため、上式は同じ観測尺度構成の場合の厳密な解です。

`research/experiments.py` の3条件は、検証で選択した**同一の初期モデルを独立にコピー**して比較します。

| 条件 | 更新するもの |
| --- | --- |
| `frozen` | 重みを変更しない。本人の受信済み過去回答による入力条件付けは更新 |
| `personal_bias` | 当該参加者の5次元バイアスだけ（Encoder/Headは固定） |
| `full_finetuning` | Encoder・Head・個人条件付け投影・バイアスの全学習パラメータ |

予測時刻tの前に、`label.at < t AND label.available < t` の未使用回答を受信順に1回ずつ更新し、その後で予測を保存します。
同時刻の回答は同時刻の予測に使わず、参加者とラベルIDとscopeの組で重複更新を防ぎます。
学習期間の質問でも、遅れて受信して学習に使えなかった回答は受信後に利用できます。
検証期間の回答も、テスト時点で利用可能ならオンライン更新に使います。
両更新条件は同じ低学習率・同じSGD・同じ回答順で比較します。Biasのみの更新では他参加者の行を動かしません。

`prequential.metrics` に各条件のMAE_macro/尺度別結果、`prequential.predictions` に予測時刻・更新数・最新受信時刻と3条件の予測を保存します。
テスト結果でモデルの昇格やハイパーパラメータ選択はしません。

## 検証

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

期待：既存テストと `tests/test_experiments.py` がすべて `passed`。
意図しない出力：`FAILED` / `ERROR` / 依存不足による `skipped`。その状態ではmainへマージしません。
テストは架空のサンプル、メモリDB、一時ディレクトリのみを使用します。
