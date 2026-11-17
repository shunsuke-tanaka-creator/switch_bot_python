# 文章エンコーダの導入

主条件は `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
revisionは `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` に固定する。
日本語を含む多言語モデルの384次元出力を、seed 17の固定ガウス射影で32次元にする。
LSTMの入力全体は98次元のまま。モデルをevalにし、文章エンコーダの重みは学習しない。
入力は末尾4096文字、その後モデルの最大128トークンに制限される。

## 配置と実行

リポジトリのルートで、ML依存を導入済みの仮想環境から実行する。

```sh
python -m research.text_setup
```

期待する出力：`"status": "ready"`、モデルの絶対パス、revision、内容ハッシュ、入力384／出力32次元。
意図しない出力：ダウンロードエラー・依存エラー・次元エラー。完了していないので学習に進まない。
既存ディレクトリは上書きせずエラーにする。配置先を変更する場合は `--output` を指定する。
公開重みを取得するこの処理だけがネットワークを使い、参加者データやDBは開かない。

```sh
python -m research.cli --participant p001 train
python -m research.cli --participant p001 context-train
python -m research.cli experiment --sample-fixture --enable-ssl --enable-partial-pooling --epochs 1 --ssl-epochs 1
```

期待する出力：通常／文脈学習は十分なラベルがあると `trained`、実験は `trained` と成果物パス。
`insufficient_labels` / `insufficient_observations` はデータ不足であり、学習成功ではない。
意図しない出力：`text_model must be an existing local model directory`、例外、非有限損失。
配置先を変えた場合は全コマンドに同じ `--text-model /absolute/path` を渡す。
サーバーには同じパスを `text_model` として設定し、既存の起動方法で反映する。
設定例3件もこのモデルを指定している。既存の実運用設定は自動書換えしない。

## 学習経路と互換性

| 経路 | 設定の受渡し |
|---|---|
| train / compare / personalize | CLI → Timeline → LocalEncoder |
| context-train / サーバー文脈学習 | text_model → Timeline → LocalEncoder |
| experiment（SSL・部分プーリング） | ExperimentConfig.text_model → cohort → Timeline |
| SSL用センサ系列 | ラベルを除去した新Timelineにも同じtext_modelを保持 |
| 通常／文脈の推論 | 設定または学習記録のローカルモデル。内容ハッシュを照合 |

CLIの既定パスは `data/text-models/paraphrase-multilingual-MiniLM-L12-v2`。
欠落・破損時に文字n-gramへ自動フォールバックしない。
比較用の文字n-gramは `--character-ngram` を明示する。
互換用Python APIの `text_model=None` と既存の未設定サーバーは従来の文字n-gram経路を保つ。
SentenceTransformerの重みを指定しただけで既存LSTM重みを使い回さず、通常・文脈モデルを再学習する。
実験成果物は本番へ自動昇格しない。実データでの精度改善は別途評価が必要。

基本・文脈・実験のレポートは `text_encoder` にbackend、内容ハッシュ、次元、射影seed、
配置時に保存したモデルID／revisionを記録する。移設しても内容が同じならハッシュは同じ。
推論は `local_files_only=True, trust_remote_code=False` で行い、確定文章を外部APIに送らない。
前後contextや自己報告の自由文を文章入力に混ぜない。iOSの既存commit形式は変更しない。

## 状態マップ

中心は観測座標ごとの逐次平均で更新する。近傍学習率による更新は使わない。
保存される辺は自己遷移を除き、時刻差0〜12時間の状態変更回数だけである。
構成比は `Q_ab = N_ab / sum(c != a, N_ac)`、分母0では未定義とする。
A,A,A,BならA→Bの回数は1、構成比は1。滞在を含む次観測の予測確率ではない。
APIの `edges[].count` は引き続き回数であり、構成比フィールドを追加したものではない。

モデル出典：[モデルカード](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)、
[SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。
