# 依存関係と検証範囲

| ファイル | 用途 | 状態 |
| --- | --- | --- |
| requirements.txt | 収集/API、numpyによる特徴・研究画面 | 既存の範囲制約を維持 |
| requirements-ml.txt | 上記＋PyTorch/SentenceTransformer | GPU/OSに合わせた導入が必要 |
| requirements-dev.txt | 収集依存＋pytest | 全モデルテストにはML依存も必要 |
| requirements-test-linux-cpu.lock.txt | Linux x86_64 / Python 3.12の収集・ML・テスト環境 | 2026-11-12にclean venvで導入し検証した57パッケージを固定 |
| requirements.lock.txt | 旧収集環境の固定版＋不足numpy、Darwinのpyobjc marker | 全OSを解決した完全lockではない |
| requirements-dev.lock.txt | 旧環境の固定pytestと関連依存 | 新しい検証環境では使用しない。既存利用の互換性のため保持 |

検証環境はLinux x86_64、Python 3.12.14、PyTorch 2.15.0+cpu、numpy 2.5.4、pytest 9.2.0、SentenceTransformers 5.7.0です。同じ依存でGitHub ActionsのUbuntu 24.04 / Python 3.12.16も88テスト成功しました。既存の本番環境は更新していません。`pip check` は依存矛盾なし。CPUの固定版をmacOSやCUDAへ流用せず、それぞれ独立したvenvで検証してください。CUDA版PyTorchは[公式](https://pytorch.org/get-started/locally/)で環境に合うものを選びます。

## Linux CPUで検証環境を再現する

新しいPython 3.12のvenv内で実行します。CPU版torchを公式indexから先に導入することで、後続のpipがCUDA版を選ぶことを防ぎます。

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
python -m pip install -r requirements-test-linux-cpu.lock.txt -r requirements-dev.txt -r requirements-ml.txt
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

期待：インストール完了、`No broken requirements found.`、全テスト `passed`。
意図しない出力：解決不能、対応wheelなし、`FAILED` / `ERROR`。本番環境へ導入せず、依存と実行環境を確認します。

`.github/workflows/validate.yml` も同じ固定版でPRごとにCPUテストを実行します。モデルの外部ダウンロードを無効化し、学習・推論は合成データと一時ディレクトリを使用します。SentenceTransformerのロード経路は小さなローカルBoWモデルで検証し、実運用の事前学習済み文章モデルの精度・互換性とは区別します。

自動学習中のインストールや動的ダウンロードは行いません。実験の `run.json` に実際のPython・依存版・Git SHAを保存します。
