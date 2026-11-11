# 依存関係と検証範囲

| ファイル | 用途 | 状態 |
| --- | --- | --- |
| requirements.txt | 収集/API、numpyによる特徴・研究画面 | 既存の範囲制約を維持 |
| requirements-ml.txt | 上記＋PyTorch/SentenceTransformer | GPU/OSに合わせた導入が必要 |
| requirements-dev.txt | 収集依存＋pytest | 全モデルテストにはML依存も必要 |
| requirements.lock.txt | 旧収集環境の固定版＋不足numpy、Darwinのpyobjc marker | 全OSを解決した完全lockではない |
| requirements-dev.lock.txt | 既存固定pytestと関連依存 | 新規環境で未解決・未検証 |

対応基準はPython 3.12、Mac収集/CPU学習、Ubuntu CPU/CUDA、iOS、JSTです。既存版の一括更新はしていません。numpyは今回の軽量検証環境の2.3.5を記録しました。LinuxのBLEにはdbus系の推移的依存もあり、歴史的freezeを万能なlockとは扱いません。

clean環境での依存解決・インストール検証は未実施です。収集とMLを別venvにし、導入を承認した環境で全テストを実行した後、OS/Python/GPU別のfreezeを確定してください。CUDA版PyTorchは[公式](https://pytorch.org/get-started/locally/)で環境に合うものを選びます。

```sh
python -m pip install -r requirements-dev.txt -r requirements-ml.txt
python -m pytest -q
```

期待：インストール完了後、全テスト `passed`。
意図しない出力：解決不能、対応wheelなし、CUDA初期化失敗、`FAILED` / `ERROR`。本番環境へ導入せず、依存と実行環境を確認します。

自動学習中のインストールや動的ダウンロードは行いません。実験の `run.json` に実際のPython・依存版・Git SHAを保存します。
