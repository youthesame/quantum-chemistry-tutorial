# 量子化学計算 入門

量子化学計算の入門教材です。各章は、ブラウザで仕組みを見る HTML と、Google Colab で自分で計算するノートブックの組でできています。

教材のページ：https://youthesame.github.io/quantum-chemistry-tutorial/

## 章

| 章 | 内容 | ノートブック |
| --- | --- | --- |
| 第1章 量子化学計算 | シュレディンガー方程式、Born–Oppenheimer近似、軌道近似と SCF | [![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youthesame/quantum-chemistry-tutorial/blob/main/dist/ch1_practice.ipynb) |
| 第2章 基底関数 | LCAO、ガウス関数、def2-SVP の SV と P | [![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youthesame/quantum-chemistry-tutorial/blob/main/dist/ch2_practice.ipynb) |
| 第3章 汎関数 | 電子密度、ヤコブのはしご、分散力 | [![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youthesame/quantum-chemistry-tutorial/blob/main/dist/ch3_practice.ipynb) |
| 第4章 構造最適化と振動 | エネルギーの地形、振動数、虚数の振動数 | [![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youthesame/quantum-chemistry-tutorial/blob/main/dist/ch4_practice.ipynb) |
| 第5章 分子軌道と光の吸収 | HOMO と LUMO、共役の長さと吸収波長 | [![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youthesame/quantum-chemistry-tutorial/blob/main/dist/ch5_practice.ipynb) |

計算には [PySCF](https://pyscf.org) を使います。ノートブックの最初のセルが Colab に必要なパッケージを入れるので、手元の PC に何かを入れる必要はありません。

## ディレクトリ

- `docs/`：公開する HTML と、図に使う計算データ
- `dist/`：学生に配るノートブック。穴埋めの部分が `___` になっている
- `src/`：ノートブックの元になる Python ファイル。穴埋めの答えを `【】` で囲んで書く
- `tools/`：`docs/` の計算データを作るスクリプト

## ライセンス

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja), [MIT](https://opensource.org/license/mit)
