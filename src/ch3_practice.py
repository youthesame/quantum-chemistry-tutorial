# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 第3章 実習　汎関数を変えてみる
#
# 上から順に `Shift + Enter` で実行します。
#
# コードの中の `___` は穴埋めです。埋めてから実行してください。分からないときは、すぐ下の「答え」を開きます。

# %% cellView="form"
# @title 準備　実行するだけでよい。Colabでは最初の1回に30秒ほどかかる
import sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyscf", "py3Dmol"], check=True)

import py3Dmol
from pyscf import gto, scf, dft
from pyscf.tools import cubegen

HARTREE_KCAL = 627.51
WATER = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"


def show_density(mol, dm, iso=0.03):
    """電子密度を3Dで表示する。"""
    cubegen.density(mol, "density.cube", dm, nx=60, ny=60, nz=60)
    text = open("density.cube").read()
    view = py3Dmol.view(width=480, height=360)
    view.addModel(text, "cube")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}})
    view.addVolumetricData(text, "cube", {"isoval": iso, "color": "#2a78d6", "opacity": 0.5})
    view.zoomTo()
    view.show()


print("準備完了")

# %% [markdown]
# ## 実習1　電子密度を見る
#
# DFT で水を計算し、電子密度を表示します。`iso` の値を 0.3 や 0.003 に変えて、雲の大きさがどう変わるかを見てみましょう。

# %%
mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
mf = dft.RKS(mol, xc="b3lyp")
mf.kernel()
show_density(mol, mf.make_rdm1(), iso=0.03)

# %% [markdown]
# ## 実習2　汎関数ごとに、結合の強さを比べる
#
# 水を O と H 2個にばらばらにするのに必要なエネルギー、原子化エネルギーを計算します。酸素原子は不対電子が2個、水素原子は1個なので、`spin` で指定します。
#
# 穴埋め：原子化エネルギーは、ばらばらの原子のエネルギーの合計から、水のエネルギーを引いた値です。

# %%
def energy(atom, spin, xc):
    m = gto.M(atom=atom, basis="def2-tzvp", spin=spin, verbose=0)
    if xc == "hf":
        mf = scf.UHF(m) if spin else scf.RHF(m)
    else:
        mf = dft.UKS(m, xc=xc) if spin else dft.RKS(m, xc=xc)
    return mf.kernel()


for xc in ["hf", "svwn", "pbe", "b3lyp"]:
    e_O = energy("O 0 0 0", 2, xc)
    e_H = energy("H 0 0 0", 1, xc)
    e_water = energy(WATER, 0, xc)
    ae = (【e_O + 2 * e_H - e_water】) * HARTREE_KCAL
    print(f"{xc:6s}: {ae:6.1f} kcal/mol")
print("実験値: 233 kcal/mol")

# %% [markdown]
# <details><summary>答え</summary>
#
# `e_O + 2 * e_H - e_water` です。水素原子は2個あるので、`e_H` を2倍します。`svwn` は LDA の汎関数の名前です。
#
# </details>

# %% [markdown]
# ## 実習3　分散力を含む汎関数と比べる
#
# メタン2分子を炭素どうし 3.8 Å の距離に置き、離れた2分子よりエネルギーが低いかどうかを調べます。負の値なら引き合っています。ωB97X-V の計算には1分ほどかかります。

# %%
A = "C 0 0 0; H 0 0 -1.09; H 1.028 0 0.363; H -0.514 0.890 0.363; H -0.514 -0.890 0.363"
B = "C 0 0 3.8; H 0 0 4.89; H 1.028 0 3.437; H -0.514 0.890 3.437; H -0.514 -0.890 3.437"


def methane_energy(atom, xc):
    mf = dft.RKS(gto.M(atom=atom, basis="def2-tzvp", verbose=0), xc=xc)
    mf.nlcgrids.level = 1  # 分散力の項の計算を速くする
    return mf.kernel()


for xc in ["b3lyp", "wb97x_v"]:
    e = (methane_energy(A + "; " + B, xc) - 2 * methane_energy(A, xc)) * HARTREE_KCAL
    print(f"{xc:8s}: {e:+.2f} kcal/mol")

# %% [markdown]
# B3LYP は正の値で、メタンどうしは引き合いません。分散力を含む ωB97X-V は負の値で、引き合います。
#
# ## 確認問題
#
# 1. DFT で、波動関数の代わりに主に扱う量は何ですか。
# 2. 実習2で、HF の原子化エネルギーが小さくなったのはなぜですか。
# 3. B3LYP で分子どうしの弱い相互作用を調べるとき、何を足しますか。
