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
# # 第5章 実習　共役の長さと吸収する光
#
# 上から順に `Shift + Enter` で実行します。
#
# コードの中の `___` は穴埋めです。埋めてから実行してください。分からないときは、すぐ下の「答え」を開きます。

# %% cellView="form"
# @title 準備　実行するだけでよい。Colabでは最初の1回に30秒ほどかかる
import sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyscf", "py3Dmol"], check=True)

import numpy as np
import py3Dmol
from pyscf import gto, dft, tdscf
from pyscf.tools import cubegen

HARTREE_EV = 27.2114


def polyene(n):
    """二重結合が n 個つながった、まっすぐな全トランスのポリエンを作る。"""
    atoms, cc, ch = [], 1.40, 1.09
    for i in range(2 * n):
        x, y = i * cc * np.cos(np.radians(30)), (i % 2) * cc * np.sin(np.radians(30))
        up = 1 if i % 2 else -1
        atoms += [("C", (x, y, 0.0)), ("H", (x, y + up * ch, 0.0))]
        if i in (0, 2 * n - 1):  # 端の炭素には水素をもう1つ付ける
            side = -1 if i == 0 else 1
            atoms.append(("H", (x + side * ch * np.cos(np.radians(30)), y - up * ch * np.sin(np.radians(30)), 0.0)))
    return gto.M(atom=atoms, basis="def2-svp", verbose=0)


def show_orbital(mol, mo, iso=0.04):
    """分子軌道を3Dで表示する。橙は波動関数が正、青は負の領域。"""
    cubegen.orbital(mol, "orbital.cube", mo, nx=60, ny=60, nz=60)
    text = open("orbital.cube").read()
    view = py3Dmol.view(width=480, height=300)
    view.addModel(text, "cube")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}})
    view.addVolumetricData(text, "cube", {"isoval": iso, "color": "#eb6834", "opacity": 0.85})
    view.addVolumetricData(text, "cube", {"isoval": -iso, "color": "#2a78d6", "opacity": 0.85})
    view.zoomTo()
    view.rotate(-60, "x")
    view.show()


NAMES = {2: "ブタジエン", 3: "ヘキサトリエン", 4: "オクタテトラエン"}
print("準備完了")

# %% [markdown]
# ## 実習1　HOMO と LUMO の差を比べる
#
# 二重結合が2個、3個、4個のポリエンを計算します。
#
# 穴埋め：HOMO と LUMO のエネルギーの差を求めます。LUMO の方が上にあります。

# %%
results = {}
for n in [2, 3, 4]:
    mol = polyene(n)
    mf = dft.RKS(mol, xc="b3lyp").run()
    i = mol.nelectron // 2 - 1  # HOMO の番号
    homo, lumo = mf.mo_energy[i] * HARTREE_EV, mf.mo_energy[i + 1] * HARTREE_EV
    gap = 【lumo - homo】
    results[n] = (mol, mf)
    print(f"{NAMES[n]:8s}: HOMO {homo:6.2f} eV　LUMO {lumo:6.2f} eV　差 {gap:.2f} eV")

# %% [markdown]
# <details><summary>答え</summary>
#
# `lumo - homo` です。二重結合が増えるほど、差が小さくなります。
#
# </details>

# %% [markdown]
# ## 実習2　吸収する光の波長を計算する
#
# TDDFT で励起エネルギーを求めます。4つの励起のうち、光を最もよく吸収するものを選びます。
#
# 穴埋め：エネルギー E [eV] の光の波長は、1239.84 / E [nm] です。

# %%
EXP = {2: 217, 3: 258, 4: 290}
for n, (mol, mf) in results.items():
    td = tdscf.TDA(mf)
    td.nstates = 4
    td.kernel()
    k = np.argmax(td.oscillator_strength())
    ev = td.e[k] * HARTREE_EV
    nm = 1239.84 / 【ev】
    print(f"{NAMES[n]:8s}: {ev:.2f} eV　→ {nm:.0f} nm　実験 {EXP[n]} nm")

# %% [markdown]
# <details><summary>答え</summary>
#
# `1239.84 / ev` です。計算値は実験より短い波長に出ますが、鎖が長くなるほど長い波長へ移る傾向は再現できています。
#
# </details>

# %% [markdown]
# ## 実習3　HOMO と LUMO の形を見る
#
# ヘキサトリエンの HOMO と LUMO を表示します。節の数を数えてみましょう。

# %%
mol, mf = results[3]
i = mol.nelectron // 2 - 1
show_orbital(mol, mf.mo_coeff[:, i])
show_orbital(mol, mf.mo_coeff[:, i + 1])

# %% [markdown]
# 上が HOMO、下が LUMO です。LUMO の方が、炭素の間で符号が入れ替わる節が1つ多くあります。節が多い軌道ほどエネルギーが高くなります。
#
# ## 確認問題
#
# 1. 分子が光を吸収するとき、電子はどの軌道からどの軌道へ移りますか。
# 2. 共役が長くなると、吸収する光の波長はどう変わりますか。
# 3. β-カロテンが橙色に見えるのはなぜですか。
