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
# # 第1章 実習　はじめての量子化学計算
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
HARTREE_EV = 27.2114


def show_orbital(mol, mo, iso=0.05):
    """分子軌道を3Dで表示する。橙は波動関数が正、青は負の領域。"""
    cubegen.orbital(mol, "orbital.cube", mo, nx=60, ny=60, nz=60)
    text = open("orbital.cube").read()
    view = py3Dmol.view(width=480, height=360)
    view.addModel(text, "cube")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}})
    view.addVolumetricData(text, "cube", {"isoval": iso, "color": "#eb6834", "opacity": 0.85})
    view.addVolumetricData(text, "cube", {"isoval": -iso, "color": "#2a78d6", "opacity": 0.85})
    view.zoomTo()
    view.show()


print("準備完了")

# %% [markdown]
# ## 実習1　H₂ の安定な結合距離を探す
#
# 原子核を止めて、結合距離ごとにエネルギーを計算します。HTML教材の場面2の曲線を、自分で描く実習です。
#
# 穴埋め：計算したエネルギー `e` を、リスト `energies` の末尾に追加します。

# %%
distances = [0.5, 0.6, 0.7, 0.74, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]  # 単位はÅ
energies = []
for r in distances:
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="def2-svp", verbose=0)
    e = dft.RKS(mol, xc="b3lyp").kernel()
    【energies.append(e)】

# 一番低いエネルギーからの差を、■ の数で表す。■ 1個が 5 kcal/mol
for r, e in zip(distances, energies):
    diff = (e - min(energies)) * HARTREE_KCAL
    print(f"{r:.2f} Å  {diff:5.1f} kcal/mol  " + "■" * round(diff / 5))

# %% [markdown]
# <details><summary>答え</summary>
#
# `energies.append(e)` です。
#
# </details>

# %% [markdown]
# 穴埋め：エネルギーが一番低い距離を取り出します。`energies.index(x)` は、`x` がリストの何番目にあるかを返します。

# %%
best = distances[energies.index(【min(energies)】)]
print(f"安定な結合距離: {best} Å　実験値: 0.741 Å")

# %% [markdown]
# <details><summary>答え</summary>
#
# `min(energies)` です。一番低いエネルギーの位置を探し、同じ位置の距離を取り出しています。
#
# </details>

# %% [markdown]
# ## 実習2　SCF の繰り返しを見る
#
# 水の軌道を、大ざっぱな初期の形から SCF で直していきます。`callback` に渡した関数は、反復のたびに呼ばれます。
#
# 穴埋め：反復の回数は、記録したエネルギーの個数と同じです。リストの長さを返す関数を使います。

# %%
water = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis="def2-svp", verbose=0)
mf = scf.RHF(water)
mf.init_guess = "hcore"  # 大ざっぱな初期の軌道から始める
history = []
mf.callback = lambda env: history.append(env["e_tot"])
mf.kernel()

for i, e in enumerate(history, 1):
    print(f"{i:2d} 回目: {e:.6f} Hartree")
print(f"{【len(history)】} 回で収束しました")

# %% [markdown]
# <details><summary>答え</summary>
#
# `len(history)` です。最初の数回は大きく変わり、最後は小数点以下6桁まで同じ値になります。
#
# </details>

# %% [markdown]
# ## 実習3　電子が入った軌道を見る
#
# 水の電子10個は、エネルギーの低い軌道から2個ずつ入ります。

# %%
for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
    if occ > 0:
        print(f"軌道 {i}: {e * HARTREE_EV:8.1f} eV　電子 {int(occ)} 個")

# %% [markdown]
# 一番エネルギーが低い軌道 0 を表示します。マウスで回してみましょう。

# %%
show_orbital(water, mf.mo_coeff[:, 0], iso=0.1)

# %% [markdown]
# 軌道 0 は、酸素原子の上に小さな球として現れます。これは酸素の 1s 軌道がほぼそのまま残った内殻の軌道で、結合にはほとんど関わりません。
#
# ## 確認問題
#
# 1. Born–Oppenheimer近似では、何を止めて、何を計算していますか。
# 2. SCF 計算で、同じ手順を繰り返す必要があるのはなぜですか。
# 3. 軌道近似で捨てている、電子同士の動きは何と呼ばれますか。
