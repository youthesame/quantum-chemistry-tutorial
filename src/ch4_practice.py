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
# # 第4章 実習　構造最適化と振動数の計算
#
# 上から順に `Shift + Enter` で実行します。
#
# コードの中の `___` は穴埋めです。埋めてから実行してください。分からないときは、すぐ下の「答え」を開きます。

# %% cellView="form"
# @title 準備　実行するだけでよい。Colabでは最初の1回に30秒ほどかかる
import sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyscf", "geometric", "py3Dmol"], check=True)

import io
from contextlib import redirect_stderr
import numpy as np
import py3Dmol
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

HARTREE_KCAL = 627.51


def show_vibration(mol, mode, amplitude=1.0):
    """分子の振動をアニメーションで表示する。"""
    xyz = mol.atom_coords(unit="Angstrom")
    lines = [f"{mol.atom_symbol(i)} {x:.4f} {y:.4f} {z:.4f} {dx:.4f} {dy:.4f} {dz:.4f}"
             for i, ((x, y, z), (dx, dy, dz)) in enumerate(zip(xyz, mode))]
    view = py3Dmol.view(width=420, height=300)
    view.addModel(f"{mol.natm}\n\n" + "\n".join(lines), "xyz")
    view.setStyle({"stick": {"radius": 0.13}, "sphere": {"scale": 0.28}})
    view.vibrate(16, amplitude, True)
    view.animate({"loop": "backAndForth", "interval": 40})
    view.zoomTo()
    view.show()


def frequencies(mol):
    """振動数を cm⁻¹ で返す。虚数の振動数は負の数で表す。"""
    mf = dft.RKS(mol, xc="b3lyp").run()
    res = thermo.harmonic_analysis(mol, mf.Hessian().kernel())
    freq = [f.real if abs(f.imag) < 1e-6 else -f.imag for f in np.atleast_1d(res["freq_wavenumber"])]
    return np.array(freq), res["norm_mode"]


print("準備完了")

# %% [markdown]
# ## 実習1　ゆがんだ水を最適化する
#
# H–O–H を 150° に開き、O–H を 1.14 Å に伸ばした水から始めます。`callback` に渡した関数は、1歩進むごとに呼ばれます。
#
# 穴埋め：1歩ごとの構造 `m` から、H–O–H の角度を求めます。原子の番号は O が0、H が1と2です。

# %%
def angle(m, i, j, k):
    """原子 i–j–k のなす角度を度で返す。"""
    x = m.atom_coords(unit="Angstrom")
    a, b = x[i] - x[j], x[k] - x[j]
    return np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))


steps = []
def record(env):
    steps.append((env["energy"], angle(env["mol"], 【1, 0, 2】)))


start = gto.M(atom="O 0 0 0; H 0 1.098 0.297; H 0 -1.098 0.297", basis="def2-svp", verbose=0)
with redirect_stderr(io.StringIO()):  # 途中経過のログを隠す
    water = optimize(dft.RKS(start, xc="b3lyp"), callback=record)

for n, (e, a) in enumerate(steps, 1):
    print(f"{n} 歩目: 角度 {a:6.1f}°　エネルギー {(e - steps[-1][0]) * HARTREE_KCAL:+7.2f} kcal/mol")

# %% [markdown]
# <details><summary>答え</summary>
#
# `1, 0, 2` です。真ん中に頂点の原子、つまり酸素の番号 0 を置きます。数歩で 103° 付近に落ち着き、エネルギーの変化も 0 に近づきます。
#
# </details>

# %% [markdown]
# ## 実習2　振動数を計算して、谷底かどうかを確かめる
#
# 最適化した水の振動数を計算します。B3LYP の振動数は実験より少し高く出るので、0.96 を掛けて実験と比べます。
#
# 穴埋め：虚数の振動数は負の数で表しています。負の数がいくつあるかを数えます。

# %%
freq, modes = frequencies(water)
for f, exp in zip(freq, [1595, 3657, 3756]):
    print(f"計算 {f:6.0f} cm⁻¹　× 0.96 = {f * 0.96:6.0f} cm⁻¹　実験 {exp} cm⁻¹")
n_imag = sum(1 for f in freq if 【f < 0】)
print(f"虚数の振動数: {n_imag} 個")

# %% [markdown]
# <details><summary>答え</summary>
#
# `f < 0` です。虚数が0個なので、最適化した構造は谷底です。
#
# </details>

# %% [markdown]
# 一番低い振動数の振動を、アニメーションで見てみましょう。`modes[0]` を `modes[1]` や `modes[2]` に変えると、ほかの振動も見られます。

# %%
show_vibration(water, modes[0])

# %% [markdown]
# ## 実習3　まっすぐな水の振動数
#
# H–O–H を 180° にした水は、谷底ではありません。振動数を計算して確かめます。

# %%
linear = gto.M(atom="O 0 0 0; H 0 0 0.967; H 0 0 -0.967", basis="def2-svp", verbose=0)
freq, modes = frequencies(linear)
print("振動数:", [f"{f:.0f}" for f in freq])
show_vibration(linear, modes[0])

# %% [markdown]
# 負の数、つまり虚数の振動数が出ました。アニメーションの向きに動かすと、エネルギーが下がります。まっすぐな水には折れ曲がる方向が2つあるので、虚数の振動数も2つ出ます。
#
# ## 確認問題
#
# 1. 構造最適化は、エネルギーの地形のどこを探す計算ですか。
# 2. 最適化のあとに振動数を計算するのはなぜですか。
# 3. 虚数の振動数がちょうど1つある構造は、何の候補ですか。
