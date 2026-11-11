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
# # 第2章 実習　基底関数を自分で変えてみる
#
# 上から順に `Shift + Enter` で実行します。
#
# コードの中の `___` は穴埋めです。埋めてから実行してください。分からないときは、すぐ下の「答え」を開きます。

# %% cellView="form"
# @title 準備　実行するだけでよい。Colabでは最初の1回に30秒ほどかかる
import sys, subprocess
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyscf", "geometric", "py3Dmol"], check=True)

import io, time
from contextlib import redirect_stderr
import numpy as np
import py3Dmol
from pyscf import gto, dft
from pyscf.tools import cubegen
from pyscf.geomopt.geometric_solver import optimize


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


def bond_angle(mol, i, j, k):
    """原子 i-j-k のなす角度を度で返す。"""
    x = mol.atom_coords(unit="Angstrom")
    a, b = x[i] - x[j], x[k] - x[j]
    return np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))


print("準備完了")

# %% [markdown]
# ## 実習1　水分子を計算する
#
# 指定するのは、原子の位置、基底関数、汎関数の3つだけです。
#
# 穴埋め：HTML教材で見た基底関数 def2-SVP を指定します。名前は小文字の文字列で書きます。

# %%
mol = gto.M(
    atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",  # 原子の位置。単位はÅ
    basis=【"def2-svp"】,                                        # 基底関数
    verbose=0,
)
mf = dft.RKS(mol, xc="b3lyp")                                # 汎関数はB3LYP
energy = mf.kernel()

print(f"エネルギー: {energy:.6f} Hartree")
print(f"基底関数の数: {mol.nao} 個")

# %% [markdown]
# <details><summary>答え</summary>
#
# `basis="def2-svp"`
#
# </details>

# %% [markdown]
# 24個の基底関数の内訳です。
#
# - `O 2px` と `O 3px`：split valence で分けた、縮んだ関数と広がった関数。3 は2つ目という番号で、3p軌道ではない
# - 酸素の `3d` と水素の `2p`：分極関数

# %%
for label in mol.ao_labels():
    print(label)

# %% [markdown]
# 穴埋め：一覧から、酸素の基底関数の数を数えます。`label` に `"O"` が含まれていたら、`n_O` を1増やします。

# %%
n_O = 0
for label in mol.ao_labels():
    if "O" in label:
        n_O = 【n_O + 1】
print(f"酸素の基底関数: {n_O} 個")
print(f"水素1個あたりの基底関数: {(mol.nao - n_O) // 2} 個")

# %% [markdown]
# <details><summary>答え</summary>
#
# `n_O = n_O + 1` です。`n_O += 1` と書いても同じです。酸素は14個、水素は5個ずつで、合計24個になります。
#
# </details>

# %% [markdown]
# ## 実習2　分子軌道を見る
#
# `mf.mo_coeff` には、HTMLの場面1で見た係数 c が入っています。

# %%
homo = mol.nelectron // 2 - 1
print(f"HOMO のエネルギー: {mf.mo_energy[homo] * 27.2114:.2f} eV")
show_orbital(mol, mf.mo_coeff[:, homo])

# %% [markdown]
# HOMO は酸素の非共有電子対で、分子面に垂直なp軌道の形です。マウスで回してみましょう。
#
# 穴埋め：LUMO は HOMO の1つ上の軌道です。LUMO の番号を `homo` を使って書きます。

# %%
lumo = 【homo + 1】
print(f"LUMO のエネルギー: {mf.mo_energy[lumo] * 27.2114:.2f} eV")
show_orbital(mol, mf.mo_coeff[:, lumo])

# %% [markdown]
# <details><summary>答え</summary>
#
# `lumo = homo + 1` です。LUMO は O–H 結合の反結合性軌道で、節があります。
#
# </details>

# %% [markdown]
# ## 実習3　分極関数を抜く
#
# 実験の結合角は 104.5° です。分極関数を抜くと、角度は大きくなるか小さくなるか。予想してから実行しましょう。

# %%
def without_polarization(basis):
    """各元素から分極関数を取り除いた基底関数を作る。"""
    max_l = {"H": 0, "O": 1}  # 残す関数の角運動量の上限。0がs、1がp、2がd
    return {el: [shell for shell in gto.basis.load(basis, el) if shell[0] <= lmax] for el, lmax in max_l.items()}


def optimized_angle(basis):
    mol = gto.M(atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", basis=basis, verbose=0)
    with redirect_stderr(io.StringIO()):  # 構造最適化の途中経過のログを隠す
        mol_eq = optimize(dft.RKS(mol, xc="b3lyp"))
    return bond_angle(mol_eq, 1, 0, 2)


print(f"分極関数なし (def2-SV): {optimized_angle(without_polarization('def2-svp')):.1f}°")
print(f"def2-SVP              : {optimized_angle('def2-svp'):.1f}°")
print("実験値                : 104.5°")

# %% [markdown]
# ## 実習4　基底関数を増やすと、何が変わって何が変わらないか
#
# HCN と異性体 HNC のエネルギー差を、基底関数を大きくしながら計算します。実験では HNC が約 15 kcal/mol 高いです。

# %%
HARTREE_KCAL = 627.51
molecules = {
    "HCN": "H 0 0 -1.066; C 0 0 0; N 0 0 1.153",
    "HNC": "H 0 0 -0.995; N 0 0 0; C 0 0 1.169",
}

for basis in ["def2-svp", "def2-tzvp", "def2-qzvp"]:
    t0 = time.perf_counter()
    e = {}
    for name, atom in molecules.items():
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        e[name] = dft.RKS(mol, xc="b3lyp").kernel()
    diff = (e["HNC"] - e["HCN"]) * HARTREE_KCAL
    print(f"{basis}: 基底関数 {mol.nao} 個, 時間 {time.perf_counter() - t0:.1f} 秒")
    print(f"    E(HCN) = {e['HCN']:.6f} Hartree,  E(HNC) = {e['HNC']:.6f} Hartree,  差 = {diff:.1f} kcal/mol")

# %% [markdown]
# 個々のエネルギーは 70 kcal/mol 以上動くのに、差はほとんど変わりません。同じ基底関数なら、誤差が両方に同じように入って打ち消し合うからです。
#
# ## 確認問題
#
# 1. def2-SVP の SV と P は、それぞれどんな関数ですか。
# 2. 実習3で、角度が実験からずれたのはなぜですか。
# 3. 実習4で、エネルギー差が基底関数にあまり左右されなかったのはなぜですか。
