"""第1章のHTMLに埋め込む計算データを作り、docs/ch1_data.js に書き出す。

使い方: uv run python tools/make_ch1_data.py
"""
import numpy as np
from pyscf import dft, gto, scf

import common
from common import BOHR, HARTREE_EV, atoms, b64, volume, write_js

data = {}

# 場面1: 水素原子の軌道。シュレディンガー方程式を厳密に解いた形
orbitals = {
    "1s": (lambda x, y, z, r: np.exp(-r), 6),
    "2p": (lambda x, y, z, r: z * np.exp(-r / 2), 16),
    "3d": (lambda x, y, z, r: (3 * z ** 2 - r ** 2) * np.exp(-r / 3), 26),
}
hydrogen = {}
for name, (f, half) in orbitals.items():
    ax = np.linspace(-half, half, common.N)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    v = f(x, y, z, np.sqrt(x ** 2 + y ** 2 + z ** 2))
    v /= np.sqrt((v ** 2).sum() * (ax[1] - ax[0]) ** 3)
    hydrogen[name] = dict(origin=[-half] * 3, step=float(ax[1] - ax[0]), n=common.N,
                          atoms=[[1, 0.0, 0.0, 0.0]], grids=[b64(v)])
data["hydrogen"] = hydrogen

# 場面2: 原子核を止めて、H2 の結合距離ごとにエネルギーと結合性軌道を求める
# 0.05 Å 刻みで細かく動かせるよう、1コマの格子は小さくする。
# 2.0 Å より遠くでは、この計算法は結合エネルギーを実際より大きく見積もるので、そこで止める
common.N = 24
scan = []
for r in np.round(np.arange(0.5, 2.001, 0.05), 2):
    mol = gto.M(atom=f"H 0 0 {-r / 2}; H 0 0 {r / 2}", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol, xc="b3lyp").run()
    c = mf.mo_coeff[:, 0]
    d = volume(mol, [c if c.sum() > 0 else -c], pad=2.2)
    d.update(r=round(float(r), 2), energy=float(mf.e_tot))
    scan.append(d)
data["h2scan"] = scan

common.N = 32
# 場面3: 水の SCF。反復のたびに HOMO の形とエネルギーがどう変わるか
mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis="def2-svp", verbose=0)
mf = scf.RHF(mol)
mf.init_guess = "hcore"  # 実習ノートブックと同じ設定にして、反復の回数をそろえる
history = []
mf.callback = lambda env: history.append((env["e_tot"], env["mo_coeff"].copy(), env["mo_energy"].copy()))
mf.kernel()
homo = mol.nelectron // 2 - 1
S = mol.intor("int1e_ovlp")
final = mf.mo_coeff[:, homo]
steps = []
for k, (e, c, eps) in enumerate(history):  # 省略せずに全反復を残す
    v = c[:, homo]
    v = v if v @ S @ final > 0 else -v  # 符号をそろえて、色が反転しないようにする
    steps.append(dict(cycle=k + 1, energy=float(e), homo=float(eps[homo] * HARTREE_EV), c=v))
d = volume(mol, [s.pop("c") for s in steps])
d.update(steps=steps, final=float(mf.e_tot), cycles=len(history))
# 占有軌道5つの形とエネルギー
d2 = volume(mol, [mf.mo_coeff[:, i] for i in range(homo + 1)])
d2.update(energies=[float(x * HARTREE_EV) for x in mf.mo_energy[:homo + 1]])
data["scf"] = d
data["occupied"] = d2
print("SCF cycles", len(history), "E", mf.e_tot)
print("H2 scan", [(s["r"], round(s["energy"], 4)) for s in scan])

write_js("ch1", data)
