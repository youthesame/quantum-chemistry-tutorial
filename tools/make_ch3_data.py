"""第3章のHTMLに埋め込む計算データを作り、docs/ch3_data.js に書き出す。

使い方: uv run python tools/make_ch3_data.py　所要時間は約5分
"""
import time

import numpy as np
from pyscf import dft, gto, scf

import common
from common import BOHR, HARTREE_EV, atoms, b64, grid_box, write_js

HARTREE_KCAL = 627.51
data = {}

# 場面1: 水の電子密度。面の内側にある電子の数を、等値面の値ごとに求める
WATER = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
mf = dft.RKS(mol, xc="b3lyp").run()
dm = mf.make_rdm1()
pts, origin, step = grid_box(mol, 2.6)
rho_box = dft.numint.eval_rho(mol, mol.eval_gto("GTOval", pts), dm)
rho_w = dft.numint.eval_rho(mol, mol.eval_gto("GTOval", mf.grids.coords), dm)
levels = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
inside = [float(mf.grids.weights[rho_w > iso] @ rho_w[rho_w > iso]) for iso in levels]
data["density"] = dict(origin=origin.tolist(), step=float(step), n=common.N, atoms=atoms(mol),
                       grids=[b64(rho_box)], levels=levels, inside=inside)
print("electrons inside", dict(zip(levels, np.round(inside, 2))))

# 場面2: 汎関数ごとの、水の原子化エネルギー。H2O → O + 2H に必要なエネルギー
def energy(atom, spin, xc):
    m = gto.M(atom=atom, basis="def2-tzvp", spin=spin, verbose=0)
    if xc == "HF":
        mf = scf.UHF(m) if spin else scf.RHF(m)
    else:
        mf = dft.UKS(m, xc=xc) if spin else dft.RKS(m, xc=xc)
    return mf.kernel()


atomization = []
for name, xc in [("HF", "HF"), ("LDA", "svwn"), ("PBE", "pbe"), ("r2SCAN", "r2scan"), ("B3LYP", "b3lyp"), ("PBE0", "pbe0")]:
    t0 = time.perf_counter()
    e = energy("O 0 0 0", 2, xc) + 2 * energy("H 0 0 0", 1, xc) - energy(WATER, 0, xc)
    atomization.append(dict(name=name, kcal=e * HARTREE_KCAL, seconds=time.perf_counter() - t0))
    print(atomization[-1])
data["atomization"] = atomization


# 場面3: メタン二量体。分散力を含まない B3LYP と、含む ωB97X-V の比較
def methane(z0, up):
    s = 1 if up else -1
    h = [(0, 0, s * 1.09), (1.028, 0, -s * 0.363), (-0.514, 0.890, -s * 0.363), (-0.514, -0.890, -s * 0.363)]
    return [("C", (0, 0, z0))] + [("H", (x, y, z0 + z)) for x, y, z in h]


def dimer_energy(geom, xc):
    m = gto.M(atom=geom, basis="def2-tzvp", verbose=0)
    mf = dft.RKS(m, xc=xc)
    mf.nlcgrids.level = 1  # 分散力の項の格子。粗くしても値はほぼ変わらず、速くなる
    return mf.kernel()


# 2つのメタンは、水素3個の面どうしを向かい合わせる
dists = [3.2, 3.4, 3.6, 3.8, 4.0, 4.4, 5.0, 6.0]
curves = {}
for xc in ["b3lyp", "wb97x_v"]:
    mono = dimer_energy(methane(0, False), xc)
    curves[xc] = [(dimer_energy(methane(0, False) + methane(R, True), xc) - 2 * mono) * HARTREE_KCAL for R in dists]
    print(xc, np.round(curves[xc], 3))
data["dimer"] = dict(dists=dists, b3lyp=curves["b3lyp"], wb97xv=curves["wb97x_v"],
                     geoms=[methane(0, False) + methane(R, True) for R in dists])

write_js("ch3", data)
