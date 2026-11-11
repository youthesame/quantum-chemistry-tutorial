"""第2章のHTMLに埋め込む計算データを作り、docs/ch2_data.js に書き出す。

使い方: uv run python tools/make_ch2_data.py
"""
import time

import numpy as np
from pyscf import dft, gto, scf

from common import BOHR, HARTREE_EV, atoms, quiet_opt, volume, write_js

data = {}

# 場面1: H2 の LCAO。部品は各Hの1s関数1個ずつ
mol = gto.M(atom="H 0 0 -0.37; H 0 0 0.37", basis="sto-3g")
mf = scf.RHF(mol).run(verbose=0)
d = volume(mol, np.eye(2))
d.update(F=(mf.get_fock() * HARTREE_EV).tolist(), S=mf.get_ovlp().tolist())
data["h2"] = d

# 場面2: スレーター関数とガウス関数の「山」の形
from scipy.optimize import minimize

r = np.linspace(0, 12, 6001)
dr = r[1] - r[0]
ip = lambda f, g: 4 * np.pi * np.sum(f * g * r ** 2) * dr
slater = np.exp(-r) / np.sqrt(np.pi)
prim = lambda a: (2 * a / np.pi) ** 0.75 * np.exp(-a * r ** 2)


def fit(n):
    def solve(la):
        G = [prim(a) for a in np.exp(la)]
        Sm = np.array([[ip(x, y) for y in G] for x in G])
        b = np.array([ip(x, slater) for x in G])
        c = np.linalg.solve(Sm, b)
        return -(c @ b) / np.sqrt(c @ Sm @ c), c
    res = minimize(lambda la: solve(la)[0], np.log(np.geomspace(0.12, 0.3 * 4.5 ** (n - 1), n)),
                   method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-10))
    ov, c = solve(res.x)
    a = np.exp(res.x)
    norm = np.sqrt(ip(sum(ci * prim(ai) for ci, ai in zip(c, a)), sum(ci * prim(ai) for ci, ai in zip(c, a))))
    return dict(alpha=a.tolist(), coef=(c / norm * (2 * a / np.pi) ** 0.75).tolist(), overlap=float(-ov))


data["gauss"] = [fit(n) for n in range(1, 7)]

# 場面3: 炭素の2p軌道を、内側用と外側用の部品の配合で膨らませたり縮めたりする
mol = gto.M(atom="C 0 0 0", basis="def2-svp", spin=2)
labels = [l.split()[2] for l in mol.ao_labels()]
idx = [labels.index("2pz"), labels.index("3pz")]
grids = dft.gen_grid.Grids(mol).build()
ao = mol.eval_gto("GTOval", grids.coords)[:, idx]
dist = np.linalg.norm(grids.coords, axis=1) * BOHR
d = volume(mol, np.eye(mol.nao)[:, idx].T, pad=3.6)
d.update(S=(ao.T * grids.weights) @ ao, R=(ao.T * grids.weights * dist) @ ao)
d.update(S=d["S"].tolist(), R=d["R"].tolist())
data["carbon_p"] = d

# 場面4: 分極関数の有無で、水の構造と非共有電子対がどう変わるか
nopol = {el: [s for s in gto.basis.load("def2-svp", el) if s[0] <= lmax] for el, lmax in {"H": 0, "O": 1}.items()}
water = {}
for name, basis in [("nopol", nopol), ("pol", "def2-svp")]:
    mol = gto.M(atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", basis=basis, verbose=0)
    mol = quiet_opt(dft.RKS(mol, xc="b3lyp"))
    mf = dft.RKS(mol, xc="b3lyp").run()
    o, h1, h2 = mol.atom_coords(unit="Angstrom")
    v1, v2 = h1 - o, h2 - o
    angle = float(np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))))
    nocc = mol.nelectron // 2
    d = volume(mol, [mf.mo_coeff[:, nocc - 2]])  # HOMO-1：分子面内の非共有電子対
    d.update(angle=angle)
    water[name] = d
data["water"] = water

# 場面5: 部品を増やすと、精度と計算時間はどうなるか（エチレン）
eth = "C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; H 0 0.923 -1.238; H 0 -0.923 -1.238"
ladder = []
for basis in ["def2-svp", "def2-tzvp", "def2-qzvp"]:
    mol = gto.M(atom=eth, basis=basis, verbose=0)
    t0 = time.perf_counter()
    e = dft.RKS(mol, xc="b3lyp").kernel()
    ladder.append(dict(basis="def2-" + basis[5:].upper(), nao=mol.nao, energy=e, seconds=time.perf_counter() - t0))
    print(ladder[-1])
data["ladder"] = ladder

write_js("ch2", data)
print("H2O", {k: v["angle"] for k, v in water.items()})
