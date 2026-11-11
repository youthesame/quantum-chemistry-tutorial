"""第4章のHTMLに埋め込む計算データを作り、docs/ch4_data.js に書き出す。

使い方: uv run python tools/make_ch4_data.py　所要時間は約2分
"""
import numpy as np
from pyscf import dft, gto
from pyscf.hessian import thermo

from common import quiet_opt, write_js

HARTREE_KCAL = 627.51
data = {}


def water(r, theta):
    """O–H の長さ r [Å] と H–O–H の角度 theta [度] から水の座標を作る。分子は yz 平面に置く。"""
    h = np.radians(theta) / 2
    return f"O 0 0 0; H 0 {r * np.sin(h)} {r * np.cos(h)}; H 0 {-r * np.sin(h)} {r * np.cos(h)}"


def energy(atom):
    return dft.RKS(gto.M(atom=atom, basis="def2-svp", verbose=0), xc="b3lyp").kernel()


# 場面1: O–H の長さと角度を変えたときのエネルギーの地形
rs = np.round(np.linspace(0.85, 1.20, 15), 4)
thetas = np.round(np.linspace(70, 160, 19), 2)
E = np.array([[energy(water(r, t)) for r in rs] for t in thetas])
emin = E.min()
data["surface"] = dict(r=rs.tolist(), theta=thetas.tolist(), kcal=((E - emin) * HARTREE_KCAL).tolist())

# 構造最適化の道のり。ゆがんだ構造から始めて、1歩ごとの構造とエネルギーを記録する
path = []


def record(env):
    m = env["mol"]
    x = m.atom_coords(unit="Angstrom")
    a, b = x[1] - x[0], x[2] - x[0]
    path.append(dict(xyz=x.tolist(), r=float(np.linalg.norm(a)),
                     theta=float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))),
                     kcal=(env["energy"] - emin) * HARTREE_KCAL))


mol = gto.M(atom=water(1.15, 145), basis="def2-svp", verbose=0)
mf = dft.RKS(mol, xc="b3lyp")
mol_eq = quiet_opt(mf, callback=record)
data["path"] = path
print("steps", len(path), [(round(p["r"], 3), round(p["theta"], 1), round(p["kcal"], 2)) for p in path])


# 場面2と3: 振動の様子。最適化した水と、まっすぐに伸ばした水
def modes(mol):
    mf = dft.RKS(mol, xc="b3lyp").run()
    res = thermo.harmonic_analysis(mol, mf.Hessian().kernel())
    freq = res["freq_wavenumber"]
    # 虚数の振動数は、負の数として返す
    freq = [float(f.real) if abs(f.imag) < 1e-6 else -float(f.imag) for f in np.atleast_1d(freq)]
    return dict(xyz=mol.atom_coords(unit="Angstrom").tolist(), freq=freq, modes=np.asarray(res["norm_mode"]).tolist())


data["vib"] = modes(mol_eq)
r_eq = path[-1]["r"]
data["linear"] = modes(gto.M(atom=water(r_eq, 180), basis="def2-svp", verbose=0))
print("freq", data["vib"]["freq"], "linear", data["linear"]["freq"])

write_js("ch4", data)
