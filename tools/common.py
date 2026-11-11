"""各章のデータ作成スクリプトで共通の関数。"""
import base64
import io
import json
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
from pyscf.geomopt.geometric_solver import optimize

BOHR = 0.529177
HARTREE_EV = 27.2114
N = 40  # 1辺あたりの格子点数


def b64(a):
    return base64.b64encode(np.asarray(a, dtype=np.float32).tobytes()).decode()


def grid_box(mol, pad=2.6):
    """分子を囲む立方体の格子を作る。単位はbohr。"""
    xyz = mol.atom_coords()
    lo, hi = xyz.min(0) - pad / BOHR, xyz.max(0) + pad / BOHR
    size = (hi - lo).max()
    center = (lo + hi) / 2
    origin = center - size / 2
    step = size / (N - 1)
    ax = np.arange(N) * step
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + origin
    return pts, origin, step


def atoms(mol):
    return [[int(mol.atom_charge(i)), *map(float, mol.atom_coord(i))] for i in range(mol.natm)]


def volume(mol, coeff_vectors, pad=2.6):
    pts, origin, step = grid_box(mol, pad)
    ao = mol.eval_gto("GTOval", pts)
    return dict(origin=origin.tolist(), step=float(step), n=N, atoms=atoms(mol),
                grids=[b64(ao @ c) for c in coeff_vectors])


def quiet_opt(mf, **kw):
    """geomeTRIC の途中経過のログを隠して構造最適化する。"""
    with redirect_stderr(io.StringIO()):
        return optimize(mf, **kw)


def write_js(name, data):
    """docs/<name>_data.js に window.<NAME> = {...} として書き出す。"""
    out = Path(__file__).resolve().parents[1] / "docs" / f"{name}_data.js"
    out.write_text(f"window.{name.upper()} = " + json.dumps(data) + ";\n")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
