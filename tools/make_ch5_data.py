"""第5章のHTMLに埋め込む計算データを作り、docs/ch5_data.js に書き出す。

使い方: uv run python tools/make_ch5_data.py　所要時間は約5分
"""
import numpy as np
from pyscf import dft, gto, tdscf

import common
from common import HARTREE_EV, volume, write_js

common.N = 36


def polyene(n):
    """二重結合が n 個つながった、まっすぐな全トランスのポリエンの座標を作る。"""
    atoms, cc, ch = [], 1.40, 1.09
    for i in range(2 * n):
        x, y = i * cc * np.cos(np.radians(30)), (i % 2) * cc * np.sin(np.radians(30))
        atoms.append(("C", (x, y, 0.0)))
        up = 1 if i % 2 else -1
        atoms.append(("H", (x, y + up * ch, 0.0)))
        if i in (0, 2 * n - 1):  # 端の炭素には水素をもう1つ付ける
            side = -1 if i == 0 else 1
            atoms.append(("H", (x + side * ch * np.cos(np.radians(30)), y - up * ch * np.sin(np.radians(30)), 0.0)))
    return atoms


chains = []
for n in [1, 2, 3, 4, 5, 6, 8, 11]:
    # 実習ノートブックと同じく、構造最適化はせずに理想的な形のまま計算する
    mol = gto.M(atom=polyene(n), basis="def2-svp", verbose=0)
    mf = dft.RKS(mol, xc="b3lyp").run()
    homo = mol.nelectron // 2 - 1
    td = tdscf.TDA(mf)
    td.nstates = 4
    td.kernel()
    # 4つの励起のうち、光を最もよく吸収するものを選ぶ
    f = td.oscillator_strength()
    k = int(np.argmax(f))
    ev = float(td.e[k] * HARTREE_EV)
    # 長い鎖は格子が粗くなるので、軌道の形は二重結合6個までにする
    d = volume(mol, [mf.mo_coeff[:, homo], mf.mo_coeff[:, homo + 1]], pad=2.4) if n <= 6 else {}
    d.update(bonds=n, homo=float(mf.mo_energy[homo] * HARTREE_EV), lumo=float(mf.mo_energy[homo + 1] * HARTREE_EV),
             excite=ev, nm=1239.84 / ev)
    chains.append(d)
    print(n, round(d["homo"], 2), round(d["lumo"], 2), round(ev, 2), round(1239.84 / ev), flush=True)

write_js("ch5", dict(chains=chains))
