# -*- coding: utf-8 -*-
"""NN 残差前馈 (协议③': 值修正 + 冻结梯度) — 只修 QP 的 D 项, 永不求导.

教训(实测): NN 直接进动力学求导 -> 坏梯度抵消解析转向增益 -> B 矩阵塌掉 -> 失稳.
v2 结构性安全设计 (2026-07-05):
 (1) 特征无 δ (6维: Vx,Vy,r,ax,Fzf,Fzr): 对转向的梯度结构性为零;
 (2) 激励门控 gate=1-exp(-(r/0.15)²-(Vy/1.0)²): 直线上修正=0, 弯中 gate≈1;
 (3) 分通道限幅 NN_CLAMP_VEC=[Vx±3, Vy±1, r±1].
加载后做红旗自检 (良性直线输出≈0, 弯中输出有界), 不过则自动禁用.
"""
from typing import Optional

import casadi as ca
import numpy as np

from .config import RunConfig
from .params import MPCParams
from .vehicle import VehicleParams


class ResidualNN:
    def __init__(self, fn: ca.Function, file: str):
        self._fn = fn
        self.file = file

    def __call__(self, x, u) -> np.ndarray:
        """残差导数修正 [dVx, dVy, dr] (已门控+限幅)."""
        return np.asarray(self._fn(x, u)).flatten()

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, cfg: RunConfig, veh: VehicleParams, mp: MPCParams, verbose: bool = True) -> Optional['ResidualNN']:
        """返回可用的 ResidualNN; 未请求 / 文件缺失 / 自检不过 -> None (纯解析)."""
        if not cfg.use_nn_ff:
            return None
        try:
            nn = np.load(cfg.nn_file)
        except FileNotFoundError:
            if verbose:
                print(f'[NN-FF] {cfg.nn_file} 不存在 -> 纯解析')
            return None
        W0, W1, W2 = ca.DM(nn['W0']), ca.DM(nn['W1']), ca.DM(nn['W2'])
        b0, b1, b2 = ca.DM(nn['b0']), ca.DM(nn['b1']), ca.DM(nn['b2'])
        Xm, Xs, Ym, Ys = ca.DM(nn['Xm']), ca.DM(nn['Xs']), ca.DM(nn['Ym']), ca.DM(nn['Ys'])

        xf = ca.SX.sym('xf', 6); uf = ca.SX.sym('uf', 2)
        Fzff = veh.FZF0 + veh.CZF * xf[0] ** 2 - veh.m * uf[1] * veh.hc / veh.L
        Fzrf = veh.FZR0 + veh.CZR * xf[0] ** 2 + veh.m * uf[1] * veh.hc / veh.L
        feat = ca.vertcat(xf[0], xf[1], xf[2], uf[1], Fzff, Fzrf)   # 无 δ
        Xn = (feat - Xm) / Xs
        h0 = ca.tanh(W0.T @ Xn + b0)
        h1 = ca.tanh(W1.T @ h0 + b1)
        resid_raw = (W2.T @ h1 + b2) * Ys + Ym
        gate = 1.0 - ca.exp(-((xf[2] / 0.15) ** 2 + (xf[1] / 1.0) ** 2))
        clv = ca.DM(mp.NN_CLAMP_VEC)
        resid = gate * ca.fmin(ca.fmax(resid_raw, -clv), clv)
        fn = ca.Function('nn_resid', [xf, uf], [resid])
        obj = cls(fn, cfg.nn_file)

        # 自检: 良性(直线)输出须≈0 (门控保证); 弯中数据域内输出须有界
        benign = float(np.abs(obj([40, 0, 0, 0, 0, 0], [0, 0])).max())
        incorner = max(float(np.abs(obj([vt, -0.5, vt * 0.012, 0, 0, 0], [0.05, 2.0])).max())
                       for vt in (25., 40., 55.))
        if benign > 0.05 or incorner > 3.0:
            if verbose:
                print(f'[NN-FF] 自检未过(良性{benign:.3f}/限0.05, 弯中{incorner:.2f}/限3.0) -> 自动禁用')
            return None
        if verbose:
            print(f'[NN-FF] 自检通过(良性{benign:.3f}, 弯中幅值{incorner:.2f}); '
                  f'无δ特征+激励门控+分通道限幅{mp.NN_CLAMP_VEC.tolist()}')
        return obj
