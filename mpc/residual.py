# -*- coding: utf-8 -*-
"""NN 残差数据在线采集 (2026-07-07 / v2 配对 2026-07-14).

每 MPC 步: 残差 = 实测导数(Vx,Vy,r) - 模型连续导数 fdot(x,u,κ)[:3]. 延迟一步配对 (本步存 x_k + 模型导数,
下步用实测 x_{k+1} 按真实 dt 差分). 热循环内只做 1 次 fdot + list.append, 文件 IO 全部推到 flush().
seg 段号在 KIN/PANIC/reset 断链时 +1, 防跨断点差分.
v2: 特征里的转向改为 δ_{k-N} (区间内真正作用的指令, N=round(DELAY_STEPS)); 文件名带 d{N}/sg/mu 后缀,
模型语义变更都分文件, 勿混训. 训练: python learn_residual_nn.py <csv>
"""
import csv
import os
import time

import numpy as np

from .config import RunConfig
from .dynamics import VehicleDynamics
from .params import MPCParams
from .vehicle import VehicleParams

RESID_COLS = ['Vx', 'Vy', 'r', 'delta_applied', 'ax_cmd', 'Fzf', 'Fzr', 'e_y', 'seg',
              'resdot_Vx', 'resdot_Vy', 'resdot_r']


class ResidualCollector:
    def __init__(self, cfg: RunConfig, veh: VehicleParams, mp: MPCParams, dyn: VehicleDynamics, enabled: bool = True):
        self.enabled = enabled
        self.veh, self.mp, self.dyn = veh, mp, dyn
        self.n_delay = cfg.n_resid_delay
        self.path = (f'residual_dataset_d{cfg.n_resid_delay}'
                     + (f'_sg{cfg.steer_gain:g}' if cfg.steer_gain != 1.0 else '')
                     + (f'_mu{cfg.mu_scale_f:g}-{cfg.mu_scale_r:g}'
                        if (cfg.mu_scale_f != 1.0 or cfg.mu_scale_r != 1.0) else '')
                     + '.csv')
        self.rows = []
        self._prev = None            # (x_k[3], model_deriv_k[3], feat_k[7], e_y_k)
        self.seg = 0
        self.runtag = time.strftime('%m%d_%H%M%S')

    def update(self, mode, u_hist, Vx, Vy, r, e_psi, e_y, ax, kappa0, dt):
        """主循环每步调用 (在 u_hist 追加本步指令之后)."""
        if not self.enabled:
            return
        v, mp = self.veh, self.mp
        if mode == 'MPC' and len(u_hist) > self.n_delay:
            # 作动延迟对齐: [k,k+1) 区间真正作用的转向是 N 步前发出的 δ_{k-N}; ax 实测滞后≈0 保持当前值
            d_eff = float(u_hist[-1 - self.n_delay][0])
            md = np.asarray(self.dyn.fdot(
                np.array([max(Vx, mp.VX_MIN), Vy, r, e_psi, e_y, 0.0]),
                np.array([d_eff, ax]), float(kappa0))).flatten()[:3]
            Fzf = v.FZF0 - v.m * ax * v.hc / v.L + v.CZF * Vx ** 2     # 与 nn_ff 的载荷式一致
            Fzr = v.FZR0 + v.m * ax * v.hc / v.L + v.CZR * Vx ** 2
            feat = [Vx, Vy, r, d_eff, ax, Fzf, Fzr]
            if self._prev is not None and 0.02 < dt < 0.15:   # 丢弃卡顿/reset 异常步长
                xp, mdp, fp, eyp = self._prev
                meas = (np.array([Vx, Vy, r]) - xp) / dt      # 实测导数 (真实 dt 差分)
                res = meas - mdp                               # 残差 = 实测 - 模型
                self.rows.append(fp + [eyp, f'{self.runtag}_{self.seg}',
                                       float(res[0]), float(res[1]), float(res[2])])
            self._prev = (np.array([Vx, Vy, r]), md, feat, float(e_y))
        else:
            if self._prev is not None:
                self.seg += 1        # 断链 (KIN/PANIC), 防跨点差分
            self._prev = None

    def flush(self):
        """批量追加写盘 (跨 run 累积; 供 learn_residual_nn.py 训练)."""
        if not self.enabled:
            return
        if not self.rows:
            print('[resid] 无残差样本 (未进入 MPC?)')
            return
        try:
            new = not os.path.exists(self.path)
            with open(self.path, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(RESID_COLS)
                w.writerows(self.rows)
            print(f'[resid] 追加 {len(self.rows)} 条残差样本 -> {self.path} (seg {self.seg+1} 段)')
        except Exception as e:
            print(f'[resid] 写盘失败(非致命): {e}')
