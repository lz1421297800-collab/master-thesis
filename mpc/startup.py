# -*- coding: utf-8 -*-
"""办法2: 低速运动学起步控制器.

MPC 的横向动力学在 Vx->0 时退化 (侧偏角公式除以速度), 不适合起步. 运动学自行车模型不除以速度,
在 Vx=0 也成立, 专门负责低速段 (Vx < V_MPC_ON). 输出 (delta, ax) 与 MPC 同构, 主循环统一处理.
"""
import numpy as np

from .track import Track
from .vehicle import VehicleParams


class KinematicStartup:
    USE_FB = True          # 起步反馈总开关 (首次验证曾设 False 纯前馈直线开出去)
    V_CAP = 22.0
    AX_MAX = 8.0
    AX_MIN = -3.0
    K_E = 0.04             # 横向偏差增益 [rad/m]
    K_PSI = 0.30           # 航向偏差增益 [rad/rad]

    def __init__(self, track: Track, veh: VehicleParams):
        self.track, self.veh = track, veh

    def control(self, idx, Vx, e_y, e_psi, Vx_ref0):
        """低速段: 运动学前馈 delta=atan(L·κ) + 弱反馈 + 弱 P 速度控制."""
        kappa0 = self.track.kappa[idx]
        delta_ff = np.arctan(self.veh.L * kappa0)
        if self.USE_FB:
            # 反馈符号已翻正: 本坐标约定下 e_y/e_psi 为正时需负向修正
            delta_fb = -(self.K_E * e_y + self.K_PSI * e_psi)
        else:
            delta_fb = 0.0
        # ×STEER_GAIN: 起步增益在旧标定下验证, 标定修正后乘回 gain 保持同样的物理手轮输出
        delta = (delta_ff + delta_fb) * self.veh.STEER_GAIN

        v_target = min(float(Vx_ref0), self.V_CAP)
        ax = np.clip(0.9 * (v_target - max(float(Vx), 0.0)), self.AX_MIN, self.AX_MAX)
        return float(delta), float(ax)
