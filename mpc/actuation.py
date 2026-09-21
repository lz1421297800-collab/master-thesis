# -*- coding: utf-8 -*-
"""控制量 -> AC 踏板/方向盘 指令映射, 以及饱和后的反算."""
import numpy as np

from .learned_maps import DriveMap
from .tire import TireLimits
from .vehicle import VehicleParams


class Actuator:
    def __init__(self, veh: VehicleParams, tire: TireLimits, drive_map: DriveMap):
        self.veh, self.tire, self.drive_map = veh, tire, drive_map

    def pedal_from_ax(self, ax_cmd, Vx):
        """ax [m/s²] -> (acc, brake) ∈ [-1,1]².
        驱动: drive map 查表逆映射 (交付=需求 的油门位置) / 旧线性映射;
        制动: 二次踏板曲线 u=√(需求/容量), 容量≈0.95·a_max(v) (下压力刹车; 旧 _BRK_REF=7.5 高速严重低估)."""
        dm = self.drive_map
        if ax_cmd >= 0:
            if dm.enabled:
                # 行经 cummax 单调不减; 加微小斜坡破平台使 np.interp 反查良定义
                row = dm.row(abs(Vx)) + 1e-6 * np.arange(len(dm.ACC))
                acc = float(np.clip(np.interp(ax_cmd, row, dm.ACC), -1.0, 1.0))
                return acc, -1.0
            pedal_ref = float(np.interp(abs(Vx), dm.DRV_V, dm.DRV_REF))
            u = float(np.clip(ax_cmd / max(pedal_ref, 0.1), 0.0, 1.0))
            return 2.0 * u - 1.0, -1.0       # u=0 -> acc=-1 (松油), u=1 -> +1 (全油)
        brk_cap = max(7.5, 0.95 * float(self.tire.alat_max(abs(Vx))))
        u = float(np.sqrt(np.clip(-ax_cmd / brk_cap, 0.0, 1.0)))
        return -1.0, 2.0 * u - 1.0           # 松油; u=0 -> brake=-1, u=1 -> 全刹

    def steer_from_delta(self, delta):
        """STEER_GAIN: 要得到真实前轮角 delta, 只需 1/gain 的手轮指令."""
        v = self.veh
        steer = v.STEER_CMD_SIGN * np.rad2deg(delta) * v.STEER_RATIO / (v.STEER_LOCK * v.STEER_GAIN)
        return float(np.clip(steer, -1, 1))

    def delta_from_steer(self, steer):
        """反算实际 delta (steer 被 ±1 饱和时), 给下一步 warm start / du penalty 用."""
        v = self.veh
        return np.deg2rad(v.STEER_CMD_SIGN * steer * v.STEER_LOCK * v.STEER_GAIN / v.STEER_RATIO)
