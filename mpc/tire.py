# -*- coding: utf-8 -*-
"""numpy 版轮胎 / 摩擦圆极限 (规划器与 QP 边界共用; 与 dynamics.py 的 CasADi 轮胎同一公式).

- mf_shape            : Pacejka 形状函数 sin(C·atan((1-E)Bα + E·atan(Bα)))
- alat_max            : 总轴稳态横向加速度上限 (直线载荷+下压力)
- alat_axle           : 逐轴稳态上限 min(前轴受限, 后轴受限)  (--axle-vgrip)
- ar_ss_from_util     : 后胎 MF 单调段反查 (--ref-ss 稳态参考侧偏角)
- axle_brake_bound_*  : 前/后轴专属刹车圆 (--front-circle), 含刹车重量转移不动点迭代
- axle_drive_bound_*  : RWD 后轴驱动圆 (--use-drive-map + --front-circle)
  *_util: QP 侧, 占用 = 轴向横向利用率 |Fy|/(μFz) (向量化)
  *_dem : 规划侧, 占用 = 稳态横向需求 v²|κ| 按横摆矩平衡分配到轴 (标量)
"""
import numpy as np

from .vehicle import VehicleParams


class TireLimits:
    def __init__(self, veh: VehicleParams):
        self.v = veh
        # MF 形状单调段反查表: 后胎 0..峰值角(5.8°) 单调增; 利用率超峰值时饱和在峰值角 (np.interp 自动夹紧)
        self._ar_grid = np.linspace(0.0, veh.AR_PEAK_RAD, 256)
        self._ar_shape = self.mf_shape(veh.PAC_BR, self._ar_grid)

    # ------------------------------------------------------------------
    def mf_shape(self, B, alpha):
        """MF 形状函数 (numpy 版, 与 CasADi 同式)."""
        Ba = B * np.asarray(alpha, float)
        return np.sin(self.v.PAC_C * np.arctan((1.0 - self.v.PAC_E) * Ba + self.v.PAC_E * np.arctan(Ba)))

    def mu_f(self, Fzf, corr=1.0):
        v = self.v
        return v.DY_REF_F * (0.5 * Fzf / v.FZ0T_F) ** (v.LS_EXPY_F - 1.0) * corr

    def mu_r(self, Fzr, corr=1.0):
        v = self.v
        return v.DY_REF_R * (0.5 * Fzr / v.FZ0T_R) ** (v.LS_EXPY_R - 1.0) * corr

    # ------------------------------------------------------------------
    def alat_max(self, vel):
        """模型稳态横向加速度上限 [m/s^2] (直线载荷, μ(载荷,速度) 曲面, 前后轴同时饱和)."""
        v = self.v
        vel = np.asarray(vel, float)
        Fzf = v.FZF0 + v.CZF * vel ** 2
        Fzr = v.FZR0 + v.CZR * vel ** 2
        muf = self.mu_f(Fzf) * np.maximum(1.0 - v.SS_F * vel, 0.5)
        mur = self.mu_r(Fzr) * np.maximum(1.0 - v.SS_R * vel, 0.5)
        return (muf * Fzf + mur * Fzr) / v.m

    def alat_axle(self, vel, corr=1.0):
        """稳态逐轴横向加速度上限 [m/s²]: min(前轴, 后轴) (直线载荷+下压力, 无纵向转移).

        总轴 a_max 只有前后轴同时饱和才可达; 稳态弯中轴力分配固定 (Fyf=m·ay·lr/L 横摆矩平衡),
        真实上限 = min(前轴受限, 后轴受限) ≤ 总轴. 前轴受限弯 (s2094 实测 af 钉峰) 差 2~3%."""
        v = self.v
        vel = np.asarray(vel, float)
        Fzf = v.FZF0 + v.CZF * vel ** 2
        Fzr = v.FZR0 + v.CZR * vel ** 2
        ay_f = self.mu_f(Fzf, corr) * Fzf * v.L / (v.lr * v.m)   # 前轴饱和时的整车 ay
        ay_r = self.mu_r(Fzr, corr) * Fzr * v.L / (v.lf * v.m)   # 后轴饱和时的整车 ay
        return np.minimum(ay_f, ay_r)

    def ar_ss_from_util(self, u):
        """后轴横向利用率 u=|Fyr|/(μr·Fzr) -> 稳态后轴侧偏角 [rad]."""
        return np.interp(np.abs(u), self._ar_shape, self._ar_grid)

    # ------------------------------------------------------------------
    def axle_brake_bound_util(self, vel, uF, uR, corr=1.0):
        """轴圆刹车上限 [m/s²], 占用以轴向横向利用率 uF/uR∈[0,1) 给出 (QP 侧, 向量化).
        纵向重量转移不动点迭代: 前轴支路斜率 ≈ +0.26, 后轴 ≈ -0.37, 均为压缩映射 -> 6 步收敛 (<0.01)."""
        v = self.v
        vel = np.asarray(vel, float)
        uF = np.clip(np.asarray(uF, float), 0.0, 0.98)
        uR = np.clip(np.asarray(uR, float), 0.0, 0.98)
        b = np.full(np.broadcast(vel, uF, uR).shape, 10.0)
        for _ in range(6):
            Fzf = v.FZF0 + v.m * b * v.hc / v.L + v.CZF * vel * vel
            Fzr = np.maximum(v.FZR0 - v.m * b * v.hc / v.L + v.CZR * vel * vel, 1.0e3)
            bF = self.mu_f(Fzf, corr) * Fzf * np.sqrt(1.0 - uF * uF) / (v.FRONT_BRK_SHARE * v.m)
            bR = self.mu_r(Fzr, corr) * Fzr * np.sqrt(1.0 - uR * uR) / ((1.0 - v.FRONT_BRK_SHARE) * v.m)
            b = np.maximum(np.minimum(bF, bR), 0.0)
        return b

    def axle_brake_bound_dem(self, vel, ay_dem, corr=1.0):
        """轴圆刹车上限 [m/s²], 占用以稳态横向需求 ay_dem=v²|κ| 给出 (规划侧, 标量).
        稳态轴力分配 (横摆矩平衡): Fyf=m·ay·lr/L, Fyr=m·ay·lf/L."""
        v = self.v
        vel = float(vel); ay_dem = abs(float(ay_dem))
        FyF = v.m * ay_dem * v.lr / v.L
        FyR = v.m * ay_dem * v.lf / v.L
        b = 10.0
        for _ in range(6):
            Fzf = v.FZF0 + v.m * b * v.hc / v.L + v.CZF * vel * vel
            Fzr = max(v.FZR0 - v.m * b * v.hc / v.L + v.CZR * vel * vel, 1.0e3)
            capF = max((self.mu_f(Fzf, corr) * Fzf) ** 2 - FyF ** 2, 0.0)
            capR = max((self.mu_r(Fzr, corr) * Fzr) ** 2 - FyR ** 2, 0.0)
            bF = capF ** 0.5 / (v.FRONT_BRK_SHARE * v.m)
            bR = capR ** 0.5 / ((1.0 - v.FRONT_BRK_SHARE) * v.m)
            b = max(min(bF, bR), 0.0)
        return b

    def axle_drive_bound_util(self, vel, uR, corr=1.0):
        """后轴驱动圆上限 [m/s², 轮胎级], 占用=后轴横向利用率 (QP 侧, 向量化). 驱动重量转移加载后轴."""
        v = self.v
        vel = np.asarray(vel, float)
        uR = np.clip(np.asarray(uR, float), 0.0, 0.98)
        a = np.full(np.broadcast(vel, uR).shape, 5.0)
        for _ in range(6):
            Fzr = np.maximum(v.FZR0 + v.m * a * v.hc / v.L + v.CZR * vel * vel, 1.0e3)
            a = np.maximum(self.mu_r(Fzr, corr) * Fzr * np.sqrt(1.0 - uR * uR) / v.m, 0.0)
        return a

    def axle_drive_bound_dem(self, vel, ay_dem, corr=1.0):
        """后轴驱动圆上限 (规划侧, 稳态后轴横向需求 Fyr=m·ay·lf/L, 标量)."""
        v = self.v
        vel = float(vel); FyR = v.m * abs(float(ay_dem)) * v.lf / v.L
        a = 5.0
        for _ in range(6):
            Fzr = max(v.FZR0 + v.m * a * v.hc / v.L + v.CZR * vel * vel, 1.0e3)
            a = max((self.mu_r(Fzr, corr) * Fzr) ** 2 - FyR ** 2, 0.0) ** 0.5 / v.m
        return a
