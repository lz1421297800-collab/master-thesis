# -*- coding: utf-8 -*-
"""CasADi 符号车辆动力学 (Hao et al. Eq.2-14): 风阻 + 纵向重量转移 + 载荷相关 Pacejka 轮胎
+ (可选) 后轴联合滑移椭圆 + (可选) 滚动阻力. 子步 Euler 离散并自动求 Jacobian.

状态 xi = [Vx, Vy, r, e_psi, e_y, s], 控制 u = [delta, ax] (ax = 纵向轮胎级加速度), 参数 kappa.
对外:
  fdot(xi, u, kappa)  -> 连续导数 (残差采集 / 分数步延迟补偿用)
  fABf(xi, u, kappa)  -> (A=∂f/∂xi, B=∂f/∂u, f) 离散一步 (LTV 线性化用)
"""
import casadi as ca

from .params import MPCParams
from .vehicle import VehicleParams


class VehicleDynamics:
    def __init__(self, veh: VehicleParams, mp: MPCParams):
        self.veh, self.mp = veh, mp
        xi = ca.SX.sym('xi', 6)   # [Vx,Vy,r,e_psi,e_y,s]
        u = ca.SX.sym('u', 2)     # [delta, ax]
        kp = ca.SX.sym('kp')      # 当前点曲率
        xidot = self._continuous(xi, u, kp)
        self.fdot = ca.Function('fdot', [xi, u, kp], [xidot])

        # --- 子步 Euler 离散 (2026-07-04, 治新胎起步抖动) ---
        # 单步 Euler 特征值 1-dt·λ, 横摆模态 λ=(lf²Cf+lr²Cr)/(Iz·Vx). 新胎刚度 +46% 后 Vx=15 处
        # λ·dt≈2.98 -> 特征值 -1.98 发散. N_SUB=4 把稳定边界压回 ~7m/s, 覆盖 V_MPC_OFF=5.
        x_sub = xi
        for _ in range(mp.N_SUB):
            x_sub = x_sub + (mp.dt / mp.N_SUB) * self.fdot(x_sub, u, kp)
        self.fABf = ca.Function('fABf', [xi, u, kp],
                                [ca.jacobian(x_sub, xi), ca.jacobian(x_sub, u), x_sub])

    # ------------------------------------------------------------------
    def _mf_shape(self, B, alpha):
        Ba = B * alpha
        return ca.sin(self.veh.PAC_C * ca.atan((1.0 - self.veh.PAC_E) * Ba + self.veh.PAC_E * ca.atan(Ba)))

    def _continuous(self, xi, u, kp):
        v, mp = self.veh, self.mp
        Vx, Vy, r, ep, ey, s = ca.vertsplit(xi)
        delta, ax = u[0], u[1]

        # 速度软下限 (侧偏角分母防奇异)
        eps = 0.1
        Vxs = 0.5 * (Vx + mp.VX_MIN + ca.sqrt((Vx - mp.VX_MIN) ** 2 + eps ** 2))

        # --- 法向载荷 Fz: 静载 + 纵向重量转移 (Eq.12-14) + 气动下压力 ---
        # 用 ax 近似 V̇x 避免代数环. 这一项让 ax 通过载荷耦合进横向动力学 (纵横耦合来源之一).
        dW = v.m * ax * v.hc / v.L
        Fzf = ca.fmax(v.FZF0 - dW + v.CZF * Vx ** 2, 1.0e3)
        Fzr = ca.fmax(v.FZR0 + dW + v.CZR * Vx ** 2, 1.0e3)

        # --- 侧偏角 (Eq.11) ---
        af = delta - (Vy + v.lf * r) / Vxs
        ar = -(Vy - v.lr * r) / Vxs

        # --- μ(载荷,速度) 曲面 + Pacejka MF ---
        muf = v.DY_REF_F * (0.5 * Fzf / v.FZ0T_F) ** (v.LS_EXPY_F - 1.0) * ca.fmax(1.0 - v.SS_F * Vx, 0.5)
        mur = v.DY_REF_R * (0.5 * Fzr / v.FZ0T_R) ** (v.LS_EXPY_R - 1.0) * ca.fmax(1.0 - v.SS_R * Vx, 0.5)
        Fyf = muf * Fzf * self._mf_shape(v.PAC_BF, af)
        Fyr = mur * Fzr * self._mf_shape(v.PAC_BR, ar)

        # ===== 后轴联合滑移椭圆 (--fy-combined, 2026-07-14) =====
        # 辨识: 实测 Fyr ≈ 0.88·纯侧偏模型·√(1−occ_r²), occ 0→0.85 逐桶吻合. 前轴无此损失 -> 只修后轴.
        # Fx 用轮胎级 ax; 驱动全在后轴 (RWD), 刹车按 1-bias; tanh(ax/0.5) 平滑切换保可导;
        # occ² 夹 0.9 防线性化梯度奇异. 副作用(有意): B 矩阵新增 dFyr/dax 耦合.
        if v.FY_COMBINED:
            s_drv = 0.5 * (1.0 + ca.tanh(ax / 0.5))          # ax>0 → 1 (驱动), ax<0 → 0 (刹车)
            rear_share = s_drv + (1.0 - s_drv) * (1.0 - v.FRONT_BRK_SHARE)
            occr2 = (v.m * ax * rear_share / (mur * Fzr)) ** 2
            Fyr = Fyr * ca.sqrt(1.0 - ca.fmin(occr2, 0.9))

        # --- 风阻 (Eq.8) + 滚动阻力 Rx (Eq.5, --model-rx) ---
        Fxw = v.CXW * Vx ** 2
        Floss = Fxw + ((v.RR0_TOT + v.RR1_TOT * Vx ** 2) if v.MODEL_RX else 0.0)

        # --- 底盘动力学 (Eq.5-7) ---
        dVx = ax - (Fyf * ca.sin(delta) + Floss) / v.m - v.g_acc * ca.sin(v.THETA) + r * Vy
        dVy = (Fyf * ca.cos(delta) + Fyr) / v.m - r * Vx
        dr = (v.lf * Fyf * ca.cos(delta) - v.lr * Fyr) / v.Iz

        # --- 运动学 (Eq.2-4) ---
        ce, se = ca.cos(ep), ca.sin(ep)
        sdot = (Vx * ce - Vy * se) / (1.0 - kp * ey)
        return ca.vertcat(dVx, dVy, dr, r - sdot * kp, Vx * se + Vy * ce, sdot)
