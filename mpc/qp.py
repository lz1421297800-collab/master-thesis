# -*- coding: utf-8 -*-
"""参数化 cvxpy LTV-QP (Hao et al. Eq.30-34). 只构造一次, 每步只更新 Parameter 数值 (~6-16ms).

决策变量: X (6, Np+1), U (2, Np), 松弛 S_ey / S_af.
参数    : A_k, B_k, D_k (逐步线性化), xi0, xi_ref, u_prev, ax_max/ax_min (摩擦圆逐步界), invVx (前轮侧偏约束).
"""
import cvxpy as cp
import numpy as np

from .params import MPCParams
from .vehicle import VehicleParams


class LTVQP:
    def __init__(self, veh: VehicleParams, mp: MPCParams, verbose: bool = True):
        self.mp = mp
        Np = mp.Np
        self.X = cp.Variable((6, Np + 1))
        self.U = cp.Variable((2, Np))
        self.P_A = [cp.Parameter((6, 6)) for _ in range(Np)]
        self.P_B = [cp.Parameter((6, 2)) for _ in range(Np)]
        self.P_D = [cp.Parameter(6) for _ in range(Np)]
        self.P_xi0 = cp.Parameter(6)
        self.P_xiref = cp.Parameter((6, Np + 1))
        self.P_uprev = cp.Parameter(2)
        self.P_axmax = cp.Parameter(Np)   # 摩擦圆: 逐步纵向加速度上界
        self.P_axmin = cp.Parameter(Np)   # 摩擦圆: 逐步制动下界 (保底 ≥4 m/s² 制动)
        self.P_invVx = cp.Parameter(Np, nonneg=True)   # 1/Vx 线性化值 (前轮侧偏约束用)
        Wdiag = np.array([mp.W_VX, mp.W_VY, mp.W_R, mp.W_EPSI, mp.W_EY, 0.0])

        X, U = self.X, self.U
        cons = [X[:, 0] == self.P_xi0]
        for k in range(Np):
            cons += [X[:, k + 1] == self.P_A[k] @ X[:, k] + self.P_B[k] @ U[:, k] + self.P_D[k]]
        cost = 0
        for k in range(1, Np + 1):
            cost += cp.sum(cp.multiply(Wdiag, cp.square(X[:, k] - self.P_xiref[:, k])))
        for k in range(Np):
            cost += mp.W_DELTA * cp.square(U[0, k]) + mp.W_AX * cp.square(U[1, k])
            prev = self.P_uprev if k == 0 else U[:, k - 1]
            cost += mp.W_DDELTA * cp.square(U[0, k] - prev[0]) + mp.W_DAX * cp.square(U[1, k] - prev[1])

        # 约束 Eq.32,33. 状态约束只加在 k>=1 (k=0 已被等式固定, 再加硬约束在越界初值下直接不可行).
        # e_y 用松弛变量软化: 正常时 slack=0; 车偏出赛道时 QP 仍有解, 引导回线.
        self.S_ey = cp.Variable(Np, nonneg=True)
        cons += [U[0, :] >= -veh.DELTA_MAX, U[0, :] <= veh.DELTA_MAX,
                 U[1, :] >= self.P_axmin, U[1, :] <= self.P_axmax,   # 摩擦圆 (代替常数箱约束)
                 X[4, 1:] >= -mp.EY_MAX - self.S_ey, X[4, 1:] <= mp.EY_MAX + self.S_ey,
                 X[0, 1:] >= mp.VX_MIN]
        # --- 前轮侧偏软约束 (治满舵卡死): 过峰后 dFy/dδ≈0, QP 打舵方向失定义 -> δ 冲到边界被 W_DDELTA 锁死.
        # 约束规划侧偏压在峰值下方 (AF_CAP=6.0°, 峰下 0.5° 余量): af_k ≈ δ_k - (Vy_k+lf·r_k)/Vx_hat_k.
        self.S_af = cp.Variable(Np, nonneg=True)
        af_lin = U[0, :] - cp.multiply(self.P_invVx, X[1, :Np] + veh.lf * X[2, :Np])
        cons += [af_lin <= veh.AF_CAP + self.S_af, af_lin >= -veh.AF_CAP - self.S_af]
        cost += mp.W_AF_SLACK_L1 * cp.sum(self.S_af) + mp.W_AF_SLACK_L2 * cp.sum_squares(self.S_af)
        cost += mp.W_EY_SLACK_L1 * cp.sum(self.S_ey) + mp.W_EY_SLACK_L2 * cp.sum_squares(self.S_ey)
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        if verbose:
            print("LTV-MPC (cvxpy) 构建完成: Np=%d, dt=%.3f, horizon=%.2fs" % (Np, mp.dt, mp.horizon))

    # ------------------------------------------------------------------
    def set_dynamics(self, k: int, A, B, D):
        self.P_A[k].value = A
        self.P_B[k].value = B
        self.P_D[k].value = D

    def set_bounds(self, x0, xi_ref, u_prev, ax_max, ax_min, inv_vx):
        self.P_xi0.value = x0
        self.P_xiref.value = xi_ref
        self.P_uprev.value = u_prev
        self.P_axmax.value = ax_max
        self.P_axmin.value = ax_min
        self.P_invVx.value = inv_vx

    def solve(self):
        """返回 (ok, U, X). eps 收紧到 1e-4 + polish: s 归零后 ||z||~40, 约束残差上界 ~5e-3."""
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                            max_iter=self.mp.OSQP_MAX_ITER, eps_abs=self.mp.OSQP_EPS,
                            eps_rel=self.mp.OSQP_EPS, polish=True)
            ok = self.prob.status in ('optimal', 'optimal_inaccurate')
        except Exception:
            ok = False
        if ok and self.U.value is not None:
            return True, self.U.value, self.X.value
        return False, None, None
