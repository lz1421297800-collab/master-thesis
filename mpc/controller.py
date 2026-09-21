# -*- coding: utf-8 -*-
"""论文式 LTV-MPC 控制器: solve(x0, Vx_ref, kappa, corr) -> (delta, ax).

每步流程:
  1. 作动延迟补偿 (DELAY_COMP / --delay-steps): 用已发指令历史把 x0 前推 d 步
  2. 参考轨迹 xi_ref (--ref-ss: Vy/e_psi 取模型稳态值)
  3. 自洽线性化点: 上一步 QP 解 (状态+控制) 与启发式轨迹 (Eq.34) 阻尼混合; --steer-calm 再 EMA
  4. 沿线性化轨迹逐步 CasADi Jacobian -> A,B,D (NN 前馈只修 D 项)
  5. 摩擦圆逐步纵向界: 驱动 = 牵引曲线 ∩ 圆 (∩ 后轴驱动圆); 刹车 = 轴圆/总圆 × 帽 × NN 修正
  6. OSQP 求解; 失败时衰减上一步控制并清热启动
"""
import time
from typing import Optional

import numpy as np

from .config import RunConfig
from .dynamics import VehicleDynamics
from .learned_maps import BrakeCap, BrakeCircleCorr, DriveMap
from .nn_ff import ResidualNN
from .params import MPCParams
from .qp import LTVQP
from .tire import TireLimits
from .vehicle import VehicleParams


class MPCController:
    def __init__(self, veh: VehicleParams, mp: MPCParams, cfg: RunConfig,
                 dyn: VehicleDynamics, qp: LTVQP, tire: TireLimits,
                 brake_cap: BrakeCap, bc_corr: BrakeCircleCorr, drive_map: DriveMap,
                 nn: Optional[ResidualNN] = None):
        self.veh, self.mp, self.cfg = veh, mp, cfg
        self.dyn, self.qp, self.tire = dyn, qp, tire
        self.brake_cap, self.bc_corr, self.drive_map, self.nn = brake_cap, bc_corr, drive_map, nn
        self.brk_eff = mp.BRK_EFF             # 爬升协议会改它 (与 planner.brk_eff 同步)
        self.u_prev = np.array([0.0, 0.0])    # [delta, ax]
        self.solve_ok = True
        self.solve_ms = 0.0
        self.U_prev_traj = None               # 上一步控制序列
        self.X_prev_traj = None               # 上一步状态序列 (自洽线性化)
        self.U_lin_traj = None                # --steer-calm: 平滑后的线性化控制轨迹
        self.ref_vy0 = 0.0                    # --ref-ss 日志诊断: k=1 参考 Vy/e_psi
        self.ref_epsi0 = 0.0
        self.u_hist = []                      # --delay-steps: 已发指令历史 (旧->新)

    # 兼容主循环: sol_prev=None 清空 warm-start 轨迹 (冷启动)
    @property
    def sol_prev(self):
        return self.U_prev_traj

    @sol_prev.setter
    def sol_prev(self, v):
        if v is None:
            self.U_prev_traj = None
            self.X_prev_traj = None
            self.U_lin_traj = None

    def reset(self):
        """server reset / 预热后: 清热启动 + 指令历史."""
        self.sol_prev = None
        self.u_prev = np.array([0.0, 0.0])
        self.u_hist = []
        self.solve_ok = True
        self.solve_ms = 0.0

    def push_u_hist(self, u_actual):
        """每步都记 (含 KIN/PANIC), 保证跨模式连续; 只留最近 6 步."""
        self.u_hist.append(np.asarray(u_actual, float).copy())
        if len(self.u_hist) > 6:
            self.u_hist.pop(0)

    # ------------------------------------------------------------------
    def _delay_compensate(self, x0, kappa0):
        """作动延迟补偿: 命令要 ~d 步后才在状态上体现; 用历史已发指令把状态前推 d 个 dt.
        分数 d=n+f: [k,k+f) 作用 u_{k-n-1}, 之后 n 个整步 u_{k-n}..u_{k-1}. d=1 走原路径 (u_prev), 黄金基线逐位不变."""
        mp = self.mp
        n_full = int(np.floor(mp.DELAY_STEPS + 1e-9))
        f_part = mp.DELAY_STEPS - n_full
        h = self.u_hist

        def u_back(j):   # u_{k-j}; 历史不足时退化到最旧可用/u_prev
            return (np.asarray(h[-j], float) if len(h) >= j
                    else (np.asarray(h[0], float) if len(h) else self.u_prev))

        if f_part > 1e-6:
            uo = u_back(n_full + 1)
            for _ in range(4):   # 分数步: 连续 fdot 4 子步 Euler (与 N_SUB 同稳定裕度)
                x0 = x0 + (f_part * mp.dt / 4) * np.asarray(self.dyn.fdot(x0, uo, kappa0)).flatten()
        for j in range(n_full, 0, -1):
            du = self.u_prev if (n_full == 1 and f_part <= 1e-6) else u_back(j)
            _, _, x0f = self.dyn.fABf(x0, du, kappa0)
            x0 = np.array(x0f).flatten()
        return x0

    def _build_xi_ref(self, x0, Vx_ref, kappa):
        """参考状态轨迹 xi_ref (6, Np+1). 默认贴线: e_psi=e_y=Vy=0, r=Vx·κ, s 累积.
        --ref-ss: 近极限稳态过弯物理上必然侧滑 (后轴要输出 Fyr=m·ay·lf/L 就必须有 ar_ss), 钉 0 的参考让
        W_VY/W_EPSI 惩罚物理必需状态 -> 弯中稳态推头偏置. 推导: 横摆矩平衡 -> 利用率 -> MF 反查 ar_ss;
        Vy_ss=lr·r-ar_ss·Vx; 贴线条件 ė_y=0 -> e_psi_ss=-atan(Vy_ss/Vx). Vy_ref 与 e_psi_ref 必须成对给."""
        mp, v = self.mp, self.veh
        Np = mp.Np
        xi_ref = np.zeros((6, Np + 1))
        Vxr = np.maximum(Vx_ref, mp.VX_MIN)
        xi_ref[0, :] = Vxr
        r_ref = Vx_ref * kappa
        xi_ref[2, :] = r_ref
        if self.cfg.ref_ss:
            ay = Vxr ** 2 * kappa                        # 稳态横向需求 (带符号)
            Fzr = v.FZR0 + v.CZR * Vxr ** 2              # 直线载荷口径 (同 v_grip, 无纵向转移)
            mur = self.tire.mu_r(Fzr)
            # 利用率夹 0.95: 只防瞬态扫进峰值附近的反查病态区 (dα/du→∞) 造成参考跳变
            util = np.minimum(v.m * np.abs(ay) * v.lf / (v.L * mur * Fzr), 0.95)
            ar_ss = np.sign(ay) * self.tire.ar_ss_from_util(util)
            vy_ss = v.lr * r_ref - ar_ss * Vxr
            xi_ref[1, :] = vy_ss
            xi_ref[3, :] = -np.arctan(vy_ss / Vxr)
        s_acc = x0[5]
        for k in range(Np + 1):
            xi_ref[5, k] = s_acc
            s_acc += max(Vx_ref[k], mp.VX_MIN) * mp.dt
        self.ref_vy0 = float(xi_ref[1, 1])            # 日志诊断用 (k=1: 代价首个作用点)
        self.ref_epsi0 = float(xi_ref[3, 1])
        return xi_ref

    def _linearization_traj(self, x0, xi_ref):
        """自洽线性化点 (修奇偶震荡): 状态展开点与控制展开点必须来自同一条轨迹.
        用上一步 QP 解做展开点, 与启发式轨迹 (Eq.34: xi_ref + lam·(x0-xi_ref_0), lam 1->0) 阻尼混合."""
        mp = self.mp
        lam = np.linspace(1.0, 0.0, mp.Np + 1)
        xi_heur = xi_ref + lam[None, :] * (x0 - xi_ref[:, 0])[:, None]
        if (self.X_prev_traj is not None) and (self.U_prev_traj is not None):
            xi_hat = mp.ALPHA_LIN * self.X_prev_traj + (1.0 - mp.ALPHA_LIN) * xi_heur
            xi_hat[:, 0] = x0
            u_hat = self.U_prev_traj.copy()
            # --steer-calm (2): 线性化控制轨迹 EMA. 近胎峰 dFy/dδ→0, 上一步解的 δ 抖动经 B(u_hat)
            # 再进 QP 形成自激 dither. 只平滑展开点, 不平滑 warm-start/输出.
            if self.cfg.steer_calm and self.U_lin_traj is not None:
                u_hat = 0.5 * u_hat + 0.5 * self.U_lin_traj
        else:
            xi_hat = xi_heur
            u_hat = np.zeros((2, mp.Np))
        if self.cfg.steer_calm:
            self.U_lin_traj = u_hat.copy()
        return xi_hat, u_hat

    def _ax_bounds(self, Vx_ref, kappa, xi_hat, u_hat, corr_k):
        """摩擦圆: 横向占用决定纵向余量 -> 逐步 (ax_max, ax_min, 1/Vx).
        横向占用按 min(v_ref, 预测车速) 算 (曾用 v_ref 导致'慢'自我延续). --qp-grip-corr: QP 侧也用 GRIP_CORR(s).
        --front-circle: 占用来自线性化轨迹的预测侧偏角 (=实际转向计划), 刹车重量转移加载前轴 -> trail-brake 不被错杀."""
        mp, v, cfg, tire = self.mp, self.veh, self.cfg, self.tire
        Np = mp.Np
        v_fc = np.minimum(Vx_ref[:Np], np.maximum(xi_hat[0, :Np], mp.VX_MIN))
        corr_fc = corr_k if cfg.qp_grip_corr else 1.0
        amax_k = np.maximum(tire.alat_max(v_fc) * corr_fc, 1e-3)
        alat_k = (v_fc ** 2) * np.abs(kappa[:Np])       # 参考曲率占用 (max(κ,预测横摆率) 方案已证伪)
        occ_k = np.clip(alat_k / amax_k, 0.0, 1.0)
        sq = np.sqrt(np.clip(1.0 - occ_k ** 2, 0.0, 1.0))

        uF_k = uR_k = None
        if cfg.front_circle:
            vx_hat = np.maximum(xi_hat[0, :Np], mp.VX_MIN)
            af_hat = u_hat[0, :] - (xi_hat[1, :Np] + v.lf * xi_hat[2, :Np]) / vx_hat
            ar_hat = -(xi_hat[1, :Np] - v.lr * xi_hat[2, :Np]) / vx_hat
            uF_k = np.abs(tire.mf_shape(v.PAC_BF, np.clip(af_hat, -v.AF_PEAK_RAD, v.AF_PEAK_RAD)))
            uR_k = np.abs(tire.mf_shape(v.PAC_BR, np.clip(ar_hat, -v.AR_PEAK_RAD, v.AR_PEAK_RAD)))

        # 驱动上界 = 牵引曲线 ∩ 摩擦圆 (∩ RWD 后轴驱动圆)
        drv_k = self.drive_map.drv_tire(v_fc)
        if cfg.front_circle and self.drive_map.enabled:
            drv_k = np.minimum(drv_k, tire.axle_drive_bound_util(v_fc, uR_k, corr_fc))
        ax_max = np.maximum(np.minimum(drv_k, amax_k * sq), 0.5)

        # 制动下限 = 轴圆 (FC) / 总圆 × 物理帽 × NN 刹车圆修正; 保底 4 m/s²
        circle_k = amax_k * sq
        if cfg.front_circle:
            brk_k = self.brk_eff * tire.axle_brake_bound_util(v_fc, uF_k, uR_k, corr_fc)
            if self.brake_cap.enabled:
                brk_k = np.minimum(brk_k, self.brake_cap.a_brake(v_fc))
        else:
            brk_k = self.brake_cap.brake_limit(v_fc, circle_k, self.brk_eff)
        brk_k = brk_k * self.bc_corr(v_fc, occ_k)
        ax_min = -np.maximum(brk_k, 4.0)
        inv_vx = 1.0 / np.maximum(xi_hat[0, :Np], mp.VX_MIN)
        return ax_max, ax_min, inv_vx

    # ------------------------------------------------------------------
    def solve(self, x0, Vx_ref, kappa, corr=None):
        mp = self.mp
        Np = mp.Np
        x0 = np.asarray(x0, float).flatten()
        Vx_ref = np.asarray(Vx_ref, float).flatten()
        kappa = np.asarray(kappa, float).flatten()
        # 抓地力修正 (与 build_v_profile 一致): 缺省=1 (预热调用/无 corr 时退化为裸模型)
        corr_k = np.ones(Np) if corr is None else np.asarray(corr, float).flatten()[:Np]

        if mp.DELAY_COMP:
            x0 = self._delay_compensate(x0, kappa[0])

        xi_ref = self._build_xi_ref(x0, Vx_ref, kappa)
        xi_hat, u_hat = self._linearization_traj(x0, xi_ref)

        # 沿线性化轨迹逐步 Jacobian -> A,B,D (NN 前馈: 残差值只修 D 项, A/B 保持纯解析)
        for k in range(Np):
            A, B, f0 = self.dyn.fABf(xi_hat[:, k], u_hat[:, k], kappa[k])
            A = np.array(A); B = np.array(B); f0 = np.array(f0).flatten()
            Dk = f0 - A @ xi_hat[:, k] - B @ u_hat[:, k]
            if self.nn is not None:
                Dk[0:3] = Dk[0:3] + mp.dt * self.nn(xi_hat[:, k], u_hat[:, k])
            self.qp.set_dynamics(k, A, B, Dk)

        ax_max, ax_min, inv_vx = self._ax_bounds(Vx_ref, kappa, xi_hat, u_hat, corr_k)
        self.qp.set_bounds(x0, xi_ref, self.u_prev, ax_max, ax_min, inv_vx)

        t0 = time.perf_counter()
        ok, Uv, Xv = self.qp.solve()
        self.solve_ms = (time.perf_counter() - t0) * 1000

        if ok:
            self.solve_ok = True
            delta = float(Uv[0, 0])
            ax = float(Uv[1, 0])
            # 保存状态+控制轨迹 (平移) 做下一步自洽线性化展开点
            self.U_prev_traj = np.column_stack([Uv[:, 1:], Uv[:, -1:]])
            self.X_prev_traj = np.column_stack([Xv[:, 1:], Xv[:, -1:]])
        else:
            self.solve_ok = False
            delta = self.u_prev[0] * 0.9
            ax = 0.0
            self.sol_prev = None

        self.u_prev = np.array([delta, ax])
        return delta, ax

    # ------------------------------------------------------------------
    def prewarm(self, logger=None, n: int = 3):
        """预热求解: cvxpy 首次 .solve() 要做规范化+求解器代码生成 (~0.8s); 在进主循环前用哑数据先解几次,
        把编译开销一次性付掉, 结果丢弃并复位内部状态 -> 接管时第一次求解就是 ~16ms."""
        try:
            x0 = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 哑状态 (直道 20m/s)
            vref = np.full(self.mp.Np + 1, 20.0)
            kap = np.zeros(self.mp.Np + 1)
            for i in range(n):
                t = time.perf_counter()
                self.solve(x0, vref, kap)
                if logger:
                    logger.info(f"[prewarm] solve {i}: {(time.perf_counter()-t)*1000:.0f} ms")
            self.reset()
            if logger:
                logger.info("[prewarm] done, controller state reset")
        except Exception as e:
            if logger:
                logger.warning(f"[prewarm] failed (non-fatal): {e}")
