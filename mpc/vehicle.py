# -*- coding: utf-8 -*-
"""F2004 车辆 / 轮胎 / 转向标定参数 (Hao et al. Eq.8, Eq.12-14 所需的全部物理常数).

VehicleParams.from_config(cfg) 把命令行标定缩放 (--iz / --steer-gain / --mu-scale / --pac-b-scale /
--brake-bias / --fy-combined / --model-rx) 一次性折进参数, 下游模块只读字段, 不再重复判断.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import IZ_DEFAULT, RunConfig


@dataclass
class VehicleParams:
    # ===== 底盘 =====
    m: float = 664.0          # 总重 (含车手+油) [kg]
    L: float = 3.050          # 轴距 [m]
    lf: float = 1.700         # CoG->前轴 [m]  (辨识: 实测前轴占比44.3%)
    lr: float = 1.350         # CoG->后轴 [m]  (lf+lr = L)
    Cf: float = 109300.0      # 前轴"小侧偏"侧偏刚度 [N/rad] (名义静载荷下; 仅打印/对照)
    Cr: float = 139100.0      # 后轴"小侧偏"侧偏刚度 [N/rad]
    # AC 惯量盒 car.ini INERTIA=(1.8,0.55,3.6) -> I_yaw = m/12·(1.8²+3.6²) = m·1.35.
    # 旧值 817 = 605(干重)·1.35, 但模型 m=664(满油) -> 自洽值 898. 闭环验证 (2026-07-15):
    # 离线瞬态残差 RMS 4.12->3.85, max|e_y| 0.66->0.39. 拟合最优 ~1045 是轮胎松弛的唯象补偿, 勿采用.
    Iz: float = IZ_DEFAULT    # 横摆转动惯量 [kg*m^2]
    g_acc: float = 9.81
    THETA: float = 0.0        # 赛道纵向坡度 [rad] (Barcelona 近似平地)
    hc: float = 0.252         # CoG 高度 [m] (辨识值, 配 m=664; 实测 m*hc=167.6)
    MU: float = 1.75          # 轮胎峰值侧向摩擦系数 (仅打印; 动力学用 μ(载荷,速度) 曲面)
    CXW: float = 0.757        # 风阻系数 [kg/m]  Fxw = CXW*Vx^2 (aero.ini 折算: CdA=1.237)
    CZF: float = 0.92         # 前轴气动下压力 [kg/m] (aero.ini 逐翼力矩合算)
    CZR: float = 1.16         # 后轴气动下压力 [kg/m]

    # ===== 纵向损耗 (tyres.ini / engine.ini / drivetrain.ini 辨识值, 2026-07-13) =====
    # 滚动阻力四胎合计 Rx = RR0_TOT + RR1_TOT·v² [N]; v=60 -> 27N = 0.04 m/s². --model-rx 才进动力学.
    RR0_TOT: float = 20.0         # [N]
    RR1_TOT: float = 1.886e-3     # [N/(m/s)²]
    R_WHEEL: float = 0.330        # [m] tyres.ini RADIUS
    GEAR_RATIOS: List[float] = field(default_factory=lambda: [2.8461, 2.0625, 1.8235, 1.6111, 1.4736, 1.3200, 1.2083])
    FINAL_DRIVE: float = 6.2
    COAST_T_REF: float = 100.0    # engine.ini [COAST_REF]
    COAST_RPM_REF: float = 19000.0
    MODEL_RX: bool = False        # --model-rx: 滚动阻力进纵向动力学

    # ===== 转向 =====
    STEER_LOCK: float = 180.0
    STEER_RATIO: float = 15.0     # car.ini 标称 (实际有效比 ~7.1, 由 STEER_GAIN 修正)
    STEER_GAIN: float = 1.0       # --steer-gain: 实际前轮角 = gain × 模型 δ (辨识 2.11)
    STEER_CMD_SIGN: float = -1.0  # AC 转向符号: 如果车反向转弯就改成 +1

    # ===== Pacejka MF 轮胎 (tyres.ini Soft 胎逐点拟合 @FZ0, 2026-07-04) =====
    # Fy = μ·Fz·sin(C·atan((1-E)·B·α + E·atan(B·α)));  前 B=21.22 后 B=23.05, C=1.5, E=0.515; 峰值 6.3°/5.8°.
    PAC_C: float = 1.5
    PAC_E: float = 0.515
    PAC_BF: float = 21.22
    PAC_BR: float = 23.05
    # μ(载荷,速度) 曲面: μ = DY_REF·(Fz_单胎/FZ0)^(LS_EXPY-1)·(1-SS·Vx)  (tyres.ini V10 结构)
    DY_REF_F: float = 1.80
    DY_REF_R: float = 1.80
    FZ0T_F: float = 3451.0        # 单胎参考载荷 [N] (tyres.ini FZ0)
    FZ0T_R: float = 3995.0
    LS_EXPY_F: float = 0.80
    LS_EXPY_R: float = 0.70
    # SPEED_SENSITIVITY 作用于接触面滑移速度而非车速, 按 '(1-SS·Vx)' 建模是误读 -> 置 0 (2026-07-04)
    SS_F: float = 0.0
    SS_R: float = 0.0
    PAC_B_SCALE_F: float = 1.0    # --pac-b-scale-f (峰值角 ∝ 1/B, AF_CAP 同步放宽)
    PAC_B_SCALE_R: float = 1.0
    MU_SCALE_F: float = 1.0
    MU_SCALE_R: float = 1.0

    # ===== 刹车分配 / 联合滑移 =====
    FRONT_BRK_SHARE: float = 0.56   # --brake-bias: 前轴刹车力占比
    FY_COMBINED: bool = False       # --fy-combined: 后轴 Fyr *= sqrt(1-occ_r²)

    # ===== 派生量 (from_config 里填) =====
    FZF0: float = 0.0             # 静态前轴总载荷 [N]
    FZR0: float = 0.0
    DELTA_MAX: float = 0.0        # 真实满舵 [rad] (gain=1 时 ±12°)
    AF_PEAK_RAD: float = 0.0      # 前胎峰值侧偏角 (tyres.ini FRICTION_LIMIT_ANGLE 6.3°; ∝1/B)
    AR_PEAK_RAD: float = 0.0      # 后胎峰值侧偏角 5.8°
    AF_CAP: float = 0.0           # QP 前轮侧偏软约束: 峰下 0.5° 余量 (6.0°)
    AF_PROTECT: float = 0.0       # 主循环前轴饱和油门保护: 峰下 0.6° (5.7°)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: RunConfig, verbose: bool = True) -> 'VehicleParams':
        v = cls()
        if cfg.iz is not None:
            v.Iz = cfg.iz
            if v.Iz != IZ_DEFAULT and verbose:
                print(f"[EXPERIMENT] Iz overridden -> {v.Iz:.1f} (默认 {IZ_DEFAULT:.0f} = 惯量盒×满油664; 817 = 旧干重605基准)")
        v.MODEL_RX = cfg.model_rx
        v.STEER_GAIN = cfg.steer_gain
        v.FRONT_BRK_SHARE = cfg.front_brk_share
        v.FY_COMBINED = cfg.fy_combined

        # --pac-b-scale[-f/-r]: 刚度因子 B 缩放, 小滑移力 ∝ B 等比降, 峰值力不变, 峰值角 ∝ 1/B 右移
        v.PAC_B_SCALE_F, v.PAC_B_SCALE_R = cfg.pac_b_scale_f, cfg.pac_b_scale_r
        v.PAC_BF *= v.PAC_B_SCALE_F
        v.PAC_BR *= v.PAC_B_SCALE_R
        if (v.PAC_B_SCALE_F != 1.0 or v.PAC_B_SCALE_R != 1.0) and verbose:
            print(f"[pac-b] B 缩放 f{v.PAC_B_SCALE_F:.3f}/r{v.PAC_B_SCALE_R:.3f}: BF={v.PAC_BF:.2f} BR={v.PAC_BR:.2f}, "
                  f"峰值角 {6.3/v.PAC_B_SCALE_F:.1f}°/{5.8/v.PAC_B_SCALE_R:.1f}°, AF_CAP 同步放宽")
        # --mu-scale-f/r: 乘在 DY_REF (MF 的 D=μFz) 上 -> 等比缩放全滑移段力
        v.MU_SCALE_F, v.MU_SCALE_R = cfg.mu_scale_f, cfg.mu_scale_r
        v.DY_REF_F *= v.MU_SCALE_F
        v.DY_REF_R *= v.MU_SCALE_R
        if (v.MU_SCALE_F != 1.0 or v.MU_SCALE_R != 1.0) and verbose:
            print(f"[mu-scale] DY_REF_F={v.DY_REF_F:.3f} DY_REF_R={v.DY_REF_R:.3f} "
                  f"(scale f{v.MU_SCALE_F:.3f}/r{v.MU_SCALE_R:.3f}) "
                  f"!! grip_corr 是旧μ下训练的, 稳定后需重训再组合")

        v.FZF0 = v.m * v.g_acc * v.lr / v.L
        v.FZR0 = v.m * v.g_acc * v.lf / v.L
        v.DELTA_MAX = np.deg2rad(v.STEER_LOCK / v.STEER_RATIO) * v.STEER_GAIN
        v.AF_PEAK_RAD = np.deg2rad(6.3) / v.PAC_B_SCALE_F
        v.AR_PEAK_RAD = np.deg2rad(5.8) / v.PAC_B_SCALE_R
        v.AF_CAP = np.deg2rad(6.0) / v.PAC_B_SCALE_F
        v.AF_PROTECT = np.deg2rad(5.7) / v.PAC_B_SCALE_F
        return v

    # ------------------------------------------------------------------
    def a_coast(self, v):
        """在挡滑行的发动机制动减速【上界】[m/s²] (诊断/论文用, 不进控制路径 — 该效应已被
        drive map 交付表的低油门列覆盖). 挡位取 rpm≤18000 的最低挡 = 最大发动机制动."""
        v = np.atleast_1d(np.asarray(v, float))
        out = np.zeros_like(v)
        for i, vv in enumerate(v):
            w_rpm = max(vv, 0.1) / (2 * np.pi * self.R_WHEEL) * 60.0
            for ig in self.GEAR_RATIOS:                      # 1挡->7挡 (传动比降序)
                rpm = w_rpm * ig * self.FINAL_DRIVE
                if rpm <= 18000.0:
                    out[i] = self.COAST_T_REF * (rpm / self.COAST_RPM_REF) * ig * self.FINAL_DRIVE / (self.R_WHEEL * self.m)
                    break
        return out

    def describe(self):
        rx60 = self.RR0_TOT + self.RR1_TOT * 3600
        print(f"F2004: m={self.m}, lf={self.lf:.3f}, lr={self.lr:.3f}, L={self.lf+self.lr:.3f}")
        print(f"Iz={self.Iz:.1f}, Cf={self.Cf:.0f}, Cr={self.Cr:.0f}")
        print(f"hc={self.hc}, MU={self.MU}, CXW={self.CXW}, CZF={self.CZF}, CZR={self.CZR}")
        print(f"静态轴载: FZF0={self.FZF0:.0f}N, FZR0={self.FZR0:.0f}N (前轴占比 {self.FZF0/(self.FZF0+self.FZR0)*100:.0f}%)")
        print(f"delta_max = +-{np.degrees(self.DELTA_MAX):.1f} deg")
        print(f"纵向损耗辨识: Rx(60)={rx60:.0f}N({rx60/self.m:.3f}m/s², "
              f"{'已进模型 --model-rx' if self.MODEL_RX else '未进模型'}); "
              f"发动机制动 a_coast(40/60)={float(self.a_coast(40.)):.1f}/{float(self.a_coast(60.)):.1f} m/s² (由drive map覆盖)")
        print(f"[TIRE-CHECK] tyres.ini MF+LS+SS  B={self.PAC_BF}/{self.PAC_BR} C={self.PAC_C} E={self.PAC_E} "
              f"DY_REF={self.DY_REF_F}/{self.DY_REF_R} LS_EXPY={self.LS_EXPY_F}/{self.LS_EXPY_R} SS={self.SS_F}/{self.SS_R} "
              f"fy_combined={'ON(rear)' if self.FY_COMBINED else 'off'}")
