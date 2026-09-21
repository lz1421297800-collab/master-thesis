# -*- coding: utf-8 -*-
"""MPC 超参数 / 代价权重 / 主循环门限. 数值与 88.7s 黄金配置逐位一致.

MPCParams  : 预测地平线, 约束, 权重, 摩擦圆裕度, 手标牵引曲线 (控制器 + 规划器共用)
LoopParams : 起步/接管/Panic/频率诊断/日志 等主循环门限
"""
from dataclasses import dataclass, field

import numpy as np

from .config import RunConfig


@dataclass
class MPCParams:
    Np: int = 25            # 预测步数
    dt: float = 0.04        # 预测步长 [s] -> 必须=真实循环周期! 实测25Hz=0.04
    AX_MIN: float = -12.0   # 最大制动减速度 [m/s^2] (仅打印/兼容; 实际下界由摩擦圆逐步给)
    VX_MIN: float = 3.0     # 最小纵向速度 [m/s] (防除零)
    EY_MAX: float = 8.0     # 赛道半宽约束 [m]
    N_SUB: int = 4          # 子步 Euler 离散 (治新胎起步抖动: 单步 Euler 在 Vx=15 处特征值 -1.98 发散)

    # 制动效率 = 规划/QP 采用的 轴圆(FC) 或 总圆 刹车利用率份额. 0.42 是无 grip-aware 圆时的稳定值,
    # 爬升协议 (--brk-eff / --brk-eff-sweep) 在 --front-circle 保护下逐档验证 (黄金 0.65).
    BRK_EFF: float = 0.42
    VREF_SCALE: float = 1.0     # CSV 速度 × VREF_SCALE = MPC 目标速度 (上限, 再与 v_grip 取 min)
    DELAY_COMP: bool = True     # 作动延迟补偿 (关闭后退化为奇偶振荡, 保留)
    DELAY_STEPS: float = 1.0    # 来自 cfg.delay_steps (可分数)

    # ===== 代价权重 =====
    W_VX: float = 1.0       # 纵向速度跟踪 [勿抬高: 2026-07-15 已证伪, W_VX=3 摩擦圆占用正反馈锁死]
    W_VY: float = 1.0       # 横向速度
    W_R: float = 15.0       # 横摆角速度跟踪
    W_EPSI: float = 40.0    # 航向误差
    W_EY: float = 120.0     # 横向位置误差
    W_DELTA: float = 0.1    # 转向角大小 (÷gain²)
    W_AX: float = 0.02      # 加速度大小 (0.2→0.02: 旧值在下压力刹车 ax=-22 时是隐藏刹车限制器)
    W_DDELTA: float = 3200.0  # 承重参数! 正则化近胎峰病态 B 矩阵, 不只是平滑 (÷gain²)
    W_DAX: float = 12.0     # 加速度变化率 (30→12: 让入弯刹车不迟到)
    # 软约束松弛权重
    W_AF_SLACK_L1: float = 800.0
    W_AF_SLACK_L2: float = 80.0
    W_EY_SLACK_L1: float = 2000.0
    W_EY_SLACK_L2: float = 200.0
    ALPHA_LIN: float = 0.6  # 自洽线性化: 上一步 QP 解与启发式轨迹的混合权重

    # ===== 抓地力裕度 =====
    # 0.88: 0.90 实测在 s3480 (乐观弯 corr0.93) 前轴过峰推头 (2026-07-07); 无 corr 剖面时回退 0.84.
    GRIP_MARGIN_DEFAULT: float = 0.88
    GRIP_MARGIN_FALLBACK: float = 0.84

    # ===== 手标牵引曲线 (无 drive map 时的全油门净加速 [m/s²]) =====
    DRV_V: np.ndarray = field(default_factory=lambda: np.array([0., 25., 40., 55., 70., 85.]))
    DRV_REF: np.ndarray = field(default_factory=lambda: np.array([16., 16., 9.5, 5.2, 3.0, 2.3]))
    WARM_F: float = 1.0     # 曾是冷胎热身斜坡; 用户 setup 自带热胎器, 恒 1.0 (保留日志列作版本证据)

    # ===== NN 残差前馈 分通道限幅 [Vx, Vy, r] =====
    NN_CLAMP_VEC: np.ndarray = field(default_factory=lambda: np.array([3.0, 1.0, 1.0]))

    # ===== OSQP =====
    OSQP_MAX_ITER: int = 8000
    OSQP_EPS: float = 1e-4  # 配合 s 归零后 ||z||~40, 约束残差上界 ~5e-3 (eps=1e-3 + s~4000 是 bang-bang 根因)

    @property
    def horizon(self) -> float:
        return self.Np * self.dt

    @classmethod
    def from_config(cls, cfg: RunConfig, verbose: bool = True) -> 'MPCParams':
        p = cls()
        p.DELAY_STEPS = cfg.delay_steps
        # ÷gain²: --steer-gain 后 δ 数值放大 gain 倍, 除以 gain² 保持同一物理转向动作的惩罚不变
        p.W_DELTA = 0.1 / cfg.steer_gain ** 2
        p.W_DDELTA = 3200.0 / cfg.steer_gain ** 2
        if cfg.brk_eff_override is not None:
            p.BRK_EFF = float(np.clip(cfg.brk_eff_override, 0.20, 1.00))
            if verbose:
                print(f"[EXPERIMENT] BRK_EFF overridden -> {p.BRK_EFF:.2f}")
        if verbose:
            print(f"MPC: Np={p.Np}, dt={p.dt}s, horizon={p.horizon:.1f}s")
            print(f"VREF_SCALE={p.VREF_SCALE} (CSV速度的{p.VREF_SCALE*100:.0f}%)")
            print(f"权重: W_VX={p.W_VX} W_VY={p.W_VY} W_R={p.W_R} W_EPSI={p.W_EPSI} W_EY={p.W_EY}")
        return p


@dataclass
class LoopParams:
    # VREF_SCALE 自动扫描 (采数模式)
    VREF_SWEEP: bool = False
    VREF_START: float = 0.75
    VREF_STEP: float = 0.05
    VREF_MAX: float = 0.95
    # Panic 保护 (隔离测试放宽: 6->12, 25->60)
    PANIC_E: float = 12.0
    PANIC_DPHI: float = float(np.deg2rad(60.0))
    # 低速运动学 / 高速 MPC 分段 (带滞回)
    V_MPC_ON: float = 15.0
    V_MPC_OFF: float = 5.0
    MPC_ALIGN_EPSI_ON: float = float(np.deg2rad(8.0))
    MPC_ALIGN_EY_ON: float = 1.0
    MPC_TRANS_STEPS: int = 30       # 切入后平滑过渡步数 (黄金验证值)
    # 频率诊断
    DT_MISMATCH_TOL: float = 0.20
    FREQ_WARN_EVERY: int = 40
    VERBOSE_EVERY: int = 20
    # NN 残差在线采集
    COLLECT_RESIDUAL: bool = True
    # 弯中偏差油门保护阈值 (05l 教训: 向外漂时先收油)
    AX_CUT_EPSI: float = float(np.deg2rad(5.0))
    AX_CUT_EY: float = 0.4
    AX_BRAKE_EPSI: float = float(np.deg2rad(9.0))
    AX_BRAKE_EY: float = 0.9
    DEFAULT_LAPS: int = 5

    def max_laps(self, cfg: RunConfig, mp: MPCParams) -> int:
        if cfg.max_laps is not None:
            return cfg.max_laps
        if self.VREF_SWEEP:
            return int(np.ceil((self.VREF_MAX - self.VREF_START) / self.VREF_STEP)) + 3
        if cfg.brk_sweep:
            return int(np.ceil((cfg.brk_sweep_max - mp.BRK_EFF) / cfg.brk_sweep_step)) + 3
        return self.DEFAULT_LAPS
