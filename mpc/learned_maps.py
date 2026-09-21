# -*- coding: utf-8 -*-
"""从日志学习的四张查表 (全部 opt-in, 文件缺失时按各自安全回退):

- GripCorr        : GRIP_CORR(s) 抓地力修正剖面 (learn_grip_profile.py -> grip_corr.npz) + GRIP_MARGIN 回退逻辑
- BrakeCap        : a_brake_phys(v) 刹车物理上限 (learn_brake_profile.py -> brake_cap.npz) + brake_limit 混合
- BrakeCircleCorr : corr(v, occ) 刹车圆纵向修正 (learn_brake_circle_corr.py -> brake_circle_corr.npz)
- DriveMap        : a_tire(v, acc) 驱动交付表 (learn_drive_map.py -> drive_map.npz)
"""
import numpy as np

from .config import RunConfig
from .params import MPCParams


class GripCorr:
    """GRIP_CORR(s): corr=1+ρ(s), 只在有近极限证据的弯 ≠1. 与 GRIP_MARGIN 捆绑:
    文件不可用 -> corr=1 且 margin 回退 0.84 (曾因静默回退 corr=1 却保留 0.88 -> 弯1 推头事故);
    显式 'none'/'off' -> corr=1 但不恐慌回退 (重基准化模式, 裕度由 --grip-margin 显式给)."""

    def __init__(self, cfg: RunConfig, mp: MPCParams, n_pts: int, verbose: bool = True):
        self.file = cfg.grip_corr_file
        self.is_default_file = (cfg.grip_corr_file == 'grip_corr.npz')
        self.margin = mp.GRIP_MARGIN_DEFAULT
        self.explicit_off = self.file.lower() in ('none', 'off')
        if self.explicit_off:
            self.corr = np.ones(n_pts)
            if verbose:
                print("[grip-corr] 显式停用 (corr=1): 重基准化模式, 配合 --mu-scale-f/r + --grip-margin")
        else:
            try:
                gc = np.load(self.file)['corr']
                if len(gc) >= n_pts:
                    self.corr = gc[:n_pts].copy()
                else:
                    self.corr = np.ones(n_pts); self.corr[:len(gc)] = gc
                if verbose:
                    print(f"{self.file} 已加载: corr {self.corr.min():.3f}~{self.corr.max():.3f} "
                          f"(npz len={len(gc)}, N_PTS={n_pts})")
            except Exception as e:
                self.corr = np.ones(n_pts)
                self.margin = mp.GRIP_MARGIN_FALLBACK   # 无 corr 时回到已验证安全配置!
                if verbose:
                    print(f"!! {self.file} 不可用({e}) -> corr=1 且 GRIP_MARGIN 回退 {self.margin:.2f} (安全配置)")
        # --grip-margin 显式覆盖 (在回退逻辑之后, 显式值永远赢)
        if cfg.grip_margin is not None:
            self.margin = cfg.grip_margin
            if verbose:
                print(f"[EXPERIMENT] GRIP_MARGIN overridden -> {self.margin:.2f}")

    def __getitem__(self, i):
        return self.corr[i]

    def __len__(self):
        return len(self.corr)


class BrakeCap:
    """a_brake_phys(v): 车的绝对刹车上限 (实测 ~26@中速, ~20@高速), a_max 缩放抓不住.
    实际刹车 = min(a_brake_phys(v), a_max·√(1-occ²)). 关闭时 brake_limit 退化为 BRK_EFF·circle."""

    def __init__(self, cfg: RunConfig, mp: MPCParams, verbose: bool = True):
        self.requested = cfg.use_brake_cap
        self.file = cfg.brake_cap_file
        self.blend = cfg.brake_cap_blend
        self.enabled = False
        self.V = np.array([0., 100.]); self.A = np.array([99., 99.])
        if self.requested:
            try:
                bc = np.load(self.file)
                self.V = bc['v']; self.A = bc['a_brake']; self.enabled = True
                if verbose:
                    print(f"{self.file} 已加载: a_brake_phys(v) {self.A.min():.1f}~{self.A.max():.1f} m/s^2 "
                          f"(v50={np.interp(50, self.V, self.A):.0f} v80={np.interp(80, self.V, self.A):.0f})")
            except Exception as e:
                if verbose:
                    print(f"{self.file} 不可用({e}) -> 回退恒定 BRK_EFF={mp.BRK_EFF}")
        elif verbose:
            print(f"brake_cap disabled -> 回退恒定 BRK_EFF={mp.BRK_EFF}")

    def a_brake(self, v):
        """刹车物理上限 [m/s²] (速度相关)."""
        return np.interp(v, self.V, self.A)

    def brake_limit(self, v, circle, brk_eff):
        """v_prof 与 QP 共用的 (总圆模式) 刹车上限."""
        legacy = brk_eff * circle
        if not self.enabled:
            return legacy
        learned = np.minimum(self.a_brake(v), circle)
        return (1.0 - self.blend) * legacy + self.blend * learned


class BrakeCircleCorr:
    """corr_bc(v, occ) = 实测减速/模型减速 (重刹样本, 密度门控: 无证据 bin corr=1). 只乘刹车上界."""

    def __init__(self, cfg: RunConfig, verbose: bool = True):
        self.enabled = cfg.use_bc_corr
        self.file = cfg.bc_corr_file
        if self.enabled:
            try:
                d = np.load(self.file)
                self.V, self.OCC, self.C = d['v'], d['occ'], d['corr']
                if verbose:
                    print(f"{self.file} 已加载: corr {self.C.min():.3f}~{self.C.max():.3f} "
                          f"(v网格{len(self.V)} x occ网格{len(self.OCC)})")
            except Exception as e:
                self.enabled = False
                if verbose:
                    print(f"!! {self.file} 不可用({e}) -> 刹车圆修正关闭 (corr=1)")

    def __call__(self, v, occ):
        """corr(v, occ) 双线性插值, 网格外夹紧. 关闭时恒 1."""
        v = np.asarray(v, float)
        if not self.enabled:
            return np.ones_like(v)
        occ = np.asarray(occ, float)
        V, OCC, C = self.V, self.OCC, self.C
        vv = np.clip(v, V[0], V[-1])
        oo = np.clip(occ, OCC[0], OCC[-1])
        iv = np.clip(np.searchsorted(V, vv) - 1, 0, len(V) - 2)
        io = np.clip(np.searchsorted(OCC, oo) - 1, 0, len(OCC) - 2)
        wv = (vv - V[iv]) / (V[iv + 1] - V[iv])
        wo = (oo - OCC[io]) / (OCC[io + 1] - OCC[io])
        return (C[iv, io] * (1 - wv) * (1 - wo) + C[iv + 1, io] * wv * (1 - wo)
                + C[iv, io + 1] * (1 - wv) * wo + C[iv + 1, io + 1] * wv * wo)


class DriveMap:
    """驱动交付表 a_tire(v, acc). 三处消费: (1) 规划前向扫掠用净全油 a_net_full;
    (2) QP 驱动上界用轮胎级全油 a_tire_full; (3) pedal_from_ax 查表逆映射. 关闭时退回手标 DRV_REF."""

    def __init__(self, cfg: RunConfig, mp: MPCParams, verbose: bool = True):
        self.enabled = cfg.use_drive_map
        self.file = cfg.drive_map_file
        self.DRV_V, self.DRV_REF = mp.DRV_V, mp.DRV_REF
        if self.enabled:
            try:
                d = np.load(self.file)
                self.V, self.ACC, self.TAB = d['v'], d['acc'], d['a_tire']
                self.TIRE_FULL = np.maximum(d['a_tire_full'], 0.5)   # QP 上界下限防零
                self.NET_FULL = np.maximum(d['a_net_full'], 0.1)     # 规划净值下限防负(顶速)
                if verbose:
                    print(f"{self.file} 已加载: 净全油 {self.NET_FULL.min():.1f}~{self.NET_FULL.max():.1f}, "
                          f"轮胎全油 {self.TIRE_FULL.min():.1f}~{self.TIRE_FULL.max():.1f} "
                          f"(v50净{np.interp(50, self.V, self.NET_FULL):.1f} vs 旧DRV{np.interp(50, self.DRV_V, self.DRV_REF):.1f})")
            except Exception as e:
                self.enabled = False
                if verbose:
                    print(f"!! {self.file} 不可用({e}) -> 退回手标 DRV_REF")

    def drv_net(self, v):
        """规划用: 全油门净加速 [m/s²] (前向扫掠直接积分, 不再减风阻)."""
        if self.enabled:
            return np.interp(v, self.V, self.NET_FULL)
        return np.interp(v, self.DRV_V, self.DRV_REF)

    def drv_tire(self, v):
        """QP 驱动上界: 全油门轮胎级加速度 [m/s²]. 旧行为: 净值表当轮胎级用 (保留基线)."""
        if self.enabled:
            return np.interp(v, self.V, self.TIRE_FULL)
        return np.interp(v, self.DRV_V, self.DRV_REF)

    def row(self, v):
        """交付表在速度 v 处的行 (沿 v 线性), 供踏板逆映射."""
        i = int(np.clip(np.searchsorted(self.V, v) - 1, 0, len(self.V) - 2))
        w = float(np.clip((v - self.V[i]) / (self.V[i + 1] - self.V[i]), 0.0, 1.0))
        return (1.0 - w) * self.TAB[i] + w * self.TAB[i + 1]
