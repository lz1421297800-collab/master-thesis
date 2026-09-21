# -*- coding: utf-8 -*-
"""规划侧: 抓地力可行弯速 v_grip -> 前向/后向扫掠可行速度剖面 v_prof -> 地平线参考序列.

规划与跟踪同一车辆模型 (论文教训): v_grip 按模型极限封顶; v_prof 逐米用摩擦圆限制的驱动/刹车递推,
得到处处自洽可行的剖面; get_mpc_reference 沿参考速度累积弧长, 使 κ/v/corr 序列与预测 s_k 严格对齐.
"""
import time

import numpy as np

from .config import RunConfig
from .learned_maps import BrakeCap, BrakeCircleCorr, DriveMap, GripCorr
from .params import MPCParams
from .tire import TireLimits
from .track import Track
from .vehicle import VehicleParams

# ===== 每弯局部裕度表 (手工兜底; 仅默认 grip_corr.npz 时生效) =====
# 弯A (1000-1230): 高速入弯手工兜底 (稳定基线一直有).
# s2094 弯: 前轴极限弯, 参考速度处 af 即钉 6.4-6.8° 过峰; 压 5% 使 af 需求回到峰下.
LOCAL_MARGIN_ZONES_DEFAULT = [
    (1000, 1230, 0.88),
    (2060, 2160, 0.95),
]


class SpeedPlanner:
    def __init__(self, track: Track, veh: VehicleParams, mp: MPCParams, tire: TireLimits,
                 grip: GripCorr, brake_cap: BrakeCap, bc_corr: BrakeCircleCorr, drive_map: DriveMap,
                 cfg: RunConfig, verbose: bool = True):
        self.track, self.veh, self.mp, self.tire = track, veh, mp, tire
        self.grip, self.brake_cap, self.bc_corr, self.drive_map = grip, brake_cap, bc_corr, drive_map
        self.cfg = cfg
        self.brk_eff = mp.BRK_EFF          # 爬升协议会改它 (BrkEffSweep.set_brk_eff)
        self.vref_scale = mp.VREF_SCALE
        self._kabs = np.maximum(np.abs(track.kappa), 1e-5)
        self.v_grip = self._compute_v_grip(verbose)
        self._apply_local_zones(verbose)
        if verbose:
            n_bind = int(np.sum(self.v_grip < track.v_ai * 0.90))
            print(f"v_grip 预计算完成: 在 VREF=0.90 下有 {n_bind} 个点被抓地力封顶 (CSV速度过快处)")
        if cfg.ref_ss and verbose:
            self._print_ref_ss_example()
        t0 = time.perf_counter()
        self.v_prof = self.build_v_profile(self.vref_scale, self.brk_eff)
        if verbose:
            print(f"v_prof 可行速度剖面: VREF={self.vref_scale}, 隐含圈时 {self.implied_lap_time():.1f}s, "
                  f"构建 {(time.perf_counter()-t0)*1000:.0f}ms")

    # ------------------------------------------------------------------
    def _compute_v_grip(self, verbose):
        """不动点迭代 v = sqrt(margin·a_cap(v)·corr/|κ|). --axle-vgrip: 逐轴稳态上限."""
        cap = self.tire.alat_axle if self.cfg.axle_vgrip else self.tire.alat_max
        v_grip = np.full(self.track.n_pts, 95.0)
        for _ in range(60):
            v_grip = 0.5 * v_grip + 0.5 * np.minimum(
                np.sqrt(self.grip.margin * cap(v_grip) * self.grip.corr / self._kabs), 95.0)
        if self.cfg.axle_vgrip and verbose:
            print(f"[axle-vgrip] 弯速封顶=逐轴稳态上限 (v30: 总轴{float(self.tire.alat_max(30.)):.1f} -> "
                  f"逐轴{float(self.tire.alat_axle(30.)):.1f} m/s²)")
        return v_grip

    def _apply_local_zones(self, verbose):
        zones = list(LOCAL_MARGIN_ZONES_DEFAULT)
        if not self.grip.is_default_file:
            # 重训剖面在这些弯已有近极限证据, 叠加手工 zone 会重复扣账 (~0.9s)
            zones = []
            if verbose:
                print(f"[grip-corr] 使用重训剖面 {self.grip.file} -> 手工 LOCAL_MARGIN_ZONES 停用")
        for lo, hi, sc in zones:
            self.v_grip[lo:hi] *= sc
        self.local_zones = zones
        if verbose:
            print(f"局部裕度生效: {zones or '无 (全部交给 NN corr)'}")

    def _print_ref_ss_example(self):
        v, t = self.veh, self.tire
        v_ex, k_ex = 35.0, 0.015
        rr_ex = v_ex * k_ex
        Fzr_ex = v.FZR0 + v.CZR * v_ex ** 2
        mur_ex = t.mu_r(Fzr_ex)
        u_ex = min(v.m * v_ex ** 2 * k_ex * v.lf / (v.L * mur_ex * Fzr_ex), 0.95)
        ar_ex = float(t.ar_ss_from_util(u_ex))
        vy_ex = v.lr * rr_ex - ar_ex * v_ex
        print(f"[ref-ss] 稳态自洽参考启用: 例 v={v_ex:.0f},κ={k_ex} -> util={u_ex:.2f}, "
              f"ar_ss={np.degrees(ar_ex):.2f}°, Vy_ref={vy_ex:+.2f} m/s, "
              f"e_psi_ref={np.degrees(-np.arctan(vy_ex/v_ex)):+.2f}°")

    # ------------------------------------------------------------------
    def build_v_profile(self, vref_scale: float, brk_eff: float) -> np.ndarray:
        """离线前向-后向扫掠 (3 轮): 前向 驱动=牵引曲线 ∩ 摩擦圆纵向余量; 后向 刹车=轴圆/总圆 × 修正."""
        tr, veh, tire, cfg = self.track, self.veh, self.tire, self.cfg
        grip, kabs = self.grip.corr, self._kabs
        n = tr.n_pts
        vcap = np.minimum(tr.v_ai * vref_scale, self.v_grip * self.mp.WARM_F)
        v = vcap.copy()
        use_fc, use_dm = cfg.front_circle, self.drive_map.enabled
        for _sweep in range(3):
            for i in range(n):            # 前向: 驱动受功率+摩擦圆限制 (a_max 含 corr)
                j = (i + 1) % n
                am = float(tire.alat_max(v[i])) * grip[i]
                fr = max(1.0 - (v[i] * v[i] * kabs[i] / am) ** 2, 0.0)
                drv = float(self.drive_map.drv_net(v[i]))
                if use_fc and use_dm:
                    # RWD 后轴驱动圆 (轮胎级) 减风阻得净值, 与 QP 侧一致
                    drv = min(drv, tire.axle_drive_bound_dem(v[i], v[i] * v[i] * kabs[i], grip[i])
                              - veh.CXW * v[i] * v[i] / veh.m)
                a = min(am * np.sqrt(fr), drv)
                v[j] = min(vcap[j], np.sqrt(v[i] * v[i] + 2.0 * max(a, 0.1)))
            for i in range(n - 1, -1, -1):  # 后向: 刹车上限 (FC=轴圆 / 旧=BRK_EFF·总圆)
                j = (i + 1) % n
                am = float(tire.alat_max(v[j])) * grip[j]
                occ = min(v[j] * v[j] * kabs[j] / am, 1.0)   # 横向占用 ay/a_max (含corr)
                if use_fc:
                    b = brk_eff * tire.axle_brake_bound_dem(v[j], v[j] * v[j] * kabs[j], grip[j])
                    if self.brake_cap.enabled:
                        b = min(b, float(self.brake_cap.a_brake(v[j])))   # 物理刹车帽独立生效
                else:
                    circle = am * np.sqrt(max(1.0 - occ ** 2, 0.0))
                    b = float(self.brake_cap.brake_limit(v[j], circle, brk_eff))
                b = max(b * float(self.bc_corr(v[j], occ)), 3.0)   # NN 刹车圆修正 (默认恒1)
                v[i] = min(v[i], np.sqrt(v[j] * v[j] + 2.0 * b))
        return v

    def implied_lap_time(self, v_prof=None) -> float:
        v = self.v_prof if v_prof is None else v_prof
        return float(np.sum(1.0 / np.maximum(v, 1.0)))

    # ------------------------------------------------------------------
    def get_mpc_reference(self, s0: float):
        """从分数弧长 s0 起, 沿 v_prof 按参考速度逐点累积弧长取 Np+1 个点 (亚网格线性插值).
        返回 (Vx_ref, kappa, corr), 三者同点采样, 与 MPC 预测的 s_k 严格对齐."""
        Np, dt, VX_MIN = self.mp.Np, self.mp.dt, self.mp.VX_MIN
        tr = self.track
        Vx_ref = np.zeros(Np + 1)
        kappa = np.zeros(Np + 1)
        corr = np.zeros(Np + 1)
        s_off = 0.0
        for k in range(Np + 1):
            pos = s0 + s_off
            vk = tr.ref_lerp(self.v_prof, pos)
            Vx_ref[k] = vk
            kappa[k] = tr.ref_lerp(tr.kappa, pos)
            corr[k] = tr.ref_lerp(self.grip.corr, pos)
            s_off += max(vk, VX_MIN) * dt
        return Vx_ref, kappa, corr
