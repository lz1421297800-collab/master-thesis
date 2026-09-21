# -*- coding: utf-8 -*-
"""参考线 (racing line) 加载 / 1m 均匀重采样 / 曲率 / 最近点搜索 / 亚网格 Frenet 误差."""
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Track:
    """均匀 1m 网格的参考线: xy / yaw / kappa / AI 参考速度 + KD-tree.

    弧长 = 索引 (米), 环形. kappa 符号约定: 与 Frenet 方程 (de_psi/de_y, e_y 左正) 手系自洽,
    在源头翻一次符号, 下游 r_ref/de_psi/前馈/热启动 全部自洽 (直道 kappa~0 不暴露, 进弯立刻发散).
    """

    def __init__(self, csv_path: str, track_length: float,
                 kappa_sigma: float = 4.0, v_sigma_pre: float = 5.0, v_sigma_post: float = 2.0):
        self.csv_path = csv_path
        self.length = float(track_length)
        df = pd.read_csv(csv_path)
        df = df.sort_values('lap_dist').drop_duplicates('lap_dist').reset_index(drop=True)
        raw_xy = df[['pos_x', 'pos_y']].values
        raw_v = df['speed'].values
        raw_s = df['lap_dist'].values

        self.n_pts = int(self.length)
        self.s = np.linspace(0, self.length, self.n_pts, endpoint=False)
        self.xy = np.column_stack([
            interp1d(raw_s, raw_xy[:, 0], fill_value='extrapolate')(self.s),
            interp1d(raw_s, raw_xy[:, 1], fill_value='extrapolate')(self.s)])
        v = np.maximum(gaussian_filter1d(interp1d(raw_s, raw_v, fill_value='extrapolate')(self.s), sigma=v_sigma_pre), 5.0)

        sx = UnivariateSpline(self.s, self.xy[:, 0], k=4, s=0)
        sy = UnivariateSpline(self.s, self.xy[:, 1], k=4, s=0)
        dx, dy = sx.derivative(1)(self.s), sy.derivative(1)(self.s)
        ddx, ddy = sx.derivative(2)(self.s), sy.derivative(2)(self.s)
        self.yaw = np.arctan2(dy, dx)
        # σ8→σ4 (2026-07-04g): σ8 把急弯 κ 峰抹低 6-19%, 导致 v_grip 虚高 + r_ref 前馈不足 -> 急弯推头.
        self.kappa = gaussian_filter1d((dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5, sigma=kappa_sigma)
        self.v_ai = np.maximum(gaussian_filter1d(v, sigma=v_sigma_post), 5.0)   # AI 原速 (未封顶)
        self.tree = cKDTree(self.xy)

    # ------------------------------------------------------------------
    def describe(self, px=None, py=None):
        if px is not None:
            d_chk, _ = self.tree.query([px, py])
            print(f"对齐检查: 距参考线 = {d_chk:.2f} m")
        print(f"v_ref: {self.v_ai.min():.1f}~{self.v_ai.max():.1f} m/s, {self.n_pts} pts")

    # ------------------------------------------------------------------
    def find_nearest(self, px, py, hint_idx=None) -> int:
        """最近参考点索引; 有上一步提示时只在 ±120m 窗内搜, 失配 (>10m) 才回退 KD-tree 全局搜."""
        if hint_idx is not None:
            lo = max(hint_idx - 120, 0)
            hi = min(hint_idx + 120, self.n_pts)
            d2 = (self.xy[lo:hi, 0] - px) ** 2 + (self.xy[lo:hi, 1] - py) ** 2
            best = lo + int(np.argmin(d2))
            if np.sqrt(d2.min()) < 10.0:
                return best
        _, best = self.tree.query([px, py])
        return int(best)

    def frenet_errors(self, px, py, yaw, idx):
        """亚网格 Frenet 误差: 沿最近点切向投影得分数弧长偏移 u, 参考点/航向线性插值.

        旧版直接吸附整数米网格点: e_psi 带 frac(Vx*dt)*25Hz 锯齿直接进 W_EPSI 反馈 -> 弯道方向盘
        高频抖动 (主频随速度移动 1.4~9.6Hz). 返回 (e_y, e_psi, s_frac); s_frac = idx+u 供参考序列插值."""
        t_hat_x, t_hat_y = np.cos(self.yaw[idx]), np.sin(self.yaw[idx])
        dx_r = px - self.xy[idx, 0]
        dy_r = py - self.xy[idx, 1]
        u = float(np.clip(dx_r * t_hat_x + dy_r * t_hat_y, -1.0, 1.0))  # 切向弧长偏移 [m]
        i2 = (idx + (1 if u >= 0.0 else -1)) % self.n_pts
        w = abs(u)
        px_ref = (1 - w) * self.xy[idx, 0] + w * self.xy[i2, 0]
        py_ref = (1 - w) * self.xy[idx, 1] + w * self.xy[i2, 1]
        phi_p = self.yaw[idx] + w * wrap_pi(self.yaw[i2] - self.yaw[idx])
        e_y = -np.sin(phi_p) * (px - px_ref) + np.cos(phi_p) * (py - py_ref)
        e_psi = wrap_pi(yaw - phi_p)
        return e_y, e_psi, idx + u

    def ref_lerp(self, arr, pos):
        """参考数组在分数弧长 pos [m] 处的线性插值 (环形)."""
        i0 = int(np.floor(pos)) % self.n_pts
        i1 = (i0 + 1) % self.n_pts
        w = pos - np.floor(pos)
        return (1.0 - w) * arr[i0] + w * arr[i1]
