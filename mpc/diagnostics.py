# -*- coding: utf-8 -*-
"""运行时诊断: 控制频率 / 帧积压 告警 + 每圈统计报告."""
import logging
from collections import deque

import numpy as np

from .params import LoopParams, MPCParams
from .vehicle import VehicleParams


class FreqDiag:
    """频率/仿真步进诊断 (彻底排除控制频率嫌疑).
    real : 客户端主循环实际频率 (墙钟). dt_mpc 必须 ≈ 1/real.
    sim  : 服务器两帧 timestamp_ac 间隔, 即车真正经历的控制周期 (更权威).
    dstep: 服务器帧计数增量. >1 = socket 缓冲有积压被 drain => 用了陈旧状态, 失稳诱因."""

    def __init__(self, mp: MPCParams, lp: LoopParams, logger: logging.Logger):
        self.dt_mpc, self.tol, self.warn_every, self.logger = mp.dt, lp.DT_MISMATCH_TOL, lp.FREQ_WARN_EVERY, logger
        self.dt_hist = deque(maxlen=50)      # 墙钟循环周期
        self.simdt_hist = deque(maxlen=50)   # 服务器帧间隔 (timestamp_ac)
        self.dsteps_hist = deque(maxlen=50)  # 服务器帧计数增量
        self._prev_ts = None
        self._prev_steps = None
        self._last_warn = -10 ** 9
        self.ts_ac = None
        self.sim_dt = np.nan
        self.d_steps = 1

    def update(self, state, dt, step):
        ts_ac = state.get('timestamp_ac', None)
        srv_steps = state.get('steps', None)
        self.sim_dt = (ts_ac - self._prev_ts) if (ts_ac is not None and self._prev_ts is not None) else np.nan
        self.d_steps = (srv_steps - self._prev_steps) if (srv_steps is not None and self._prev_steps is not None) else 1
        self._prev_ts, self._prev_steps = ts_ac, srv_steps
        self.ts_ac = ts_ac
        self.dt_hist.append(dt)
        self.dsteps_hist.append(self.d_steps)
        if np.isfinite(self.sim_dt):
            self.simdt_hist.append(self.sim_dt)

        if len(self.dt_hist) >= 20 and (step - self._last_warn) >= self.warn_every:
            real_dt = float(np.median(self.dt_hist))
            sim_dt_med = float(np.median(self.simdt_hist)) if len(self.simdt_hist) >= 5 else real_dt
            ref_dt = sim_dt_med if (np.isfinite(sim_dt_med) and sim_dt_med > 1e-3) else real_dt
            rel = abs(self.dt_mpc - ref_dt) / max(ref_dt, 1e-3)
            mean_dskip = float(np.mean(self.dsteps_hist))
            if rel > self.tol:
                self.logger.warning(
                    f"[freq] dt_mpc={self.dt_mpc:.3f}s != 实测控制周期: "
                    f"real={real_dt:.3f}s({1/real_dt:.1f}Hz) sim={ref_dt:.3f}s({1/ref_dt:.1f}Hz) "
                    f"偏差{rel*100:.0f}% -> 把 dt_mpc 改成 {ref_dt:.3f} 后重建MPC")
                self._last_warn = step
            elif mean_dskip > 1.5:
                self.logger.warning(
                    f"[freq] 帧积压: 平均每步跳过 {mean_dskip:.1f} 帧 "
                    f"(real={real_dt:.3f}s sim={ref_dt:.3f}s) -> 求解慢于发帧, 控制基于陈旧状态")
                self._last_warn = step

    @property
    def real_hz(self):
        return 1 / max(np.median(self.dt_hist), 1e-3) if len(self.dt_hist) else 0.0

    @property
    def sim_hz(self):
        return 1 / max(np.median(self.simdt_hist), 1e-3) if len(self.simdt_hist) else 0.0

    @property
    def mean_dskip(self):
        return float(np.mean(self.dsteps_hist)) if len(self.dsteps_hist) else 1.0


def lap_report(history, lap, vref_scale, veh: VehicleParams, mp: MPCParams):
    """刚完成一圈的 MPC 段统计: RMS/max e_y, 转向饱和率, 前轮侧偏 p90/max. 样本不足返回 None."""
    lr = [h for h in history if h['lap'] == lap and h['mode'] == 'MPC']
    if len(lr) <= 20:
        return None
    ey = np.array([h['e_y'] for h in lr])
    st = np.array([h['steer'] for h in lr])
    vx = np.array([max(h['Vx'], mp.VX_MIN) for h in lr])
    af = np.abs(np.array([h['delta_applied'] for h in lr])
                - (np.array([h['Vy'] for h in lr]) + veh.lf * np.array([h['r'] for h in lr])) / vx)
    return dict(lap=lap, vref=round(vref_scale, 3), n=len(lr),
                rms_ey=float(np.sqrt((ey ** 2).mean())), max_ey=float(np.abs(ey).max()),
                sat_pct=float((np.abs(st) >= 0.99).mean() * 100),
                af_p90_deg=float(np.degrees(np.quantile(af, 0.9))),
                af_max_deg=float(np.degrees(af.max())))
