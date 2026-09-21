# -*- coding: utf-8 -*-
"""BRK_EFF 每圈爬升协议 (--brk-eff-sweep, 2026-07-11).

健康圈 (panic=0 且 max|e_y|<=1.5) -> +step; 不健康圈 / 圈中失稳 (|e_y|>3 或 panic) -> 回退一档并冻结.
!! 教训: 过线时在线重建 v_prof 冻结控制 1.0s, 车 76m/s 满油门盲跑 53m 冲过 T1 刹车点 -> 出线.
   修复: 启动时预计算所有档位 v_prof, 换挡只换指针 (零冻结).
规划器 (planner.brk_eff / v_prof) 与 控制器 (controller.brk_eff) 必须同步换挡.
"""
import logging
import time

import numpy as np

from .config import RunConfig
from .controller import MPCController
from .planner import SpeedPlanner

BRK_EFF_FLOOR = 0.42


class BrkEffSweep:
    def __init__(self, cfg: RunConfig, planner: SpeedPlanner, ctrl: MPCController,
                 logger: logging.Logger, max_laps: int):
        self.enabled = cfg.brk_sweep
        self.step, self.max = cfg.brk_sweep_step, cfg.brk_sweep_max
        self.planner, self.ctrl, self.logger = planner, ctrl, logger
        self.frozen = False
        self.cache = {}
        self.start = round(planner.brk_eff, 3)
        if not self.enabled:
            return
        be0 = self.start
        n_up = int(np.ceil((self.max - be0) / self.step)) + 1
        ladder = {round(min(be0 + k * self.step, self.max), 3) for k in range(n_up)}
        # 每档的回退值也预计算 (上限档被夹紧后 0.70-0.05=0.65 会脱离主梯子)
        cands = sorted(ladder | {round(max(c - self.step, BRK_EFF_FLOOR), 3) for c in ladder})
        tc = time.perf_counter()
        for be in cands:
            self.cache[be] = planner.build_v_profile(planner.vref_scale, be)
        self.set_brk_eff(be0)
        print(f"[BRK爬升] 预计算 {len(cands)} 条 v_prof 耗时 {time.perf_counter()-tc:.1f}s: {cands}")
        print(f"[BRK爬升] 起点 {be0:.2f}, 每健康圈 +{self.step:.2f}, 上限 {self.max:.2f}, "
              f"MAX_LAPS={max_laps}; 换挡零冻结; 不健康圈自动回退一档并冻结")

    @property
    def brk_eff(self):
        return self.planner.brk_eff

    def set_brk_eff(self, value):
        """规划-控制同步换挡; 缓存缺失时才在线重建 (不应发生)."""
        value = round(value, 3)
        self.planner.brk_eff = value
        self.ctrl.brk_eff = value
        self.planner.v_prof = (self.cache[value] if value in self.cache
                               else self.planner.build_v_profile(self.planner.vref_scale, value))

    def on_lap_complete(self, history, lap):
        """过线: 只统计 MPC 段 (KIN 起步/切换瞬态的 e_y 不构成对当前 BRK_EFF 的证据)."""
        if not self.enabled or self.frozen:
            return
        lh = [h for h in history if h['lap'] == lap and h['mode'] == 'MPC']
        pan = sum(h['panic'] for h in lh)
        mey = max((abs(h['e_y']) for h in lh), default=0.0)
        if pan == 0 and mey <= 1.5:
            if self.brk_eff < self.max - 1e-9:
                self.set_brk_eff(min(self.brk_eff + self.step, self.max))
                self.logger.info('[BRK爬升] lap%d 健康(max|e_y|=%.2f) -> BRK_EFF=%.2f (缓存换挡)' % (lap, mey, self.brk_eff))
            else:
                self.logger.info('[BRK爬升] lap%d 健康, 已在上限 %.2f 保持' % (lap, self.brk_eff))
        else:
            self.set_brk_eff(max(self.brk_eff - self.step, BRK_EFF_FLOOR))
            self.frozen = True
            self.logger.info('[BRK爬升] lap%d 不健康(panic=%d, max|e_y|=%.2f) -> 回退 BRK_EFF=%.2f 并冻结'
                             % (lap, pan, mey, self.brk_eff))

    def check_midlap(self, panic, e_y, s):
        """圈中紧急回退: 首次爬升实测 0.62 在 s2120 圈中失稳, 过线回退根本来不及."""
        if not self.enabled or self.frozen or self.brk_eff <= self.start + 1e-9:
            return
        if panic or abs(e_y) > 3.0:
            self.set_brk_eff(max(self.brk_eff - self.step, BRK_EFF_FLOOR))
            self.frozen = True
            self.logger.warning('[BRK爬升] 圈中失稳(s=%.0f, e_y=%.1f, panic=%d) -> 紧急回退 BRK_EFF=%.2f 并冻结'
                                % (s, e_y, int(panic), self.brk_eff))
