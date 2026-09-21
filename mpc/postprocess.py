# -*- coding: utf-8 -*-
"""运行结束后处理: 日志自动保存 / 统计摘要 / 频率诊断汇总 / 速度-轨迹出图."""
import os
import time

import numpy as np
import pandas as pd

from .track import Track

LOG_PATH = 'mpc_tracking_log.csv'
LOG_COPY_DIR = 'logs_nn'


def autosave(history):
    """正常结束/崩溃/中断都存 (防止 history 被后续清空导致空文件); 并存一份带时间戳的训练集副本."""
    if len(history) == 0:
        print("[autosave] skipped: history is empty")
        return
    try:
        pd.DataFrame(history).to_csv(LOG_PATH, index=False)
        print(f"[autosave] saved {LOG_PATH} ({len(history)} rows)")
        try:
            os.makedirs(LOG_COPY_DIR, exist_ok=True)
            pcopy = os.path.join(LOG_COPY_DIR, 'mpc_log_' + time.strftime('%m%d_%H%M%S') + '.csv')
            pd.DataFrame(history).to_csv(pcopy, index=False)
            print(f'[autosave] 训练集副本 -> {pcopy}')
        except Exception as e:
            print(f'[autosave] 副本失败(非致命): {e}')
    except Exception as e:
        print(f"[autosave] failed (non-fatal): {e}")


def print_vref_sweep(lap_reports):
    if not lap_reports:
        return
    print('\n===== VREF 扫描汇总 (每圈) =====')
    print('%4s %6s %7s %9s %8s %7s %10s' % ('lap', 'VREF', 'n', 'RMS_ey', 'max_ey', 'sat%', '前轮p90°'))
    for r in lap_reports:
        print('%4d %6.2f %7d %9.3f %8.2f %7.1f %10.1f' %
              (r['lap'], r['vref'], r['n'], r['rms_ey'], r['max_ey'], r['sat_pct'], r['af_p90_deg']))


def print_stats(df: pd.DataFrame, dt_mpc: float):
    df.to_csv(LOG_PATH, index=False)
    print(f'saved {LOG_PATH} ({len(df)} rows)')
    if len(df) == 0:
        return
    print(f"RMS e_y   = {np.sqrt((df.e_y**2).mean()):.3f} m")
    print(f"RMS e_psi = {np.degrees(np.sqrt((df.e_psi**2).mean())):.2f} deg")
    print(f"max |e_y| = {df.e_y.abs().max():.2f} m")
    print(f"Vx MAE    = {(df.Vx - df.v_ref).abs().mean():.2f} m/s")
    print(f"solve rate = {df.solve_ok.mean()*100:.1f}%")
    print(f"avg solve  = {df.solve_ms.mean():.1f} ms")

    real_dt = df.dt.median()
    print("--- 频率诊断 ---")
    print(f"real loop  = {1/real_dt:.2f} Hz (墙钟 dt={real_dt:.4f}s, dt_mpc={dt_mpc})")
    ref = real_dt
    if 'sim_dt' in df.columns and df.sim_dt.notna().any():
        sd = df.sim_dt[df.sim_dt.notna() & (df.sim_dt > 0)]
        if len(sd):
            ref = sd.median()
            print(f"sim (srv)  = {1/ref:.2f} Hz (timestamp_ac 帧间隔={ref:.4f}s)  <- 更权威")
    rel = abs(dt_mpc - ref) / ref
    verdict = "OK 匹配, 频率不是问题" if rel <= 0.20 else f"!! 不匹配 -> 把 dt_mpc 改成 {ref:.3f} 重建MPC"
    print(f"dt_mpc 偏差 = {rel*100:.0f}%  -> {verdict}")
    if 'd_steps' in df.columns:
        print(f"平均跳帧   = {df.d_steps.mean():.2f} (1=无积压; >1.5=求解慢于发帧, 状态陈旧)")


def make_plots(df: pd.DataFrame, track: Track, dt_mpc: float):
    """(1) 实际vs参考速度沿里程; (2) 实际vs参考轨迹. 存 PNG (时间戳 + 固定名), 尝试弹窗."""
    import matplotlib.pyplot as plt
    if len(df) == 0:
        print('[plot] history 为空, 跳过'); return
    ts = time.strftime('%m%d_%H%M%S')
    zh = False
    try:   # 中文字体 (Windows 常见), 找不到则退回英文标签避免豆腐块
        from matplotlib.font_manager import fontManager
        avail = {f.name for f in fontManager.ttflist}
        for cand in ('Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi'):
            if cand in avail:
                plt.rcParams['font.sans-serif'] = [cand, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                zh = True
                break
    except Exception:
        zh = False
    T = (lambda z, e: z if zh else e)

    # 选一整圈: MPC 步数最多的那圈
    mpc = df[df['mode'] == 'MPC']
    if len(mpc) < 20:
        print('[plot] MPC 样本太少, 跳过'); return
    best_lap = mpc.groupby('lap').size().idxmax()
    g = mpc[mpc.lap == best_lap].sort_values('s').reset_index(drop=True)
    keep = [0]   # 去停顿/回绕: 只保留 s 严格递增的点
    for i in range(1, len(g)):
        if g.s.iloc[i] - g.s.iloc[keep[-1]] >= 2.0:
            keep.append(i)
    g = g.iloc[keep].reset_index(drop=True)
    lap_time = len(df[df.lap == best_lap]) * dt_mpc
    ai_at_s = np.interp(g.s.values % track.length, track.s, track.v_ai)

    try:
        fig1, ax1 = plt.subplots(figsize=(13, 4.5))
        ax1.plot(g.s, ai_at_s, color='0.6', ls=':', lw=1.2, label=T('AI 原速 (天花板)', 'AI raw (ceiling)'))
        ax1.plot(g.s, g.v_ref, color='#199e70', ls='--', lw=1.6, label=T('MPC 参考 (v_prof)', 'MPC ref (v_prof)'))
        ax1.plot(g.s, g.Vx, color='#2a78d6', lw=2.0, label=T('实际速度 Vx', 'actual Vx'))
        for nm, en, s in [('弯1', 'C1', 4003), ('发卡', 'Hairpin', 3475), ('T9', 'T9', 2886), ('弯A', 'CA', 1155)]:
            if g.s.min() <= s <= g.s.max():
                ax1.axvline(s, color='0.75', ls='--', lw=0.8)
                ax1.text(s, ax1.get_ylim()[1], T(nm, en), fontsize=8, ha='center', va='bottom', color='0.4')
        ax1.set_xlabel(T('赛道里程 s [m]', 'lap distance s [m]')); ax1.set_ylabel(T('速度 [m/s]', 'speed [m/s]'))
        rms = np.sqrt((g.e_y ** 2).mean())
        ax1.set_title(T(f'速度 vs 里程  (lap{best_lap}, ~{lap_time:.1f}s, RMS_ey={rms:.3f}m)',
                        f'Speed vs distance  (lap{best_lap}, ~{lap_time:.1f}s, RMS_ey={rms:.3f}m)'))
        ax1.legend(loc='lower right', fontsize=9); ax1.grid(alpha=0.25)
        fig1.tight_layout()
        fig1.savefig(f'plot_speed_{ts}.png', dpi=110)
        fig1.savefig('plot_speed_latest.png', dpi=110)
        print(f'[plot] 速度图 -> plot_speed_{ts}.png (+_latest)')
    except Exception as e:
        print(f'[plot] 速度图失败: {e}')

    try:
        fig2, ax2 = plt.subplots(figsize=(9, 8))
        ax2.plot(track.xy[:, 0], track.xy[:, 1], color='0.7', lw=3, alpha=0.5, label=T('参考线 (AI)', 'reference (AI)'))
        sc = ax2.scatter(g.x, g.y, c=g.Vx, cmap='turbo', s=6, label=T('实际轨迹', 'actual'))
        cb = fig2.colorbar(sc, ax=ax2, shrink=0.7); cb.set_label(T('实际速度 [m/s]', 'actual speed [m/s]'))
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
        ax2.set_title(T(f'实际轨迹 vs 参考线  (lap{best_lap}, max|e_y|={g.e_y.abs().max():.2f}m)',
                        f'Actual vs reference path  (lap{best_lap}, max|e_y|={g.e_y.abs().max():.2f}m)'))
        ax2.legend(loc='upper right', fontsize=9); ax2.grid(alpha=0.2)
        fig2.tight_layout()
        fig2.savefig(f'plot_traj_{ts}.png', dpi=110)
        fig2.savefig('plot_traj_latest.png', dpi=110)
        print(f'[plot] 轨迹图 -> plot_traj_{ts}.png (+_latest)')
    except Exception as e:
        print(f'[plot] 轨迹图失败: {e}')

    try:
        plt.show()
    except Exception:
        pass
