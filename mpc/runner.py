# -*- coding: utf-8 -*-
"""主入口: 连接 AC -> 组装 规划器/控制器 -> 25Hz 闭环主循环 (KIN 起步 / MPC / PANIC) -> 日志与出图.

控制框架与 run_mpc_barcelona (88.7s 黄金配置) 逐位一致, 只是把各核心函数拆到独立模块:
  config -> vehicle/params -> track -> tire/learned_maps -> dynamics/nn_ff -> planner -> qp/controller
  -> startup/actuation -> residual/diagnostics/sweep -> postprocess
用法: C:\\Users\\14212\\anaconda3\\envs\\p309\\python.exe -m mpc [flags]   (在仓库根目录运行)
Ctrl-C 中断安全: 日志自动保存 (autosave + logs_nn 副本).
"""
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .actuation import Actuator
from .config import SCRIPT_VERSION, RunConfig, parse_config
from .controller import MPCController
from .diagnostics import FreqDiag, lap_report
from .dynamics import VehicleDynamics
from .learned_maps import BrakeCap, BrakeCircleCorr, DriveMap, GripCorr
from .nn_ff import ResidualNN
from .params import LoopParams, MPCParams
from .planner import SpeedPlanner
from .postprocess import autosave, make_plots, print_stats, print_vref_sweep
from .qp import LTVQP
from .residual import ResidualCollector
from .startup import KinematicStartup
from .sweep import BrkEffSweep
from .tire import TireLimits
from .track import Track
from .vehicle import VehicleParams

logger = logging.getLogger('mpc')

RACING_LINE_CSV = 'ks_barcelona_racing_line.csv'
AC_CONFIG_YML = 'config.yml'
AC_GYM_DIR = './assetto_corsa_gym'


@dataclass
class MPCSystem:
    """一次组装好的全部部件 (便于离线测试: build_system(cfg, track_length) 不需要 AC)."""
    cfg: RunConfig
    veh: VehicleParams
    mp: MPCParams
    lp: LoopParams
    track: Track
    tire: TireLimits
    grip: GripCorr
    brake_cap: BrakeCap
    bc_corr: BrakeCircleCorr
    drive_map: DriveMap
    dyn: VehicleDynamics
    nn: Optional[ResidualNN]
    qp: LTVQP
    planner: SpeedPlanner
    ctrl: MPCController
    kin: KinematicStartup
    act: Actuator


def build_system(cfg: RunConfig, track_length: float, verbose: bool = True) -> MPCSystem:
    """按原脚本顺序组装 (打印顺序也一致): 参考线 -> 车辆 -> MPC 参数 -> 动力学/NN/QP -> 学习表 -> 规划 -> 控制."""
    track = Track(RACING_LINE_CSV, track_length)
    veh = VehicleParams.from_config(cfg, verbose)
    if verbose:
        veh.describe()
    mp = MPCParams.from_config(cfg, verbose)
    lp = LoopParams()
    dyn = VehicleDynamics(veh, mp)
    nn = ResidualNN.load(cfg, veh, mp, verbose)
    qp = LTVQP(veh, mp, verbose)
    tire = TireLimits(veh)
    bc_corr = BrakeCircleCorr(cfg, verbose)
    grip = GripCorr(cfg, mp, track.n_pts, verbose)
    brake_cap = BrakeCap(cfg, mp, verbose)
    drive_map = DriveMap(cfg, mp, verbose)
    planner = SpeedPlanner(track, veh, mp, tire, grip, brake_cap, bc_corr, drive_map, cfg, verbose)
    ctrl = MPCController(veh, mp, cfg, dyn, qp, tire, brake_cap, bc_corr, drive_map, nn)
    kin = KinematicStartup(track, veh)
    act = Actuator(veh, tire, drive_map)
    if verbose:
        print("helpers ready"); print("kinematic startup controller ready"); print("LTV-MPC controller (cvxpy) ready")
    return MPCSystem(cfg, veh, mp, lp, track, tire, grip, brake_cap, bc_corr, drive_map,
                     dyn, nn, qp, planner, ctrl, kin, act)


def connect_ac():
    """连接 AC gym 客户端, 返回 (client, track_length, first_state)."""
    from omegaconf import OmegaConf
    sys.path.extend([os.path.abspath(AC_GYM_DIR)])
    import AssettoCorsaEnv.assettoCorsa as assettoCorsa
    config = OmegaConf.load(AC_CONFIG_YML)
    client = assettoCorsa.make_client_only(config.AssettoCorsa)
    static_info = client.simulation_management.get_static_info()
    track_length = float(static_info['TrackLength'])
    print(f"Track: {static_info['TrackFullName']}, length = {track_length:.1f} m")
    client.setup_connection()
    state = client.step_sim()
    print(f"pos=({state['world_position_x']:.1f}, {state['world_position_y']:.1f}), "
          f"Ux={state['local_velocity_x']:.1f}")
    return client, track_length, state


# ======================================================================
def run_loop(sysm: MPCSystem, client, max_laps: int):
    """25Hz 闭环主循环. 返回 (history, lap_reports, resid)."""
    cfg, veh, mp, lp = sysm.cfg, sysm.veh, sysm.mp, sysm.lp
    track, planner, ctrl, kin, act = sysm.track, sysm.planner, sysm.ctrl, sysm.kin, sysm.act
    dyn, nn = sysm.dyn, sysm.nn

    sweep = BrkEffSweep(cfg, planner, ctrl, logger, max_laps)
    resid = ResidualCollector(cfg, veh, mp, dyn, enabled=lp.COLLECT_RESIDUAL)
    freq = FreqDiag(mp, lp, logger)
    vref_scale = lp.VREF_START if lp.VREF_SWEEP else planner.vref_scale
    if lp.VREF_SWEEP:
        planner.vref_scale = vref_scale
        planner.v_prof = planner.build_v_profile(vref_scale, planner.brk_eff)

    client.reset()
    time.sleep(0.05)
    logger.info("start: reset done (kinematic startup)")
    ctrl.prewarm(logger)

    history = []; lap_reports = []
    start_lap = None; cur_lap = None; prev_idx = None; last_s = None
    use_mpc = False; mpc_enter_step = None
    last_u_actual = np.array([0.0, 0.0])
    calm_vyr = None     # --steer-calm (1): x0 的 Vy/r EMA 状态 (仅喂 QP, 日志/残差仍用原始值)
    t_prev = time.perf_counter(); step = 0

    try:
        while True:
            state = client.step_sim()
            if state.get('done'):
                logger.info('done=True'); break
            t_now = time.perf_counter()
            dt = max(t_now - t_prev, 1e-3)
            t_prev = t_now
            freq.update(state, dt, step)

            px = state['world_position_x']; py = state['world_position_y']; yaw = state['yaw']
            Vx = state['local_velocity_x']; Vy = state['local_velocity_y']; r = state['angular_velocity_y']
            # --vy-off: AC 的 LocalVelocity/WorldPosition 在车体原点, 位置与速度成对平移到质心
            if cfg.vy_off != 0.0:
                px -= cfg.vy_off * np.cos(yaw)
                py -= cfg.vy_off * np.sin(yaw)
                Vy = Vy - cfg.vy_off * r
            s_nsp = track.length * state['NormalizedSplinePosition']
            lap = state.get('LapCount', 0)
            if start_lap is None:
                start_lap = lap

            # --- 过线: 统计刚完成的一圈; VREF 扫描提速; BRK_EFF 爬升换挡 ---
            if cur_lap is None:
                cur_lap = lap
            elif lap != cur_lap:
                rpt = lap_report(history, cur_lap, vref_scale, veh, mp)
                if rpt is not None:
                    lap_reports.append(rpt)
                    logger.info('[VREF扫描] lap%d VREF=%.2f | RMS_ey=%.3f max_ey=%.2f 饱和=%.1f%% 前轮侧偏p90=%.1f° max=%.1f°' %
                                (rpt['lap'], rpt['vref'], rpt['rms_ey'], rpt['max_ey'], rpt['sat_pct'],
                                 rpt['af_p90_deg'], rpt['af_max_deg']))
                if lp.VREF_SWEEP and vref_scale < lp.VREF_MAX - 1e-9:
                    vref_scale = min(vref_scale + lp.VREF_STEP, lp.VREF_MAX)
                    planner.vref_scale = vref_scale
                    planner.v_prof = planner.build_v_profile(vref_scale, planner.brk_eff)   # 换挡重建 (~1s)
                    logger.info('[VREF扫描] -> 提速到 VREF=%.2f (v_prof 已重建)' % vref_scale)
                sweep.on_lap_complete(history, cur_lap)
                cur_lap = lap

            # --- 路径匹配 + 亚网格 Frenet 误差 ---
            idx = track.find_nearest(px, py, hint_idx=prev_idx)
            prev_idx = idx
            e_y, e_psi, s_frac = track.frenet_errors(px, py, yaw, idx)

            # --- 参考 (--ref-align: 参考采样起点随延迟补偿同步前移 N·Vx·dt) ---
            s0_ref = s_frac + (mp.DELAY_STEPS * mp.dt * max(Vx, 0.0)
                               if (cfg.ref_align and mp.DELAY_COMP and use_mpc) else 0.0)
            Vx_ref, kappa_ref, corr_ref = planner.get_mpc_reference(s0_ref)

            # --- 控制器选择 (带滞回的速度门限; MPC 一旦接管就一直用, 只有几乎停下才退回) ---
            av = abs(Vx)
            aligned = (abs(e_psi) < lp.MPC_ALIGN_EPSI_ON and abs(e_y) < lp.MPC_ALIGN_EY_ON)
            if not use_mpc and av >= lp.V_MPC_ON and aligned:
                use_mpc = True
                mpc_enter_step = step
                ctrl.sol_prev = None
                ctrl.u_prev = last_u_actual.copy()
                logger.info(f"[step {step}] Vx={av:.1f} -> switch to MPC")
            elif use_mpc and av < lp.V_MPC_OFF:
                use_mpc = False
                ctrl.sol_prev = None
                logger.info(f"[step {step}] Vx={av:.1f} (stopped) -> back to kinematic")

            # --- 求解 ---
            if use_mpc:
                # s 用相对里程 (每步从 0 起): 绝对里程 ~4000 进 QP 会把 OSQP 终止判据放大到 ~4 -> 满舵 bang-bang
                if cfg.steer_calm:
                    vyr_now = np.array([Vy, r])
                    calm_vyr = (vyr_now if calm_vyr is None
                                else cfg.calm_alpha * vyr_now + (1.0 - cfg.calm_alpha) * calm_vyr)
                    x0_mpc = np.array([max(Vx, mp.VX_MIN), calm_vyr[0], calm_vyr[1], e_psi, e_y, 0.0])
                else:
                    x0_mpc = np.array([max(Vx, mp.VX_MIN), Vy, r, e_psi, e_y, 0.0])

                tau = 1.0
                Vx_ref_mpc = Vx_ref.copy()
                if mpc_enter_step is not None:
                    tau = float(np.clip((step - mpc_enter_step) / lp.MPC_TRANS_STEPS, 0.0, 1.0))
                    if tau < 1.0:   # v_cap 只在接管过渡期生效 (2026-07-15 修复: 过渡结束即解除)
                        Vx_ref_mpc = np.minimum(Vx_ref_mpc, Vx + 3.0 + 12.0 * tau)

                delta, ax = ctrl.solve(x0_mpc, Vx_ref_mpc, kappa_ref, corr_ref)

                if tau < 1.0:   # 过渡限幅随标定缩放 (物理不变)
                    delta_lim = np.deg2rad(4.0 + 8.0 * tau) * veh.STEER_GAIN
                    ax_lim = 2.5 + 5.5 * tau
                    delta = float(np.clip(delta, -delta_lim, delta_lim))
                    ax = float(np.clip(ax, -6.0, ax_lim))
                    delta = float(np.clip(delta, -veh.DELTA_MAX, veh.DELTA_MAX))
                mode = 'MPC'
            else:
                delta, ax = kin.control(idx, Vx, e_y, e_psi, Vx_ref[0])
                ctrl.solve_ok = True       # 仅供日志
                ctrl.solve_ms = 0.0
                mode = 'KIN'
                calm_vyr = None            # 断链: 重进 MPC 时 EMA 从新测量起步

            if mode == 'MPC':
                # 05l 教训: 向外漂时先收油
                if abs(e_psi) > lp.AX_CUT_EPSI or abs(e_y) > lp.AX_CUT_EY:
                    ax = min(ax, 0.0)
                if abs(e_psi) > lp.AX_BRAKE_EPSI or abs(e_y) > lp.AX_BRAKE_EY:
                    ax = min(ax, -3.0)
                # 前轴饱和油门保护 (2026-07-11): 用实测状态算 af, 不信线性化; |af| 过峰下 0.6° 即禁正油门
                af_meas = delta - (Vy + veh.lf * r) / max(abs(Vx), mp.VX_MIN)
                if abs(af_meas) > veh.AF_PROTECT:
                    ax = min(ax, 0.0)

            # --- Panic 保护 (大偏差/失控) + BRK 爬升圈中紧急回退 ---
            panic = (abs(e_y) > lp.PANIC_E) or (abs(e_psi) > lp.PANIC_DPHI)
            sweep.check_midlap(panic, e_y, s_nsp)
            if panic:
                # ×STEER_GAIN: panic 线性增益在旧标定下验证, 乘回保持物理输出不变
                delta = float(np.clip(-(0.30 * e_psi + 0.05 * e_y) * veh.STEER_GAIN,
                                      -veh.DELTA_MAX * 0.5, veh.DELTA_MAX * 0.5))
                ax = -4.0 if abs(Vx) > 3.0 else 1.0
                ctrl.sol_prev = None      # 清热启动, 保持 use_mpc (panic 自救后继续 MPC)
                mode = 'PANIC'

            # --- delta -> AC steer (不做事后转向率硬限制, 平滑全在 QP 的 W_DDELTA 里) ---
            steer = act.steer_from_delta(delta)
            delta_applied = act.delta_from_steer(steer)
            ctrl.u_prev = np.array([delta_applied, ax])
            last_u_actual = np.array([delta_applied, ax])
            ctrl.push_u_hist(last_u_actual)
            acc, brake = act.pedal_from_ax(ax, Vx)

            # server reset 检测
            if last_s is not None and abs(s_nsp - last_s) > 100 and av < 1.0:
                logger.warning(f'[step {step}] reset detected, clearing state')
                ctrl.reset()
                use_mpc = False
            last_s = s_nsp

            client.controls.set_controls(steer=steer, acc=acc, brake=brake)
            client.respond_to_server()

            # --- 解析模型对下一步 (Vx,Vy,r) 的预测 + NN 修正量 (诊断列) ---
            try:
                _, _, fp = dyn.fABf(np.array([max(Vx, mp.VX_MIN), Vy, r, e_psi, e_y, float(idx)]),
                                    np.array([delta_applied, ax]), float(kappa_ref[0]))
                fp = np.asarray(fp).flatten()
                pred_Vx, pred_Vy, pred_r = float(fp[0]), float(fp[1]), float(fp[2])
            except Exception:
                pred_Vx = pred_Vy = pred_r = np.nan
            if nn is not None:
                try:
                    nnv = nn(np.array([max(Vx, mp.VX_MIN), Vy, r, e_psi, e_y, 0.0]), np.array([delta_applied, ax]))
                    nn_dVx, nn_dVy, nn_dr = float(nnv[0]), float(nnv[1]), float(nnv[2])
                except Exception:
                    nn_dVx = nn_dVy = nn_dr = np.nan
            else:
                nn_dVx = nn_dVy = nn_dr = 0.0

            resid.update(mode, ctrl.u_hist, Vx, Vy, r, e_psi, e_y, ax, kappa_ref[0], dt)

            history.append(dict(
                t=t_now, step=step, s=s_nsp, lap=lap,
                x=px, y=py, yaw=yaw, Vx=Vx, Vy=Vy, r=r,
                e_y=e_y, e_psi=e_psi, v_ref=Vx_ref[0],
                delta_mpc=delta, delta_applied=delta_applied,
                ax_cmd=ax, steer=steer, acc=acc, brake=brake,
                mode=mode, solve_ok=int(ctrl.solve_ok),
                solve_ms=ctrl.solve_ms, panic=int(panic), dt=dt, warm_f=mp.WARM_F,
                experiment_tag=cfg.experiment_tag,
                use_brake_cap=int(sysm.brake_cap.requested), brake_cap_file=cfg.brake_cap_file if sysm.brake_cap.requested else '',
                brake_cap_blend=cfg.brake_cap_blend if sysm.brake_cap.requested else 0.0,
                use_nn_ff=int(nn is not None), nn_file=cfg.nn_file if nn is not None else '',
                use_front_circle=int(cfg.front_circle), brake_bias=veh.FRONT_BRK_SHARE,
                brk_eff=ctrl.brk_eff, use_bc_corr=int(sysm.bc_corr.enabled),
                steer_calm=int(cfg.steer_calm), calm_alpha=cfg.calm_alpha if cfg.steer_calm else 1.0,
                use_drive_map=int(sysm.drive_map.enabled), axle_vgrip=int(cfg.axle_vgrip),
                ref_ss=int(cfg.ref_ss), ref_vy=ctrl.ref_vy0, ref_epsi=ctrl.ref_epsi0,
                delay_steps=mp.DELAY_STEPS, ref_align=int(cfg.ref_align), vy_off=cfg.vy_off,
                mu_scale_f=cfg.mu_scale_f, mu_scale_r=cfg.mu_scale_r,
                # AC 原生逐胎侧偏角/轮载 (identify_model_bias.py 黄金标准辨识用)
                sa_fl=state.get('SlipAngle_fl', np.nan), sa_fr=state.get('SlipAngle_fr', np.nan),
                sa_rl=state.get('SlipAngle_rl', np.nan), sa_rr=state.get('SlipAngle_rr', np.nan),
                load_fl=state.get('fl_wheel_load', np.nan), load_fr=state.get('fr_wheel_load', np.nan),
                load_rl=state.get('rl_wheel_load', np.nan), load_rr=state.get('rr_wheel_load', np.nan),
                ts_ac=freq.ts_ac, sim_dt=freq.sim_dt, d_steps=freq.d_steps,
                pred_Vx=pred_Vx, pred_Vy=pred_Vy, pred_r=pred_r,
                nn_dVx=nn_dVx, nn_dVy=nn_dVy, nn_dr=nn_dr,
            ))

            if step % lp.VERBOSE_EVERY == 0:
                logger.info(
                    f" {step:5d} lap{lap} s={s_nsp:6.1f} "
                    f"Vx={Vx:5.1f}/{Vx_ref[0]:5.1f} "
                    f"ey={e_y:+.2f} ep={np.degrees(e_psi):+5.1f}d "
                    f"d={np.degrees(delta):+5.1f}d st={steer:+.2f} "
                    f"ax={ax:+5.1f} acc={acc:+.2f} br={brake:+.2f} [{mode}]"
                    f" | real={freq.real_hz:4.1f}Hz sim={freq.sim_hz:4.1f}Hz dskip={freq.mean_dskip:.2f}")

            step += 1
            if lap - start_lap >= max_laps:
                logger.info(f'done {max_laps} laps'); break

    except KeyboardInterrupt:
        logger.info('user interrupt')
    finally:
        client.controls.set_controls(steer=0, acc=-1, brake=-1)
        client.respond_to_server()
        print(f'total {len(history)} steps')
        autosave(history)
        resid.flush()
        try:
            if lp.VREF_SWEEP:
                print_vref_sweep(lap_reports)
        except Exception:
            pass
    return history, lap_reports, resid


# ======================================================================
def main(argv=None):
    print(f'===== run_mpc_barcelona (mpc package) {SCRIPT_VERSION} =====')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    cfg = parse_config(argv)
    client, track_length, state = connect_ac()
    sysm = build_system(cfg, track_length)
    sysm.track.describe(state['world_position_x'], state['world_position_y'])
    max_laps = sysm.lp.max_laps(cfg, sysm.mp)

    history, lap_reports, _ = run_loop(sysm, client, max_laps)

    df = pd.DataFrame(history)
    print_stats(df, sysm.mp.dt)
    try:
        make_plots(df, sysm.track, sysm.mp.dt)
    except Exception as e:
        print(f'[plot] 绘图整体失败 (非致命): {e}')
    try:
        client.close(); print('closed')
    except Exception as e:
        print('close failed:', e)
    return df


if __name__ == '__main__':
    main()
