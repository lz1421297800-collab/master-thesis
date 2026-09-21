# -*- coding: utf-8 -*-
"""命令行参数 -> RunConfig (实验开关 / 学习文件路径 / 标定缩放) + EXPERIMENT_TAG.

所有 argparse 与 环境变量兜底 (USE_BRAKE_CAP=1 等) 都收敛在这里; 其余模块只读 RunConfig,
不再各自 import argparse / os.environ.
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

SCRIPT_VERSION = ('v2026-07-15b (Iz 默认 898 = AC惯量盒×满油664, 修正旧 817 的干重基准错配; '
                  '删除已证伪的 W_VX 实验; v_cap 只在接管过渡期生效. '
                  '当前最佳 88.7s: --model-rx --grip-corr-file grip_corr_ai_patched_v2.npz --front-circle '
                  '--brk-eff 0.65 --use-brake-circle-corr --use-drive-map --axle-vgrip --ref-ss --ref-align '
                  '--fy-combined --steer-gain 2.11 --grip-margin 0.90)')

IZ_DEFAULT = 898.0   # 横摆惯量默认值 (AC 惯量盒 × 满油质量 664); --iz 可覆盖
DEFAULT_GRIP_CORR_FILE = 'grip_corr.npz'


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '0').strip().lower() in ('1', 'true', 'yes', 'on')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run Barcelona MPC experiments.', add_help=True)
    p.add_argument('--use-brake-cap', action='store_true',
                   help='Enable the learned brake cap profile.')
    p.add_argument('--brake-cap-file', default=os.environ.get('BRAKE_CAP_FILE', 'brake_cap.npz'),
                   help='Brake cap .npz file to load when --use-brake-cap is set.')
    p.add_argument('--brake-cap-blend', type=float, default=float(os.environ.get('BRAKE_CAP_BLEND', '1.0')),
                   help='Blend learned brake cap with the old stable BRK_EFF circle: 0=old, 1=learned.')
    p.add_argument('--use-nn-ff', action='store_true',
                   help='Enable residual NN feed-forward correction.')
    p.add_argument('--nn-file', default=os.environ.get('NN_FILE', 'residual_nn.npz'),
                   help='Residual NN .npz file to load when --use-nn-ff is set.')
    p.add_argument('--front-circle', action='store_true',
                   help='Brake bound from per-axle (front/rear) friction circles instead of BRK_EFF*total circle.')
    p.add_argument('--brake-bias', type=float, default=float(os.environ.get('BRAKE_BIAS', '0.56')),
                   help='Front share of total brake force. brakes.ini FRONT_SHARE=0.55 (cockpit adjustable); '
                        'default 0.56 kept = validated golden config (sensitivity <1%% in 0.54-0.56).')
    p.add_argument('--model-rx', action='store_true',
                   help='Add identified rolling resistance to the longitudinal dynamics (tyres.ini; '
                        '~0.04 m/s^2, thesis-completeness term, negligible for control).')
    p.add_argument('--brk-eff', type=float, default=None,
                   help='Override BRK_EFF (tire-circle brake utilization) for the climb protocol.')
    p.add_argument('--use-brake-circle-corr', action='store_true',
                   help='Multiply brake bound by learned corr(v,occ) from brake_circle_corr.npz.')
    p.add_argument('--brake-circle-corr-file', default=os.environ.get('BC_CORR_FILE', 'brake_circle_corr.npz'),
                   help='Learned brake-circle correction .npz (from learn_brake_circle_corr.py).')
    p.add_argument('--steer-calm', action='store_true',
                   help='Anti-dither: EMA filter on Vy/r into x0 + smoothed linearization controls.')
    p.add_argument('--calm-alpha', type=float, default=float(os.environ.get('CALM_ALPHA', '0.6')),
                   help='EMA weight of the NEW measurement for --steer-calm (lower = smoother, more lag).')
    p.add_argument('--grip-corr-file', default=os.environ.get('GRIP_CORR_FILE', DEFAULT_GRIP_CORR_FILE),
                   help='Grip correction profile .npz (default: baseline grip_corr.npz).')
    p.add_argument('--axle-vgrip', action='store_true',
                   help='Corner speed cap from per-axle steady-state limits min(front, rear) instead of total a_max.')
    p.add_argument('--ref-ss', action='store_true',
                   help='Steady-state consistent reference: Vy_ref/e_psi_ref from model sideslip '
                        'instead of pinning 0 (stops the QP fighting physical sideslip in corners).')
    p.add_argument('--delay-steps', type=float, default=float(os.environ.get('DELAY_STEPS', '1')),
                   help='Actuation delay (in steps, FRACTIONAL allowed, e.g. 1.5) compensated by '
                        'DELAY_COMP. Measured lag is distributed over 1-2 steps (corr lag1=0.42, '
                        'lag2=0.54): pure 2.0 double-counts u_{k-1} -> violent dither. Default 1 = old behavior.')
    p.add_argument('--ref-align', action='store_true',
                   help='Shift reference sampling start by the delay-comp distance (delay_steps*Vx*dt); '
                        'without it kappa/v_ref lag the propagated x0 by 2-6 m.')
    p.add_argument('--vy-off', type=float, default=float(os.environ.get('VY_OFF', '0.0')),
                   help='Longitudinal offset [m] of the AC velocity/position reporting point ahead of CoG. '
                        'Corrects Vy and position to CoG. Identify with identify_model_bias.py.')
    p.add_argument('--mu-scale-r', type=float, default=float(os.environ.get('MU_SCALE_R', '1.0')),
                   help='Scale on the rear tire force curve (multiplies DY_REF_R everywhere). '
                        'CAUTION: grip_corr was trained under the old mu -> retrain before combining.')
    p.add_argument('--mu-scale-f', type=float, default=float(os.environ.get('MU_SCALE_F', '1.0')),
                   help='Scale on the front tire force curve (multiplies DY_REF_F everywhere).')
    p.add_argument('--pac-b-scale', type=float, default=float(os.environ.get('PAC_B_SCALE', '1.0')),
                   help='Scale on Pacejka stiffness factor B (front+rear): lowers sub-peak force slope, '
                        'keeps peak grip, shifts peak angle by 1/scale (AF_CAP/peak angles auto-adjusted).')
    p.add_argument('--pac-b-scale-f', type=float, default=None,
                   help='Per-axle override of --pac-b-scale for the front tire B.')
    p.add_argument('--pac-b-scale-r', type=float, default=None,
                   help='Per-axle override of --pac-b-scale for the rear tire B.')
    p.add_argument('--grip-margin', type=float, default=None,
                   help='Override GRIP_MARGIN (default 0.88 with corr profile / 0.84 fallback). '
                        'Combine with --grip-corr-file none for the clean rebase test.')
    p.add_argument('--qp-grip-corr', action='store_true',
                   help='Apply the learned GRIP_CORR(s) to the QP-side friction circles too '
                        '(lateral a_max, axle brake/drive circles), matching the planner.')
    p.add_argument('--iz', type=float, default=None,
                   help='Override yaw inertia Iz [kg*m^2]. Default 898 = AC INERTIA box x m/12 at the '
                        'fuelled model mass 664. Pass 817 only to reproduce the old (dry-mass) behaviour.')
    p.add_argument('--steer-gain', type=float, default=float(os.environ.get('STEER_GAIN', '1.0')),
                   help='Steering calibration: actual wheel angle = gain x model delta (identified 2.11). '
                        'W_DELTA/W_DDELTA are auto-divided by gain^2 and command limits scaled.')
    p.add_argument('--fy-combined', action='store_true',
                   help='Combined-slip ellipse on REAR lateral force in the dynamics: Fyr *= sqrt(1-occ_r^2). '
                        'Adds dFyr/dax coupling to the QP B-matrix (throttle costs rear grip).')
    p.add_argument('--use-drive-map', action='store_true',
                   help='Learned drive delivery map (drive_map.npz): plan accel, QP drive bound, pedal inverse.')
    p.add_argument('--drive-map-file', default=os.environ.get('DRIVE_MAP_FILE', 'drive_map.npz'))
    p.add_argument('--brk-eff-sweep', action='store_true',
                   help='Raise BRK_EFF by --sweep-step after each healthy lap (panic=0, max|e_y|<=1.5); '
                        'revert and hold on a bad lap. Start value = --brk-eff (or file default).')
    p.add_argument('--sweep-step', type=float, default=0.05)
    p.add_argument('--sweep-max', type=float, default=0.70)
    p.add_argument('--tag', default=os.environ.get('EXPERIMENT_TAG', ''),
                   help='Short label written into the CSV log.')
    p.add_argument('--max-laps', type=int, default=None,
                   help='Override the default number of laps for quick sanity checks.')
    return p


@dataclass
class RunConfig:
    """解析并归一化后的运行配置 (所有数值均已 clip 到安全范围)."""
    # --- 学习文件 / 开关 ---
    use_brake_cap: bool
    brake_cap_file: str
    brake_cap_blend: float
    use_nn_ff: bool
    nn_file: str
    use_bc_corr: bool
    bc_corr_file: str
    use_drive_map: bool
    drive_map_file: str
    grip_corr_file: str
    grip_margin: Optional[float]
    # --- 摩擦圆 / 刹车 ---
    front_circle: bool
    front_brk_share: float
    brk_eff_override: Optional[float]
    brk_sweep: bool
    brk_sweep_step: float
    brk_sweep_max: float
    # --- 参考 / 补偿 ---
    axle_vgrip: bool
    ref_ss: bool
    ref_align: bool
    delay_steps: float
    n_resid_delay: int
    steer_calm: bool
    calm_alpha: float
    qp_grip_corr: bool
    # --- 模型标定 ---
    vy_off: float
    mu_scale_f: float
    mu_scale_r: float
    pac_b_scale_f: float
    pac_b_scale_r: float
    fy_combined: bool
    steer_gain: float
    model_rx: bool
    iz: Optional[float]
    # --- 运行 ---
    experiment_tag: str
    max_laps: Optional[int]
    unknown_args: List[str] = field(default_factory=list)


def _build_tag(a, cfg_vals: dict) -> str:
    """EXPERIMENT_TAG: 基础名 + 各开关后缀 (写进 CSV 日志, 用于离线区分实验)."""
    c = cfg_vals
    tag = a.tag or (
        ('brake-' + os.path.splitext(os.path.basename(c['brake_cap_file']))[0]) if c['use_brake_cap']
        else ('nn-' + os.path.splitext(os.path.basename(c['nn_file']))[0]) if c['use_nn_ff']
        else 'baseline')
    if c['model_rx']:
        tag += '-rx'
    if c['grip_corr_file'] != DEFAULT_GRIP_CORR_FILE:
        tag += '-gc' + os.path.splitext(os.path.basename(c['grip_corr_file']))[0].replace('grip_corr_', '')
    if c['front_circle']:
        tag += f"-fc{c['front_brk_share']:.2f}"
    if c['brk_eff_override'] is not None:
        tag += f"-be{c['brk_eff_override']:.2f}"
    if c['use_bc_corr']:
        tag += '-bcc'
    if c['steer_calm']:
        tag += f"-calm{c['calm_alpha']:.1f}"
    if c['use_drive_map']:
        tag += '-dm'
    if c['axle_vgrip']:
        tag += '-axv'
    if c['ref_ss']:
        tag += '-refss'
    if c['delay_steps'] != 1:
        tag += f"-dly{c['delay_steps']:g}"
    if c['ref_align']:
        tag += '-ra'
    if c['vy_off'] != 0.0:
        tag += f"-vyo{c['vy_off']:+.2f}"
    if c['mu_scale_r'] != 1.0 or c['mu_scale_f'] != 1.0:
        tag += f"-muf{c['mu_scale_f']:.2f}r{c['mu_scale_r']:.2f}"
    if c['pac_b_scale_f'] != 1.0 or c['pac_b_scale_r'] != 1.0:
        tag += f"-pbf{c['pac_b_scale_f']:.2f}r{c['pac_b_scale_r']:.2f}"
    if c['fy_combined']:
        tag += '-fyc'
    if c['qp_grip_corr']:
        tag += '-qgc'
    if c['steer_gain'] != 1.0:
        tag += f"-sg{c['steer_gain']:g}"
    if a.grip_margin is not None:
        tag += f"-gm{a.grip_margin:g}"
    if a.iz is not None and a.iz != IZ_DEFAULT:
        tag += f"-iz{a.iz:g}"
    if c['brk_sweep']:
        tag += f"-besweep{c['brk_sweep_step']:.2f}to{c['brk_sweep_max']:.2f}"
    return tag


def parse_config(argv=None, verbose: bool = True) -> RunConfig:
    a, unknown = build_parser().parse_known_args(argv)
    if unknown and verbose:
        print(f"[EXPERIMENT] ignored unknown args: {unknown}")

    pac_b = float(np.clip(a.pac_b_scale, 0.60, 1.20))
    front_circle = bool(a.front_circle or _env_flag('FRONT_CIRCLE'))
    brk_sweep = bool(a.brk_eff_sweep or _env_flag('BRK_EFF_SWEEP'))
    if brk_sweep and not front_circle:
        if verbose:
            print('!! --brk-eff-sweep 只在 --front-circle 保护下爬升 (旧总圆结构 0.48+ 有崩溃史) -> 自动附加 --front-circle')
        front_circle = True

    vals = dict(
        use_brake_cap=bool(a.use_brake_cap or _env_flag('USE_BRAKE_CAP')),
        brake_cap_file=a.brake_cap_file,
        brake_cap_blend=float(np.clip(a.brake_cap_blend, 0.0, 1.0)),
        use_nn_ff=bool(a.use_nn_ff or _env_flag('USE_NN_FF')),
        nn_file=a.nn_file,
        use_bc_corr=bool(a.use_brake_circle_corr or _env_flag('USE_BC_CORR')),
        bc_corr_file=a.brake_circle_corr_file,
        use_drive_map=bool(a.use_drive_map or _env_flag('USE_DRIVE_MAP')),
        drive_map_file=a.drive_map_file,
        grip_corr_file=a.grip_corr_file,
        grip_margin=(float(np.clip(a.grip_margin, 0.70, 1.00)) if a.grip_margin is not None else None),
        front_circle=front_circle,
        front_brk_share=float(np.clip(a.brake_bias, 0.30, 0.75)),
        brk_eff_override=a.brk_eff,
        brk_sweep=brk_sweep,
        brk_sweep_step=float(np.clip(a.sweep_step, 0.01, 0.10)),
        brk_sweep_max=float(np.clip(a.sweep_max, 0.42, 1.00)),
        axle_vgrip=bool(a.axle_vgrip or _env_flag('AXLE_VGRIP')),
        ref_ss=bool(a.ref_ss or _env_flag('REF_SS')),
        ref_align=bool(a.ref_align or _env_flag('REF_ALIGN')),
        delay_steps=float(np.clip(a.delay_steps, 1.0, 4.0)),   # 可分数 (如 1.5)
        steer_calm=bool(a.steer_calm or _env_flag('STEER_CALM')),
        calm_alpha=float(np.clip(a.calm_alpha, 0.2, 1.0)),
        qp_grip_corr=bool(a.qp_grip_corr or _env_flag('QP_GRIP_CORR')),
        vy_off=float(np.clip(a.vy_off, -2.0, 2.0)),
        mu_scale_f=float(np.clip(a.mu_scale_f, 0.85, 1.10)),
        mu_scale_r=float(np.clip(a.mu_scale_r, 0.85, 1.10)),
        pac_b_scale_f=(float(np.clip(a.pac_b_scale_f, 0.60, 1.25)) if a.pac_b_scale_f is not None else pac_b),
        pac_b_scale_r=(float(np.clip(a.pac_b_scale_r, 0.60, 1.25)) if a.pac_b_scale_r is not None else pac_b),
        fy_combined=bool(a.fy_combined or _env_flag('FY_COMBINED')),
        steer_gain=float(np.clip(a.steer_gain, 0.5, 3.0)),
        model_rx=bool(a.model_rx or _env_flag('MODEL_RX')),
        iz=(float(np.clip(a.iz, 600.0, 1100.0)) if a.iz is not None else None),
        max_laps=a.max_laps,
    )
    vals['n_resid_delay'] = int(round(vals['delay_steps']))   # 残差配对用整数步
    vals['experiment_tag'] = _build_tag(a, vals)
    cfg = RunConfig(unknown_args=list(unknown), **vals)

    if verbose:
        print(f"[EXPERIMENT] tag={cfg.experiment_tag} request_brake_cap={cfg.use_brake_cap} "
              f"brake_cap_file={cfg.brake_cap_file} request_nn_ff={cfg.use_nn_ff} nn_file={cfg.nn_file} "
              f"brake_cap_blend={cfg.brake_cap_blend:.2f} max_laps_override={cfg.max_laps}")
        print(f"[EXPERIMENT] front_circle={cfg.front_circle} brake_bias={cfg.front_brk_share:.2f} "
              f"brk_eff_override={cfg.brk_eff_override} bc_corr={cfg.use_bc_corr} bc_corr_file={cfg.bc_corr_file}")
    return cfg
