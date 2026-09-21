# mpc/ — Barcelona LTV-MPC 模块化版本

`run_mpc_barcelona - 88.7s.py` (2051 行单文件) 拆成独立模块, **控制框架与数值逐位不变**
(离线用同一个确定性假车模型跑 orig vs new: baseline / 88.7s 黄金 flags / 全开关组合 三组 1200-1500 步日志逐列 bit-identical).

## 运行

在仓库根目录 (需要 `config.yml`, `ks_barcelona_racing_line.csv`, `assetto_corsa_gym/`, 各 `*.npz`):

```bash
C:\Users\14212\anaconda3\envs\p309\python.exe run_mpc_barcelona_modular.py --model-rx --grip-corr-file grip_corr_ai_patched_v2.npz --front-circle --brk-eff 0.65 --use-brake-circle-corr --use-drive-map --axle-vgrip --ref-ss --ref-align --fy-combined --steer-gain 2.11 --grip-margin 0.90
```

或 `python -m mpc [flags]`. 所有 flags / 环境变量 / 日志列 / 输出文件名 与原脚本一致.

## 数据流

```
config.parse_config ──► RunConfig
        │
        ├─► vehicle.VehicleParams.from_config   (Iz / steer-gain / mu-scale / pac-b 折算, 派生 FZF0, DELTA_MAX, AF_CAP...)
        ├─► params.MPCParams.from_config        (Np, dt, 权重 ÷gain², BRK_EFF 覆盖)  +  params.LoopParams (门限)
        │
track.Track(csv, TrackLength) ──► xy / yaw / kappa / v_ai / KD-tree / frenet_errors / ref_lerp
        │
tire.TireLimits(veh) ──► alat_max / alat_axle / 轴刹车圆 / 后轴驱动圆 / ar_ss 反查
learned_maps: GripCorr(s) · BrakeCap(v) · BrakeCircleCorr(v,occ) · DriveMap(v,acc)
        │
planner.SpeedPlanner ──► v_grip ──► build_v_profile(vref, brk_eff) ──► get_mpc_reference(s0) = (Vx_ref, kappa, corr)
        │
dynamics.VehicleDynamics ──► fdot / fABf (CasADi)      nn_ff.ResidualNN (可选, 只修 D 项)
qp.LTVQP ──► 参数化 cvxpy/OSQP 问题
controller.MPCController.solve(x0, Vx_ref, kappa, corr) ──► (delta, ax)
        │
startup.KinematicStartup (Vx<15 起步)   actuation.Actuator (delta↔steer, ax→踏板)
residual.ResidualCollector · diagnostics.FreqDiag/lap_report · sweep.BrkEffSweep
        │
runner.run_loop ──► history ──► postprocess.autosave / print_stats / make_plots
```

## 与原脚本的差异 (仅非控制路径)

- 全局可变量 `BRK_EFF` 改为 `planner.brk_eff` + `controller.brk_eff`, 由 `BrkEffSweep.set_brk_eff` 同步换挡
  (原脚本 `build_v_profile` 读全局, 现显式传参 `build_v_profile(vref_scale, brk_eff)`).
- `--brk-eff-sweep` 自动附加 `--front-circle` 时, EXPERIMENT_TAG 不再重复两次 `-fc0.56` (原脚本的小 bug).
- `VREF_SWEEP=True` 采数模式下起跑前会按 `VREF_START` 重建一次 v_prof (原脚本第一圈仍用 VREF=1 的剖面).
- 已弃用常量 (`AX_MAX`, `BRAKE_DECEL`, `BRAKE_MARGIN`, `NN_RESID_CLAMP`, `WARM_F` 斜坡) 不再保留 (`WARM_F=1.0` 仍写入日志列).
- `build_system(cfg, track_length)` 可脱离 AC 组装全部部件, 便于离线单测.
