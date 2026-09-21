# mpc/ 包 vs 原单文件脚本 的逐步等价测试 (离线, 不需要 AC)

`fake_ac.py` 把 `AssettoCorsaEnv.assettoCorsa` 换成一个确定性 Frenet 自行车模型假车
(有 1 步作动延迟), 原脚本与 `mpc` 包各跑一遍同样的 flags, 再用 `diff_logs.py` 逐列对比
`mpc_tracking_log.csv` (除墙钟列 t/dt/solve_ms/ts_ac/sim_dt/d_steps).

```
# 准备两个工作目录 orig/ new/, 各放 config.yml, ks_barcelona_racing_line.csv, 用到的 *.npz;
# orig/ 里再放原脚本副本 run_orig.py
set MPC_REPO=<仓库根目录>  &  set PYTHONIOENCODING=utf-8  &  set FAKE_STEPS=1500
cd orig & python ..\run_compare.py orig --max-laps 1 <flags>
cd new  & python ..\run_compare.py new  --max-laps 1 <flags>
python diff_logs.py orig\mpc_tracking_log.csv new\mpc_tracking_log.csv
```

2026-09-21 结果: baseline / 88.7s 黄金 flags / 全开关组合 (brk-eff-sweep, brake-cap, nn-ff, steer-calm,
delay-steps 1.5, vy-off, mu-scale, pac-b-scale, qp-grip-corr, iz 817 ...) 三组各 1200-1500 步 全部 IDENTICAL
(仅 experiment_tag 字符串差一处: 原脚本 sweep 自动附加 front-circle 时重复 `-fc0.56`).

注: 该假车只保证与控制器模型状态约定一致, 不代表 AC 真车 (在它上面控制器会 PANIC 循环), 只用于等价性回归.
