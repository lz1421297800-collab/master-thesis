# -*- coding: utf-8 -*-
"""Barcelona LTV-MPC (Assetto Corsa) — 模块化版本 (控制框架与 run_mpc_barcelona 88.7s 黄金配置一致).

模块索引:
  config        命令行 / 环境变量 -> RunConfig + EXPERIMENT_TAG
  vehicle       F2004 车辆/轮胎/转向标定参数 (含 --iz/--steer-gain/--mu-scale/--pac-b-scale 折算)
  params        MPC 超参数/权重 (MPCParams) + 主循环门限 (LoopParams)
  track         参考线重采样 / 曲率 / 最近点 / 亚网格 Frenet 误差
  tire          numpy 轮胎极限: a_lat_max, 逐轴上限, 轴刹车圆/驱动圆, MF 反查
  learned_maps  GripCorr / BrakeCap / BrakeCircleCorr / DriveMap 四张学习表
  dynamics      CasADi 符号动力学 fdot / fABf (子步 Euler + 自动 Jacobian)
  nn_ff         NN 残差前馈 (只修 D 项, 冻结梯度, 门控+限幅+自检)
  planner       v_grip -> v_prof 前后向扫掠 -> 地平线参考 (Vx_ref, kappa, corr)
  qp            参数化 cvxpy/OSQP LTV-QP
  controller    MPCController.solve: 延迟补偿 / 参考 / 自洽线性化 / 摩擦圆界 / 求解
  startup       低速运动学起步控制器
  actuation     ax -> 踏板, delta <-> steer
  residual      NN 残差样本在线采集
  diagnostics   频率诊断 + 每圈报告
  sweep         BRK_EFF 爬升协议 (预计算缓存零冻结换挡)
  postprocess   autosave / 统计 / 出图
  runner        connect_ac / build_system / run_loop / main
"""
from .config import RunConfig, parse_config
from .runner import MPCSystem, build_system, main, run_loop

__all__ = ['RunConfig', 'parse_config', 'MPCSystem', 'build_system', 'run_loop', 'main']
