# -*- coding: utf-8 -*-
r"""独立运行版 MPC — 模块化入口 (等价于 python -m mpc; 全部逻辑在 mpc/ 包里).
用法: C:\Users\14212\anaconda3\envs\p309\python.exe run_mpc_barcelona_modular.py [flags]
黄金 88.7s 配置:
  --model-rx --grip-corr-file grip_corr_ai_patched_v2.npz --front-circle --brk-eff 0.65
  --use-brake-circle-corr --use-drive-map --axle-vgrip --ref-ss --ref-align --fy-combined
  --steer-gain 2.11 --grip-margin 0.90
"""
from mpc.runner import main

if __name__ == '__main__':
    main()
