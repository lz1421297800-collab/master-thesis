# -*- coding: utf-8 -*-
"""Run original script and mpc package against the same fake plant; diff per-step logs.
usage: python run_compare.py <orig|new> [flags...]   (cwd must be the orig/ or new/ dir)"""
import os
import runpy
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_ac
fake_ac.install()

which = sys.argv[1]
flags = sys.argv[2:]
sys.argv = ['run', *flags]
if which == 'orig':
    runpy.run_path(os.path.join(os.getcwd(), 'run_orig.py'), run_name='__main__')
else:
    sys.path.insert(0, os.environ['MPC_REPO'])
    from mpc.runner import main
    main(flags)
