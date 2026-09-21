# -*- coding: utf-8 -*-
"""Fake AssettoCorsaEnv.assettoCorsa: deterministic bicycle plant simulated in Frenet coordinates
(same state conventions as the controller model), world pose synthesized from (s, e_y, e_psi).
Injected into sys.modules so both the original script and the mpc package run without AC."""
import os
import sys
import types

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d

N_STEPS = int(os.environ.get('FAKE_STEPS', '900'))
STEER_GAIN_PLANT = float(os.environ.get('FAKE_STEER_GAIN', '1.0'))   # true wheel angle = gain x (12deg x steer)
DELAY = int(os.environ.get('FAKE_DELAY', '1'))   # actuation delay in steps (AC ~1)
RELAX = float(os.environ.get('FAKE_RELAX', '0.6'))   # tire relaxation length [m] (first-order Fy lag)
MODEL_PLANT = int(os.environ.get('FAKE_MODEL_PLANT', '0'))   # 1: use mpc.dynamics fABf as the plant
STEER_TAU = float(os.environ.get('FAKE_STEER_TAU', '0.0'))   # steering actuator first-order lag [s]
DRV_SCALE = float(os.environ.get('FAKE_DRV_SCALE', '1.0'))   # engine strength scale


class _Plant:
    def __init__(self, csv='ks_barcelona_racing_line.csv'):
        df = pd.read_csv(csv).sort_values('lap_dist').drop_duplicates('lap_dist').reset_index(drop=True)
        self.length = float(df.lap_dist.max()) + 1.0
        n = int(self.length)
        s = np.linspace(0, self.length, n, endpoint=False)
        self.xy = np.column_stack([interp1d(df.lap_dist, df.pos_x, fill_value='extrapolate')(s),
                                   interp1d(df.lap_dist, df.pos_y, fill_value='extrapolate')(s)])
        sx = UnivariateSpline(s, self.xy[:, 0], k=4, s=0); sy = UnivariateSpline(s, self.xy[:, 1], k=4, s=0)
        dx, dy, ddx, ddy = sx.derivative(1)(s), sy.derivative(1)(s), sx.derivative(2)(s), sy.derivative(2)(s)
        self.phi = np.arctan2(dy, dx)
        self.kappa = gaussian_filter1d((dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5, sigma=4)
        self.n = n
        # state: Vx, Vy, r, e_psi, e_y, s
        self.Vx, self.Vy, self.r, self.e_psi, self.e_y, self.s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.t = 0.0; self.steps = 0; self.lap = 0
        self.steer = 0.0; self.acc = -1.0; self.brake = -1.0
        self.cmd_q = [(0.0, -1.0, -1.0)] * DELAY
        self.Fyf_st = 0.0; self.Fyr_st = 0.0
        self.m, self.Iz, self.lf, self.lr = 664., 898., 1.7, 1.35
        self.Cf, self.Cr = 109300., 139100.

    def _lerp(self, arr, pos):
        i0 = int(np.floor(pos)) % self.n; i1 = (i0 + 1) % self.n; w = pos - np.floor(pos)
        return (1 - w) * arr[i0] + w * arr[i1]

    def advance(self, dt=0.04, nsub=8):
        h = dt / nsub
        self.cmd_q.append((self.steer, self.acc, self.brake))
        self.steer, self.acc, self.brake = self.cmd_q.pop(0)
        delta_cmd = -self.steer * np.deg2rad(12.0) * STEER_GAIN_PLANT
        if STEER_TAU > 0:
            if not hasattr(self, '_delta_act'): self._delta_act = 0.0
            self._delta_act += (delta_cmd - self._delta_act) * min(dt / STEER_TAU, 1.0)
            delta = self._delta_act
        else:
            delta = delta_cmd
        thr = (self.acc + 1) / 2; brk = (self.brake + 1) / 2
        ax = DRV_SCALE * thr * np.interp(self.Vx, [0, 25, 40, 55, 70, 85], [14, 14, 9, 5, 3, 2.3]) - brk ** 2 * 22.0 - 0.0012 * self.Vx ** 2
        if MODEL_PLANT:
            if not hasattr(self, '_dyn'):
                sys.path.insert(0, os.environ['MPC_REPO'])
                from mpc.config import parse_config; from mpc.vehicle import VehicleParams
                from mpc.params import MPCParams; from mpc.dynamics import VehicleDynamics
                cfg = parse_config([], verbose=False); veh = VehicleParams.from_config(cfg, False)
                self._dyn = VehicleDynamics(veh, MPCParams.from_config(cfg, False))
            x = np.array([max(self.Vx, 3.0), self.Vy, self.r, self.e_psi, self.e_y, self.s])
            kp = self._lerp(self.kappa, self.s)
            _, _, xn = self._dyn.fABf(x, np.array([delta, ax + 0.757 * self.Vx ** 2 / 664.0]), kp)
            xn = np.asarray(xn).flatten()
            if self.Vx < 3.0:   # below model validity: simple longitudinal only
                self.Vx = max(self.Vx + dt * ax, 0.0); self.s = (self.s + dt * self.Vx) % self.length
            else:
                self.Vx, self.Vy, self.r, self.e_psi, self.e_y = xn[0], xn[1], xn[2], xn[3], xn[4]
                if xn[5] >= self.length: self.lap += 1
                self.s = xn[5] % self.length
            self.t += dt; self.steps += 1
            return
        for _ in range(nsub):
            Vxs = max(self.Vx, 1.0)
            dW = self.m * ax * 0.252 / 3.05
            Fzf = max(2883 - dW + 0.92 * self.Vx ** 2, 1e3); Fzr = max(3631 + dW + 1.16 * self.Vx ** 2, 1e3)
            af = delta - (self.Vy + self.lf * self.r) / Vxs
            ar = -(self.Vy - self.lr * self.r) / Vxs
            muf = 1.8 * (0.5 * Fzf / 3451.0) ** (-0.2); mur = 1.8 * (0.5 * Fzr / 3995.0) ** (-0.3)
            mf = lambda B, a: np.sin(1.5 * np.arctan((1 - 0.515) * B * a + 0.515 * np.arctan(B * a)))
            Fyf_ss = muf * Fzf * mf(21.22, af)
            Fyr_ss = mur * Fzr * mf(23.05, ar)
            if RELAX > 0:
                a_rel = min(Vxs * h / RELAX, 1.0)
                self.Fyf_st += a_rel * (Fyf_ss - self.Fyf_st); self.Fyr_st += a_rel * (Fyr_ss - self.Fyr_st)
                Fyf, Fyr = self.Fyf_st, self.Fyr_st
            else:
                Fyf, Fyr = Fyf_ss, Fyr_ss
            if self.Vx < 0.5:
                Fyf = Fyr = 0.0; self.Vy = 0.0; self.r = 0.0
            kp = self._lerp(self.kappa, self.s)
            sdot = (self.Vx * np.cos(self.e_psi) - self.Vy * np.sin(self.e_psi)) / (1.0 - kp * self.e_y)
            dVx = ax - Fyf * np.sin(delta) / self.m + self.r * self.Vy
            dVy = (Fyf * np.cos(delta) + Fyr) / self.m - self.r * self.Vx
            dr = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz
            self.Vx = max(self.Vx + h * dVx, 0.0)
            self.Vy += h * dVy; self.r += h * dr
            self.e_psi += h * (self.r - sdot * kp)
            self.e_y += h * (self.Vx * np.sin(self.e_psi) + self.Vy * np.cos(self.e_psi))
            s_new = self.s + h * sdot
            if s_new >= self.length:
                self.lap += 1
            self.s = s_new % self.length
        self.t += dt; self.steps += 1

    def state(self, done=False):
        phi = self._lerp(self.phi, self.s)  # note: phi wrap ignored (fine away from +-pi)
        px = self._lerp(self.xy[:, 0], self.s) - np.sin(phi) * self.e_y
        py = self._lerp(self.xy[:, 1], self.s) + np.cos(phi) * self.e_y
        return dict(world_position_x=float(px), world_position_y=float(py), yaw=float(phi + self.e_psi),
                    local_velocity_x=self.Vx, local_velocity_y=self.Vy, angular_velocity_y=self.r,
                    NormalizedSplinePosition=self.s / self.length, LapCount=self.lap,
                    timestamp_ac=self.t, steps=self.steps, done=done)


class _Controls:
    def __init__(self, plant): self.p = plant
    def set_controls(self, steer, acc, brake):
        self.p.steer, self.p.acc, self.p.brake = float(steer), float(acc), float(brake)


class _SimMgmt:
    def __init__(self, plant): self.p = plant
    def get_static_info(self):
        return {'TrackLength': self.p.length, 'TrackFullName': 'fake_barcelona'}


class FakeClient:
    def __init__(self):
        self.p = _Plant()
        self.controls = _Controls(self.p)
        self.simulation_management = _SimMgmt(self.p)
        self.n = 0
    def setup_connection(self): pass
    def reset(self):
        self.p = _Plant(); self.controls.p = self.p; self.simulation_management.p = self.p; self.n = 0
    def step_sim(self):
        self.n += 1
        return self.p.state(done=(self.n > N_STEPS))
    def respond_to_server(self): self.p.advance()
    def close(self): pass


def make_client_only(_cfg):
    return FakeClient()


def install():
    pkg = types.ModuleType('AssettoCorsaEnv'); pkg.__path__ = []
    mod = types.ModuleType('AssettoCorsaEnv.assettoCorsa'); mod.make_client_only = make_client_only
    sys.modules['AssettoCorsaEnv'] = pkg
    sys.modules['AssettoCorsaEnv.assettoCorsa'] = mod
