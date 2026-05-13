"""Post-processing of GUBAS binary output files.

Reads ``output_t/t_out.bin`` and ``output_x/x_out.bin``, computes energies
and angular momenta, and writes CSV output files.
"""

import csv
import os
import struct
import numpy as np
from numpy import linalg as la

from .inertia import inertia_rot
from .potential import potential

STATE_LEN = 30  # elements per state vector


def _open_csv(path):
    if os.path.isfile(path):
        os.remove(path)
    return open(path, "a", newline="")


def postprocess(
    g, n, tk, a, b,
    ta, tb, ia_diag, ib_diag,
    mc, ms, m,
    tf, out_freq, h, integ, case,
    out_time_name="",
    flyby_toggle=0, helio_toggle=0,
):
    """Run post-processing on the binary output written by the integrator.

    Args:
        g, n, tk, a, b : gravity constant and expansion parameters
        ta, tb         : primary and secondary inertia integral arrays (q+1)³
        ia_diag        : primary principal MOI  [Ixx, Iyy, Izz] (kg·km²)
        ib_diag        : secondary principal MOI [Ixx, Iyy, Izz] (kg·km²)
        mc, ms, m      : primary mass, secondary mass, reduced mass (kg)
        tf             : final time (s)
        out_freq       : output frequency in seconds (0 = every step, -1 = time list)
        h              : fixed time step (s)
        integ          : integrator flag (1=RK4, 2=LGVI, 3=RK87, 4=ABM)
        case           : case name string for output filenames
        out_time_name  : path to time-list CSV (only used when out_freq == -1)
        flyby_toggle   : 1 if flyby output files should be read
        helio_toggle   : 1 if heliocentric output files should be read
    """
    ia = np.diag(ia_diag)
    ib = np.diag(ib_diag)
    tag = f"{tf}_{out_freq}_{h}_{n}_{integ}_{case}"

    lag_path  = f"LagrangianStateOut_{tag}.csv"
    fh_path   = f"FHamiltonianStateOut_{tag}.csv"
    h_path    = f"HamiltonianStateOut_{tag}.csv"
    ea_path   = f"Energy+AngMom_{tag}.csv"
    cea_path  = f"Conservation_Energy+AngMom_{tag}.csv"

    lag_f = _open_csv(lag_path);  lag_w  = csv.writer(lag_f)
    fh_f  = _open_csv(fh_path);   fh_w   = csv.writer(fh_f)
    h_f   = _open_csv(h_path);    h_w    = csv.writer(h_f)
    ea_f  = _open_csv(ea_path);   ea_w   = csv.writer(ea_f)
    cea_f = _open_csv(cea_path);  cea_w  = csv.writer(cea_f)

    if flyby_toggle:
        hyp_path = f"HyperbolicState_{tag}.csv"
        hyp_f = _open_csv(hyp_path); hyp_w = csv.writer(hyp_f)
    if helio_toggle:
        sol_path = f"SolarState_{tag}.csv"
        sol_f = _open_csv(sol_path); sol_w = csv.writer(sol_f)

    e0 = h0 = None

    if out_freq == -1 or integ == 3:
        # ── time-list or RK87 mode: scan t_out.bin to build index list ──────
        times = [] if integ == 3 else list(np.genfromtxt(out_time_name, delimiter=","))
        indices = []
        with open("output_t/t_out.bin", "rb") as f:
            count = 0
            while True:
                raw = f.read(8)
                if not raw:
                    break
                t_val = struct.unpack("d", raw)[0]
                if integ == 3:
                    indices.append(count)
                elif t_val in times:
                    indices.append(count)
                count += 1

        x_file = open("output_x/x_out.bin", "rb")
        t_file = open("output_t/t_out.bin", "rb")
        for idx in indices:
            t_file.seek(idx * 8)
            t_val = struct.unpack("d", t_file.read(8))[0]
            x_file.seek(STATE_LEN * idx * 8)
            u = struct.unpack(f"{STATE_LEN}d", x_file.read(8 * STATE_LEN))
            e0, h0 = _write_row(
                t_val, u, g, n, tk, a, b, ta, tb, ia, ib, m,
                lag_w, fh_w, h_w, ea_w, cea_w,
                e0, h0,
            )
        x_file.close(); t_file.close()

    else:
        # ── fixed-frequency mode ─────────────────────────────────────────────
        if out_freq == 0.0:
            out_freq = h
        seek_step = int(out_freq / h)
        n_steps   = int(tf / out_freq + 1)

        x_file = open("output_x/x_out.bin", "rb")
        t_file = open("output_t/t_out.bin", "rb")
        hyp_file = sun_file = None
        if flyby_toggle:
            hyp_file = open("output_h/h_out.bin", "rb")
        if helio_toggle:
            sun_file = open("output_sun/sun_out.bin", "rb")

        for f in range(n_steps):
            t_file.seek(seek_step * f * 8)
            t_val = struct.unpack("d", t_file.read(8))[0]
            x_file.seek(STATE_LEN * seek_step * f * 8)
            u = struct.unpack(f"{STATE_LEN}d", x_file.read(8 * STATE_LEN))
            e0, h0 = _write_row(
                t_val, u, g, n, tk, a, b, ta, tb, ia, ib, m,
                lag_w, fh_w, h_w, ea_w, cea_w,
                e0, h0,
            )
            if flyby_toggle and hyp_file:
                hyp_file.seek(6 * seek_step * f * 8)
                hp = struct.unpack("6d", hyp_file.read(48))
                hyp_w.writerow(list(hp))
            if helio_toggle and sun_file:
                sun_file.seek(6 * seek_step * f * 8)
                sol = struct.unpack("6d", sun_file.read(48))
                sol_w.writerow(list(sol))

        x_file.close(); t_file.close()
        if hyp_file:  hyp_file.close()
        if sun_file:  sun_file.close()

    lag_f.close(); fh_f.close(); h_f.close(); ea_f.close(); cea_f.close()
    if flyby_toggle:  hyp_f.close()
    if helio_toggle:  sol_f.close()


def _write_row(t_val, u, g, n, tk, a, b, ta, tb, ia, ib, m,
               lag_w, fh_w, h_w, ea_w, cea_w, e0, h0):
    cc  = np.reshape(u[12:21], (3, 3))   # A→N
    c   = np.reshape(u[21:30], (3, 3))   # B→A
    cs  = cc @ c                          # B→N
    r_a = np.array(u[0:3])               # rel pos in A (km)
    v_a = np.array(u[3:6])               # rel vel in A (km/s)
    wc  = np.array(u[6:9])               # primary ω in A
    ws_a = np.array(u[9:12])             # secondary ω in A
    ws  = c.T @ ws_a                     # secondary ω in B

    r_n = cc @ r_a                        # rel pos in N
    v_n = cc @ v_a                        # rel vel in N
    r_mag = la.norm(r_n)
    e_vec = (cc.T @ (r_n / r_mag)).reshape(1, 3)

    tbp = inertia_rot(c, n, tb)
    u_pot = potential(g, n, tk, a, b, e_vec, r_mag, ta, tbp)

    kt  = 0.5 * m * v_n @ v_n
    kr1 = 0.5 * wc  @ ia @ wc
    kr2 = 0.5 * ws  @ ib @ ws
    h_vec = (m * np.cross(r_n, v_n)
             + cc @ (ia @ wc)
             + cs @ (ib @ ws))
    energy = u_pot + kt + kr1 + kr2

    if e0 is None:
        e0 = energy
        h0 = h_vec.copy()
    de = (e0 - energy) / e0 if e0 != 0 else 0.0
    dh = (la.norm(h0) - la.norm(h_vec)) / la.norm(h0) if la.norm(h0) != 0 else 0.0

    # convert to MKS for output (positions m, momenta kg·m²/s, potential J)
    rpos_m   = r_a * 1000.0
    lvel_m   = v_a * 1000.0
    rmom     = m * lvel_m
    cla      = (ia @ wc) * 1e6
    clb      = (c @ (ib @ ws)) * 1e6
    u_j      = u_pot * 1e6

    lag_w.writerow( [t_val, *rpos_m, *lvel_m, *wc, *ws, *c.reshape(9), *cc.reshape(9), u_j])
    fh_w.writerow(  [t_val, *rpos_m, *rmom, *cla, *clb, *c.reshape(9), *cc.T.reshape(9), u_j])
    h_w.writerow(   [t_val, *rpos_m, *rmom, *cla, *clb, *c.reshape(9), *cc.reshape(9), u_j])
    ea_w.writerow(  [energy, *h_vec])
    cea_w.writerow( [de, dh])

    return e0, h0
