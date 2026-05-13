"""Config file reader for GUBAS (hou_config.cfg → simulation parameters)."""

import configparser
import math
import numpy as np
from numpy import linalg as la

from .benchmark import read_bench


def hou_config_read(filename):
    """Parse a GUBAS config file and return all simulation parameters.

    Units returned are km-kg-s (the internal GUBAS convention):
      - positions / axes : km
      - densities        : kg/km³
      - G                : km³/(kg·s²)
      - angles           : rad

    Returns a tuple of 54 values matching the order expected by
    ``write_icfile`` and ``hou_shell_cfg.py``.
    """
    cfg = configparser.ConfigParser()
    cfg.read(filename)

    fahnestock_flag = cfg.getboolean("Initial Conditions", "Fahnestock Input File Flag")
    c_flag  = cfg.getboolean("Initial Conditions", "B into A Euler Flag")
    cc_flag = cfg.getboolean("Initial Conditions", "A into N Euler Flag")
    integ   = cfg.getint("Integration Settings", "Integrator Flag")
    tgen    = cfg.getint("Body Model Definitions", "Inertia Integral Generation Flag")
    a_shape = cfg.getint("Body Model Definitions", "Primary Shape Flag")
    b_shape = cfg.getint("Body Model Definitions", "Secondary Shape Flag")

    n  = cfg.getint("Mutual Gravity Expansion Parameters", "Gravity Expansion Truncation Order")
    na = cfg.getint("Mutual Gravity Expansion Parameters", "Primary Inertia Integral Truncation Order")
    nb = cfg.getint("Mutual Gravity Expansion Parameters", "Secondary Inertia Integral Truncation Order")

    t0  = cfg.getfloat("Integration Settings", "Start Time")
    tf  = cfg.getfloat("Integration Settings", "Final Time")
    h   = cfg.getfloat("Integration Settings", "Fixed Time Step")
    tol = cfg.getfloat("Integration Settings", "Absolute Tolerance")

    aa = cfg.getfloat("Body Model Definitions", "Primary Semi-Major Axis")
    ba = cfg.getfloat("Body Model Definitions", "Primary Semi-Intermediate Axis")
    ca = cfg.getfloat("Body Model Definitions", "Primary Semi-Minor Axis")
    ab = cfg.getfloat("Body Model Definitions", "Secondary Semi-Major Axis")
    bb = cfg.getfloat("Body Model Definitions", "Secondary Semi-Intermediate Axis")
    cb = cfg.getfloat("Body Model Definitions", "Secondary Semi-Minor Axis")

    tet_file_a  = cfg.get("Body Model Definitions", "Primary Tetrahedron File")
    vert_file_a = cfg.get("Body Model Definitions", "Primary Vertex File")
    tet_file_b  = cfg.get("Body Model Definitions", "Secondary Tetrahedron File")
    vert_file_b = cfg.get("Body Model Definitions", "Secondary Vertex File")

    post_processing = cfg.getint("Output Settings", "Post Processing")
    out_freq        = cfg.getfloat("Output Settings", "Fixed Output Frequency")
    out_time_name   = cfg.get("Output Settings", "Specified Time List Filename")
    case            = cfg.get("Output Settings", "Case Name")

    flyby_toggle  = cfg.getint("Additional Forces and Perturbations", "Flyby")
    sg_toggle     = cfg.getint("Additional Forces and Perturbations", "Solar Gravity")
    tt_toggle     = cfg.getint("Additional Forces and Perturbations", "Tidal Torque")
    helio_toggle  = cfg.getint("Additional Forces and Perturbations", "Heliocentric Orbit")
    mplanet       = cfg.getfloat("Additional Forces and Perturbations", "Planetary Mass")
    a_hyp         = cfg.getfloat("Additional Forces and Perturbations", "Semimajor Axis")
    e_hyp         = cfg.getfloat("Additional Forces and Perturbations", "Eccentricity")
    i_hyp         = cfg.getfloat("Additional Forces and Perturbations", "Inclination")
    raan_hyp      = cfg.getfloat("Additional Forces and Perturbations", "RAAN")
    om_hyp        = cfg.getfloat("Additional Forces and Perturbations", "Argument of Periapsis")
    tau_hyp       = cfg.getfloat("Additional Forces and Perturbations", "Flyby Time")
    msolar        = cfg.getfloat("Additional Forces and Perturbations", "Solar Mass")
    a_helio       = cfg.getfloat("Additional Forces and Perturbations", "Heliocentric Semimajor Axis")
    e_helio       = cfg.getfloat("Additional Forces and Perturbations", "Heliocentric Eccentricity")
    i_helio       = cfg.getfloat("Additional Forces and Perturbations", "Heliocentric Inclination")
    raan_helio    = cfg.getfloat("Additional Forces and Perturbations", "Heliocentric RAAN")
    om_helio      = cfg.getfloat("Additional Forces and Perturbations", "Heliocentric Argument of Periapsis")
    tau_helio     = cfg.getfloat("Additional Forces and Perturbations", "Time of periapsis passage")
    sol_rad       = cfg.getfloat("Additional Forces and Perturbations", "Solar Orbit Radius")
    au_def        = cfg.getfloat("Additional Forces and Perturbations", "AU Definition") / 1000.0  # m → km
    love1         = cfg.getfloat("Additional Forces and Perturbations", "Primary Love Number")
    love2         = cfg.getfloat("Additional Forces and Perturbations", "Secondary Love Number")
    refrad1       = cfg.getfloat("Additional Forces and Perturbations", "Primary Reference Radius")
    refrad2       = cfg.getfloat("Additional Forces and Perturbations", "Secondary Reference Radius")
    eps1          = cfg.getfloat("Additional Forces and Perturbations", "Primary Tidal Lag Angle")
    eps2          = cfg.getfloat("Additional Forces and Perturbations", "Secondary Tidal Lag Angle")
    msun          = cfg.getfloat("Additional Forces and Perturbations", "Sun Mass")

    if fahnestock_flag:
        g, rho_a, rho_b, x0 = read_bench(
            "systemdata_standard_MKS_units", "initstate_standard_MKS_units"
        )
    else:
        g    = cfg.getfloat("Gravity Parameter", "G") / 1.0e9  # m³/(kg·s²) → km³/(kg·s²)
        rho_a = cfg.getfloat("Body Model Definitions", "Primary Density") * 1.0e12   # g/cm³ → kg/km³
        rho_b = cfg.getfloat("Body Model Definitions", "Secondary Density") * 1.0e12

        x0 = np.zeros(30)
        x0[0] = cfg.getfloat("Initial Conditions", "Relative Position X") / 1000.0
        x0[1] = cfg.getfloat("Initial Conditions", "Relative Position Y") / 1000.0
        x0[2] = cfg.getfloat("Initial Conditions", "Relative Position Z") / 1000.0
        x0[3] = cfg.getfloat("Initial Conditions", "Relative Velocity X") / 1000.0
        x0[4] = cfg.getfloat("Initial Conditions", "Relative Velocity Y") / 1000.0
        x0[5] = cfg.getfloat("Initial Conditions", "Relative Velocity Z") / 1000.0
        x0[6] = cfg.getfloat("Initial Conditions", "Primary Angular Velocity X")
        x0[7] = cfg.getfloat("Initial Conditions", "Primary Angular Velocity Y")
        x0[8] = cfg.getfloat("Initial Conditions", "Primary Angular Velocity Z")
        x0[9]  = cfg.getfloat("Initial Conditions", "Secondary Angular Velocity X")
        x0[10] = cfg.getfloat("Initial Conditions", "Secondary Angular Velocity Y")
        x0[11] = cfg.getfloat("Initial Conditions", "Secondary Angular Velocity Z")

        if not c_flag:
            x0[21] = cfg.getfloat("Initial Conditions", "B into A (1,1)")
            x0[22] = cfg.getfloat("Initial Conditions", "B into A (1,2)")
            x0[23] = cfg.getfloat("Initial Conditions", "B into A (1,3)")
            x0[24] = cfg.getfloat("Initial Conditions", "B into A (2,1)")
            x0[25] = cfg.getfloat("Initial Conditions", "B into A (2,2)")
            x0[26] = cfg.getfloat("Initial Conditions", "B into A (2,3)")
            x0[27] = cfg.getfloat("Initial Conditions", "B into A (3,1)")
            x0[28] = cfg.getfloat("Initial Conditions", "B into A (3,2)")
            x0[29] = cfg.getfloat("Initial Conditions", "B into A (3,3)")
            c_mat = x0[21:30].reshape(3, 3)
        else:
            th1 = cfg.getfloat("Initial Conditions", "B into A Euler 1 X")
            th2 = cfg.getfloat("Initial Conditions", "B into A Euler 2 Y")
            th3 = cfg.getfloat("Initial Conditions", "B into A Euler 3 Z")
            c_mat = np.array([
                [math.cos(th2)*math.cos(th3),
                 math.sin(th1)*math.sin(th2)*math.cos(th3) + math.cos(th1)*math.sin(th3),
                 -math.cos(th1)*math.sin(th2)*math.cos(th3) + math.sin(th1)*math.sin(th3)],
                [-math.cos(th2)*math.sin(th3),
                 -math.sin(th1)*math.sin(th2)*math.sin(th3) + math.cos(th1)*math.cos(th3),
                 math.cos(th1)*math.sin(th2)*math.sin(th3) + math.sin(th1)*math.cos(th3)],
                [math.sin(th2), -math.sin(th1)*math.cos(th2), math.cos(th1)*math.cos(th2)],
            ]).T
            x0[21:30] = c_mat.reshape(9)

        if not cc_flag:
            x0[12] = cfg.getfloat("Initial Conditions", "A into N (1,1)")
            x0[13] = cfg.getfloat("Initial Conditions", "A into N (1,2)")
            x0[14] = cfg.getfloat("Initial Conditions", "A into N (1,3)")
            x0[15] = cfg.getfloat("Initial Conditions", "A into N (2,1)")
            x0[16] = cfg.getfloat("Initial Conditions", "A into N (2,2)")
            x0[17] = cfg.getfloat("Initial Conditions", "A into N (2,3)")
            x0[18] = cfg.getfloat("Initial Conditions", "A into N (3,1)")
            x0[19] = cfg.getfloat("Initial Conditions", "A into N (3,2)")
            x0[20] = cfg.getfloat("Initial Conditions", "A into N (3,3)")
        else:
            th1 = cfg.getfloat("Initial Conditions", "A into N Euler 1 X")
            th2 = cfg.getfloat("Initial Conditions", "A into N Euler 2 Y")
            th3 = cfg.getfloat("Initial Conditions", "A into N Euler 3 Z")
            cc_mat = np.array([
                [math.cos(th2)*math.cos(th3),
                 math.sin(th1)*math.sin(th2)*math.cos(th3) + math.cos(th1)*math.sin(th3),
                 -math.cos(th1)*math.sin(th2)*math.cos(th3) + math.sin(th1)*math.sin(th3)],
                [-math.cos(th2)*math.sin(th3),
                 -math.sin(th1)*math.sin(th2)*math.sin(th3) + math.cos(th1)*math.cos(th3),
                 math.cos(th1)*math.sin(th2)*math.sin(th3) + math.sin(th1)*math.cos(th3)],
                [math.sin(th2), -math.sin(th1)*math.cos(th2), math.cos(th1)*math.cos(th2)],
            ]).T
            x0[12:21] = cc_mat.reshape(9)

        # rotate secondary angular velocity from A frame to B frame
        x0[9:12] = c_mat @ x0[9:12]

    return (
        g, n, na, nb, aa, ba, ca, ab, bb, cb,
        a_shape, b_shape, rho_a, rho_b, t0, tf,
        tet_file_a, vert_file_a, tet_file_b, vert_file_b,
        x0, tgen, integ, h, tol,
        out_freq, out_time_name, case,
        flyby_toggle, helio_toggle, sg_toggle, tt_toggle,
        mplanet, a_hyp, e_hyp, i_hyp, raan_hyp, om_hyp, tau_hyp,
        msolar, a_helio, e_helio, i_helio, raan_helio, om_helio, tau_helio,
        sol_rad, au_def,
        love1, love2, refrad1, refrad2, eps1, eps2,
        msun, post_processing,
    )
