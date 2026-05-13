"""Write the ic_input.txt file consumed by the Rust integrator binary."""


def write_icfile(
    g, n, na, nb, aa, ba, ca, ab, bb, cb,
    a_shape, b_shape, rho_a, rho_b, t0, tf,
    ta_file, tb_file, ia_file, ib_file,
    tet_file_a, vert_file_a, tet_file_b, vert_file_b,
    x0, tgen, integ, h, tol,
    flyby_toggle, helio_toggle, sg_toggle, tt_toggle,
    mplanet, a_hyp, e_hyp, i_hyp, raan_hyp, om_hyp, tau_hyp,
    msolar, a_helio, e_helio, i_helio, raan_helio, om_helio, tau_helio,
    sol_rad, au_def,
    love1, love2, refrad1, refrad2, eps1, eps2,
    msun,
):
    """Write all simulation parameters to ``ic_input.txt`` (one value per line).

    All numeric values should already be in the internal km-kg-s convention
    (the conversion happens in ``hou_config_read``).
    """
    lines = [
        repr(g), repr(n), repr(na), repr(nb),
        repr(aa), repr(ba), repr(ca),
        repr(ab), repr(bb), repr(cb),
        repr(a_shape), repr(b_shape),
        repr(rho_a), repr(rho_b),
        repr(t0), repr(tf),
        ta_file, tb_file, ia_file, ib_file,
        tet_file_a, vert_file_a,
        tet_file_b, vert_file_b,
    ]
    lines += [repr(float(v)) for v in x0]
    lines += [
        repr(tgen), repr(integ), repr(h), repr(tol),
        repr(flyby_toggle), repr(helio_toggle), repr(sg_toggle), repr(tt_toggle),
        repr(mplanet),
        repr(a_hyp), repr(e_hyp), repr(i_hyp), repr(raan_hyp), repr(om_hyp), repr(tau_hyp),
        repr(msolar),
        repr(a_helio), repr(e_helio), repr(i_helio),
        repr(raan_helio), repr(om_helio), repr(tau_helio),
        repr(sol_rad), repr(au_def),
        repr(love1), repr(love2),
        repr(refrad1), repr(refrad2),
        repr(eps1), repr(eps2),
        repr(msun),
    ]
    with open("ic_input.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
