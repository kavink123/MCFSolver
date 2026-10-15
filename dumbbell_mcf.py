"""
Axisymmetric mean curvature flow of a rotationally symmetric dumbbell.

Radius profile f(x,t) of a surface of revolution satisfies

    f_t = f_xx / (1 + f_x^2) - 1/f

on [0, x_max] with f_x(0) = 0 (reflection symmetry) and f(x_max) = 0.
The initial surface is a cylinder of radius r on |x| <= x_j joined to
spherical caps of radius R centred at +-L/2, x_j = L/2 - sqrt(R^2 - r^2).

Usage:
    python dumbbell_mcf.py table1        Table 1 (L = 25, two grids)
    python dumbbell_mcf.py scan          dR = 0.01 scan, bracket for R*
    python dumbbell_mcf.py sphere        shrinking-sphere validation
    python dumbbell_mcf.py dt            time-step refinement
    python dumbbell_mcf.py factors       a and b at the transition
    python dumbbell_mcf.py bounds        check f_min^2 >= r^2 - 2t, f_max^2 <= R^2 - 2t
    python dumbbell_mcf.py phase         fixed-alpha transition curve and alpha_min
    python dumbbell_mcf.py sensitivity   stopping tolerance and junction smoothing
    python dumbbell_mcf.py figures       Figures 1-3
    python dumbbell_mcf.py all

Results are appended to results.json.
"""

import argparse
import json
import math
import os

import numpy as np

r0 = 4.0        # neck radius, fixed
L0 = 25.0       # benchmark cap-centre separation
DX = 0.02
DX_FINE = 0.01
DT_FAC = 0.30   # dt = DT_FAC * dx^2
FMIN = 1e-3     # stopping tolerance, f_min / r = 2.5e-4
TOL = 0.01      # f_cap > TOL at neck collapse counts as a pinch


# ---------------------------------------------------------------- geometry

def x_junction(R, L, r=r0):
    return L / 2 - math.sqrt(R * R - r * r)


def L_for_alpha(alpha, R, r=r0):
    # cap-centre separation giving neck aspect ratio alpha = x_j / r
    return 2 * alpha * r + 2 * math.sqrt(R * R - r * r)


def profile(x, R, L, r=r0, w=0.0):
    xj = x_junction(R, L, r)
    f = np.zeros_like(x)
    cyl = x <= max(xj, 0.0)
    f[cyl] = r
    arc = (~cyl) & (x <= L / 2 + R)
    f[arc] = np.sqrt(np.maximum(R * R - (x[arc] - L / 2) ** 2, 0.0))
    if w > 0:
        # tanh blend of width w across the corner at x_j
        xx = np.minimum(x, L / 2 + R)
        arc_all = np.sqrt(np.maximum(R * R - (xx - L / 2) ** 2, 0.0))
        s = 0.5 * (1 + np.tanh((x - xj) / w))
        f = (1 - s) * r + s * arc_all
        f[x > L / 2 + R] = 0.0
    return f


def sphere(x, R):
    return np.sqrt(np.maximum(R * R - x * x, 0.0))


# ------------------------------------------------------------------ solver

def evolve(f0, x, dx, dt_fac=DT_FAC, fmin=FMIN, tmax=14.0,
           neck_hi=None, i_cap=None, track=False, stop_at_neck=True):
    """
    Explicit Euler, centred differences. Returns a dict with
      t_neck : first time min f over x < neck_hi drops below fmin
      f_cap  : f(L/2) at that instant (0 if the surface is extinguishing)
      t_ext  : first time max f < fmin
      hist   : (t, neck min, max) samples if track
    """
    f = f0.copy()
    n = len(f)
    dt = dt_fac * dx * dx
    neck = (x < neck_hi) if neck_hi is not None else None
    fx = np.zeros(n)
    fxx = np.zeros(n)
    hist = []
    t_neck = f_cap = t_ext = None

    for step in range(1, int(tmax / dt) + 1):
        fx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
        fxx[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dx * dx)
        fx[0] = 0.0
        fxx[0] = 2 * (f[1] - f[0]) / (dx * dx)     # ghost node f[-1] = f[1]
        fx[-1] = fxx[-1] = 0.0

        act = f > fmin
        rhs = np.zeros(n)
        rhs[act] = fxx[act] / (1 + fx[act] ** 2) - 1.0 / f[act]
        f += dt * rhs
        np.maximum(f, 0.0, out=f)
        f[-1] = 0.0
        t = step * dt

        if track and step % 200 == 0:
            hist.append((t, float(f[neck].min()), float(f.max())))
        if neck is not None and t_neck is None and f[neck].min() < fmin:
            t_neck = t
            f_cap = float(f[i_cap])
            if stop_at_neck:
                return dict(t_neck=t_neck, f_cap=f_cap, t_ext=None,
                            hist=hist, x=x, f=f.copy())
        if f.max() < fmin:
            t_ext = t
            if t_neck is None:
                t_neck, f_cap = t, 0.0
            break

    return dict(t_neck=t_neck, f_cap=f_cap, t_ext=t_ext,
                hist=hist, x=x, f=f.copy())


def run(R, L=L0, dx=DX, dt_fac=DT_FAC, fmin=FMIN, w=0.0, tmax=14.0,
        track=False, stop_at_neck=True):
    x = np.arange(0.0, L / 2 + R + 2.0 + dx, dx)
    return evolve(profile(x, R, L, w=w), x, dx, dt_fac, fmin, tmax,
                  neck_hi=L / 4, i_cap=int(round((L / 2) / dx)),
                  track=track, stop_at_neck=stop_at_neck)


def pinches(R, L, dx=DX, tol=TOL):
    if R <= r0 or x_junction(R, L) <= 0:
        return False
    d = run(R, L, dx=dx, tmax=25.0)
    return d["f_cap"] is not None and d["f_cap"] > tol


def bracket(Rs, fcaps, tol=TOL):
    # first R with f_cap > tol; returns (R_lo, R_hi, q*)
    for i, (R, v) in enumerate(zip(Rs, fcaps)):
        if v > tol:
            lo = Rs[i - 1] if i else None
            return lo, R, r0 / (0.5 * (lo + R)) if lo else None
    return None, None, None


# ------------------------------------------------------------------- tasks

def table1(out):
    print("R      q       t(.02)   fcap(.02)  t(.01)   fcap(.01)")
    rows = []
    for R in (5.30, 5.35, 5.40, 5.45, 5.50, 5.55, 5.60, 5.70):
        a = run(R, dx=DX)
        b = run(R, dx=DX_FINE)
        print("%.2f  %.4f  %.3f  %.4f    %.3f  %.4f"
              % (R, r0 / R, a["t_neck"], a["f_cap"], b["t_neck"], b["f_cap"]))
        rows.append([R, r0 / R, a["t_neck"], a["f_cap"], b["t_neck"], b["f_cap"]])
    out["table1"] = rows


def scan(out):
    Rs = [round(5.28 + 0.01 * k, 2) for k in range(16)]
    f02 = [run(R, dx=DX)["f_cap"] for R in Rs]
    f01 = [run(R, dx=DX_FINE)["f_cap"] for R in Rs]
    for R, u, v in zip(Rs, f02, f01):
        print("R=%.2f  fcap %.5f (dx=.02)  %.5f (dx=.01)" % (R, u, v))
    for lab, fc in (("dx=0.02", f02), ("dx=0.01", f01)):
        lo, hi, q = bracket(Rs, fc)
        print("%s: R* in (%.2f, %.2f], q* = %.4f" % (lab, lo, hi, q))
    out["scan"] = dict(R=Rs, fcap_dx02=f02, fcap_dx01=f01)


def sphere_test(out):
    print("R0    dx     T_exact   T_num    rel.err")
    rows = []
    for R0 in (4.0, 5.0, 5.4, 6.0):
        for dx in (DX, DX_FINE):
            x = np.arange(0.0, R0 + 2.0 + dx, dx)
            d = evolve(sphere(x, R0), x, dx, tmax=R0 * R0 / 4 + 4)
            T = R0 * R0 / 4
            err = (d["t_ext"] - T) / T
            print("%.1f  %.3f  %.4f  %.4f  %+.4f" % (R0, dx, T, d["t_ext"], err))
            rows.append([R0, dx, T, d["t_ext"], err])
    out["sphere"] = rows


def dt_test(out):
    worst = 0.0
    for R in (5.30, 5.35, 5.40, 5.45, 5.50, 5.60):
        u = run(R, dt_fac=0.30)["f_cap"]
        v = run(R, dt_fac=0.15)["f_cap"]
        worst = max(worst, abs(u - v))
        print("R=%.2f  fcap %.5f -> %.5f  diff %.2e" % (R, u, v, abs(u - v)))
    print("max change %.2e" % worst)
    out["dt_max_change"] = worst


def factors(out):
    Rstar = 5.345
    T = run(5.35)["t_neck"]
    a = T / (r0 * r0 / 2)
    b = T / (Rstar * Rstar / 4)
    print("T = %.4f" % T)
    print("a = T/(r^2/2)  = %.4f" % a)
    print("b = T/(R*^2/4) = %.4f" % b)
    print("sqrt(b/2a) = %.4f" % math.sqrt(b / (2 * a)))
    print("b alone %+.2f%%, a alone %+.2f%%, net %+.2f%%"
          % (100 * (math.sqrt(b) - 1), 100 * (1 / math.sqrt(a) - 1),
             100 * (math.sqrt(b / a) - 1)))
    for R in (5.40, 5.50, 5.70, 6.00):
        t = run(R)["t_neck"]
        print("R=%.2f  t_neck=%.4f  a=%.4f" % (R, t, t / (r0 * r0 / 2)))
    out["factors"] = dict(T=T, a=a, b=b)


def bounds(out):
    # m(t)^2 >= r^2 - 2t for the minimum over |x| <= L/2,
    # M(t)^2 <= R^2 - 2t for the maximum
    rows = []
    for R in (5.30, 5.45, 5.70):
        x = np.arange(0.0, L0 / 2 + R + 2.0 + DX, DX)
        d = evolve(profile(x, R, L0), x, DX, tmax=14.0,
                   neck_hi=L0 / 2, i_cap=int(round((L0 / 2) / DX)), track=True)
        wlo = whi = 0.0
        for t, m, M in d["hist"]:
            lo = r0 * r0 - 2 * t
            if lo > 0:
                wlo = max(wlo, lo - m * m)
            whi = max(whi, M * M - (R * R - 2 * t))
        print("R=%.2f  samples %d  worst defect: min %.2e  max %.2e  T_neck=%.4f"
              % (R, len(d["hist"]), wlo, whi, d["t_neck"]))
        rows.append([R, wlo, whi, d["t_neck"]])
    print("r^2/2 = %.3f" % (r0 * r0 / 2))
    out["bounds"] = rows


def phase(out):
    alphas = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00]
    Rs = [round(4.3 + 0.15 * k, 2) for k in range(28)]
    curve = []
    for al in alphas:
        flags = [pinches(R, L_for_alpha(al, R), dx=0.05) for R in Rs][::-1]
        ups = sum(1 for i in range(len(flags) - 1) if not flags[i] and flags[i + 1])
        downs = sum(1 for i in range(len(flags) - 1) if flags[i] and not flags[i + 1])
        lo, hi = 4.25, 9.5
        q = None
        if (not pinches(lo, L_for_alpha(al, lo), dx=0.05)
                and pinches(hi, L_for_alpha(al, hi), dx=0.05)):
            while hi - lo > 0.004:
                mid = 0.5 * (lo + hi)
                if pinches(mid, L_for_alpha(al, mid), dx=0.05):
                    hi = mid
                else:
                    lo = mid
            q = r0 / (0.5 * (lo + hi))
        curve.append((al, q))
        print("alpha=%.2f  transitions on %d off %d  q*=%s"
              % (al, ups, downs, "%.4f" % q if q else "none"))
    out["curve"] = curve

    def any_pinch(al):
        return any(pinches(R, L_for_alpha(al, R), dx=0.05)
                   for R in [round(4.4 + 0.2 * k, 2) for k in range(24)])

    lo, hi = 0.2, 1.2
    while hi - lo > 0.02:
        mid = 0.5 * (lo + hi)
        if any_pinch(mid):
            hi = mid
        else:
            lo = mid
    print("alpha_min in (%.2f, %.2f]" % (lo, hi))
    out["alpha_min"] = 0.5 * (lo + hi)


def sensitivity(out):
    Rs = [round(5.28 + 0.01 * k, 2) for k in range(16)]
    res = {}
    print("stopping tolerance")
    for fm in (1e-2, 1e-3, 1e-4, 1e-5):
        fc = [run(R, fmin=fm)["f_cap"] or 0 for R in Rs]
        q = bracket(Rs, fc)[2]
        print("  fmin=%.0e  q*=%s" % (fm, "%.4f" % q if q else "none"))
        res["fmin_%g" % fm] = q
    print("junction smoothing")
    base = None
    for w in (0.0, 0.1, 0.2, 0.5, 1.0):
        fc = [run(R, w=w)["f_cap"] or 0 for R in Rs]
        q = bracket(Rs, fc)[2]
        if q is None:
            continue
        if base is None:
            base = q
        print("  w=%.2f  q*=%.4f  shift %+.4f" % (w, q, q - base))
        res["w_%g" % w] = q
    out["sensitivity"] = res


def figures(out, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 8.5, "savefig.dpi": 600})

    # Figure 1: profiles at the recorded instant
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True)
    for ax, R, tag in ((axs[0], 5.30, "(a)"), (axs[1], 5.45, "(b)")):
        d = run(R)
        x = np.arange(0.0, L0 / 2 + R + 2.0 + DX, DX)
        ax.plot(x, profile(x, R, L0), "k-", lw=1.0)
        ax.plot(d["x"], d["f"], "r-", lw=1.4)
        ax.set_xlabel("$x$")
        ax.set_xlim(0, 19.4)
        ax.set_ylim(0, 6.2)
        ax.set_title("%s  $R=%.2f$" % (tag, R), fontsize=9)
    axs[0].set_ylabel("$f(x)$")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_profiles.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: f_cap at neck collapse vs R
    if "scan" not in out:
        scan(out)
    s = out["scan"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(0, color="0.6", lw=0.5)
    ax.plot(s["R"], s["fcap_dx02"], "o-", ms=3, lw=1, label=r"$\Delta x=0.02$")
    ax.plot(s["R"], s["fcap_dx01"], "s--", ms=3, lw=1, label=r"$\Delta x=0.01$")
    ax.axvline(5.345, color="k", lw=0.6, ls=":")
    ax.set_xlabel("$R$")
    ax.set_ylabel(r"$f_{\rm cap}$ at neck collapse")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_fcap_vs_R.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: phase boundary in (alpha, q)
    if "curve" not in out:
        phase(out)
    al = [c[0] for c in out["curve"] if c[1]]
    qq = [c[1] for c in out["curve"] if c[1]]
    amin = out.get("alpha_min", 0.38)
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    ax.fill_between([amin] + al, 0, [0] + qq, color="0.92", zorder=0)
    ax.axhline(qq[-1], color="r", lw=0.8, ls="--")
    ax.axhline(1 / math.sqrt(2), color="0.4", lw=0.8, ls="--")
    ax.axhline(0.2186, color="k", lw=0.8, ls=":")
    ax.axvline(amin, color="b", lw=0.7, ls=":")
    ax.plot(al, qq, "o-", ms=3.4, lw=1.2)
    ax.set_xlabel(r"$\alpha = x_j/r$")
    ax.set_ylabel(r"$q = r/R$")
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 0.92)
    ax.text(2.55, 0.775, r"$\approx 0.754$", color="r", fontsize=7.5)
    ax.text(2.55, 0.672, r"$1/\sqrt{2}$", color="0.4", fontsize=7.5)
    ax.text(2.55, 0.235, "Angenent bound", fontsize=7.5)
    ax.text(1.55, 0.30, "neckpinch", fontsize=8.5, style="italic")
    ax.text(1.30, 0.845, "no neckpinch", fontsize=8.5, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_phase_boundary.png"), bbox_inches="tight")
    plt.close(fig)


TASKS = {
    "table1": table1, "scan": scan, "sphere": sphere_test, "dt": dt_test,
    "factors": factors, "bounds": bounds, "phase": phase,
    "sensitivity": sensitivity, "figures": figures,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=list(TASKS) + ["all"])
    p.add_argument("--out", default=".", help="output directory")
    args = p.parse_args()

    path = os.path.join(args.out, "results.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    tasks = list(TASKS) if args.task == "all" else [args.task]
    for name in tasks:
        print("--", name)
        if name == "figures":
            figures(out, args.out)
        else:
            TASKS[name](out)
        json.dump(out, open(path, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
