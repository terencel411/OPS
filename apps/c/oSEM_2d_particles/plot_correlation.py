"""Two-point spatial correlation of the inlet field -- the SEM validation plot.

    python3 plot_correlation.py h5files_tbl/osem_output_*.h5 --outdir corr_tbl
    python3 plot_correlation.py --yref 0.0045
    python3 plot_correlation.py h5files_tbl/*.h5 --split    # has it converged?

Answers "does the field contain structures of the size I asked for", over many
frames. An exact analytic prediction is drawn with the measurement; --split
separates drift from sampling noise. Frames must be >= 1 flow-through apart.
"""

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLOCK = "osem_block"
H5DIR = "h5files"
OUTDIR = "correlation"
COMPS = ("uprime", "vprime", "wprime")
LABEL = {"uprime": "u'", "vprime": "v'", "wprime": "w'"}

# r_max / eddy_radius as the app ships it: r_max = 0.41*delta, eddy_radius =
# 0.2*delta. Used only to trim the padded strips off the z ends (see below);
# override with --trim if those constants ever change.
RMAX_OVER_R = 0.41 / 0.2


# ---------------------------------------------------------------- analytic

def shape_autocorr(nsep=241, nrho=400, nzeta=1200):
    """C(s)/C(0) for the kernel's shape function, s in units of the radius.
    The shape is a Gaussian of standard deviation sigma = eddy_radius, HARD
    TRUNCATED at |r| < eddy_radius -- i.e."""
    def f(r):
        return np.where(r < 1.0, np.exp(-0.5 * r * r), 0.0)

    rho = np.linspace(0.0, 1.0, nrho)
    zeta = np.linspace(-1.0, 3.0, nzeta)
    R, Z = np.meshgrid(rho, zeta, indexing="ij")
    r1 = np.hypot(R, Z)
    f1 = f(r1) * R                      # the rho of "rho drho" folded in

    sep = np.linspace(0.0, 2.0, nsep)
    C = np.empty_like(sep)
    for i, s in enumerate(sep):
        C[i] = np.trapezoid(np.trapezoid(f1 * f(np.hypot(R, Z - s)), zeta,
                                         axis=1), rho)
    return sep, C / C[0]


# ---------------------------------------------------------------- frames

def frame_meta(path):
    with h5py.File(path, "r") as f:
        return {
            "ny": int(f["NY"][0]), "nz": int(f["NZ"][0]),
            "ymin": float(f["box"][0]), "ymax": float(f["box"][1]),
            "zmin": float(f["box"][2]), "zmax": float(f["box"][3]),
            "er": float(f["eddy_r"][0]),
            "step": int(f["timestep"][0]),
        }


def frame_fields(path, j0, j1):
    """The three components, halo stripped, transposed to [y][z], z trimmed."""
    with h5py.File(path, "r") as f:
        return {n: f[BLOCK][n][:][1:-1, 1:-1].T[:, j0:j1] for n in COMPS}


# ---------------------------------------------------------------- estimator

def corr_lags(a, b, maxlag):
    """sum_z a[y,z] b[y,z+lag] for lag in [-maxlag, maxlag], LINEAR not
    circular. z is not periodic here -- the eddies reflect off the box faces
    -- so the transform is zero padded and each lag is normalised by its own
    overlap count by the caller."""
    n = a.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft, axis=-1)
    B = np.fft.rfft(b, nfft, axis=-1)
    c = np.fft.irfft(np.conj(A) * B, nfft, axis=-1)
    # c[..., l] holds lag +l; negative lags live at the far end.
    return np.concatenate([c[..., -maxlag:], c[..., :maxlag + 1]], axis=-1)


def partials(files, iref, j0, j1, maxlag):
    """Per-frame RAW sums, read once. Everything else is arithmetic on these."""
    P, A, Q = {}, {}, {}
    for n in COMPS:
        P[n], A[n], Q[n] = [], [], []
    for p in files:
        d = frame_fields(p, j0, j1)
        for n in COMPS:
            a = d[n]
            P[n].append(corr_lags(np.broadcast_to(a[iref], a.shape).copy(), a,
                                  maxlag))
            A[n].append(a.sum(axis=1))
            Q[n].append((a * a).sum(axis=1))
    return ({n: np.stack(P[n]) for n in COMPS},
            {n: np.stack(A[n]) for n in COMPS},
            {n: np.stack(Q[n]) for n in COMPS})


def combine(par, idx, iref, nz_used, maxlag):
    """R for a subset of frames, from the stored partial sums."""
    P, A, Q = par
    nf = len(idx)
    lags = np.arange(-maxlag, maxlag + 1)
    npair = (nz_used - np.abs(lags)).astype(float) * nf
    nsamp = float(nz_used * nf)

    R, mean, rms = {}, {}, {}
    for n in COMPS:
        Ps = P[n][idx].sum(axis=0)
        mu = A[n][idx].sum(axis=0) / nsamp
        var = Q[n][idx].sum(axis=0) / nsamp - mu * mu
        # E[ab] - E[a]E[b]. The mean correction uses the whole-row means rather
        # than the means of each lag's overlap window; with |<u'>|/rms ~ 0.03
        # that difference is far below the sampling noise.
        cov = Ps / npair[None, :] - mu[iref] * mu[:, None]

        # The wall row is exactly zero: R11 -> 0 there, so u' vanishes and the
        # normalisation is 0/0. Left undefined; the contour plot omits it.
        den = np.sqrt(np.maximum(var[iref] * var, 0.0))
        good = den > 1e-12 * max(den.max(), 1e-30)
        R[n] = np.full_like(cov, np.nan)
        R[n][good] = cov[good] / den[good, None]
        mean[n] = mu
        rms[n] = np.sqrt(np.maximum(var, 0.0))
    return R, lags, mean, rms


def scales_of(par, idx, iref, nz_used, maxlag, sep_z, half):
    """Integral length scale per component for one subset of frames."""
    R, _, _, _ = combine(par, idx, iref, nz_used, maxlag)
    return ({n: integral_scale(sep_z[half:], R[n][iref, half:])
             for n in COMPS}, R)


def jackknife_se(par, iref, nz_used, maxlag, sep_z, half, comp="uprime"):
    """Leave-one-frame-out standard error of L. The half-split SE it replaced
    used one difference of two numbers, so the error bar itself carried ~100%
    uncertainty."""
    F = len(par[0][COMPS[0]])
    allidx = np.arange(F)
    L = np.empty(F)
    for k in range(F):
        R, _, _, _ = combine(par, allidx[allidx != k], iref, nz_used, maxlag)
        L[k] = integral_scale(sep_z[half:], R[comp][iref, half:])
    return float(np.sqrt((F - 1.0) / F * np.sum((L - L.mean()) ** 2))), L


def integral_scale(sep, r):
    """\\int_0^{first zero} R d(sep). Integrating past the first zero crossing
    accumulates noise and, for a compactly supported blob, negative side lobes
    that are not part of the scale."""
    k = np.argmax(r <= 0.0) if np.any(r <= 0.0) else len(r)
    if k < 2:
        return 0.0
    return float(np.trapezoid(r[:k], sep[:k]))


# ---------------------------------------------------------------- main

def main():
    argv = sys.argv[1:]

    def opt(name, default, cast):
        if name in argv:
            k = argv.index(name)
            if k + 1 >= len(argv):
                sys.exit("%s needs a value" % name)
            v = cast(argv[k + 1])
            del argv[k:k + 2]
            return v
        return default

    outdir = opt("--outdir", OUTDIR, str)
    yref_want = opt("--yref", None, float)
    trim = opt("--trim", RMAX_OVER_R, float)

    do_split = "--split" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        files = sorted(glob.glob(os.path.join(H5DIR, "osem_output_[0-9]*.h5")))
    if not files:
        sys.exit("no osem_output_*.h5 found in %s/ or %s"
                 % (H5DIR, os.getcwd()))
    # Chronological order matters for --split, and a shell glob is only in
    # step order by accident of the zero padding. Sort on the timestep itself.
    files = sorted(files, key=lambda p: frame_meta(p)["step"])

    m = frame_meta(files[0])
    y = np.linspace(m["ymin"], m["ymax"], m["ny"] + 1)
    z = np.linspace(m["zmin"], m["zmax"], m["nz"] + 1)
    dy, dz = y[1] - y[0], z[1] - z[0]
    er = m["er"]

    # Trim the padded strips off both z ends: the grid spans the eddy box, so
    # the outermost r_max has eddies on one side only and a variance deficit to
    # match (Defect 2 in UNDERSTANDING_oSEM.md).
    ntrim = int(np.ceil(trim * er / dz))
    j0, j1 = ntrim, m["nz"] + 1 - ntrim
    if j1 - j0 < 32:
        sys.exit("z range collapses after trimming %d cells a side; "
                 "lower --trim or raise -nz" % ntrim)

    if yref_want is None:
        yref_want = 0.5 * (m["ymax"] - er)      # mid height, clear of the pad
    iref = int(np.argmin(np.abs(y - yref_want)))

    maxlag = min(int(3.0 * er / dz), (j1 - j0) // 3)
    ilo = max(0, iref - int(3.0 * er / dy))
    ihi = min(m["ny"], iref + int(3.0 * er / dy))

    steps = [frame_meta(p)["step"] for p in files]
    gap = int(np.min(np.diff(sorted(steps)))) if len(steps) > 1 else 0

    print("frames           : %d   steps %d..%d, spacing %d"
          % (len(files), min(steps), max(steps), gap))
    print("eddy radius      : %.5f  ->  %.1f cells in y, %.1f in z"
          % (er, er / dy, er / dz))
    print("z trimmed        : %d cells (%.2f radii) each side, %d used"
          % (ntrim, trim, j1 - j0))
    print("reference row    : y = %.5f  (row %d of %d)" % (y[iref], iref,
                                                           m["ny"]))
    print("lags             : +/-%d cells in z, rows %d..%d in y"
          % (maxlag, ilo, ihi))

    nz_used = j1 - j0
    par = partials(files, iref, j0, j1, maxlag)
    R, lags, mean, rms = combine(par, np.arange(len(files)), iref, nz_used,
                                 maxlag)
    sep_z = lags * dz

    # ---- diagnostics -----------------------------------------------------
    print("\n<u'> per row at the reference (should be ~0 next to the rms):")
    for n in COMPS:
        print("   %-3s mean %+11.4e   rms %10.4f   ratio %.4f"
              % (LABEL[n], mean[n][iref], rms[n][iref],
                 abs(mean[n][iref]) / max(rms[n][iref], 1e-30)))

    # R must be even in dz. Tested one radius off the reference row, not on it:
    # a row's correlation with itself is even by construction, so measuring it
    # there always returns 0 and tests nothing.
    half = maxlag
    isym = min(iref + max(1, int(er / dy)), m["ny"])
    asym = max(float(np.nanmax(np.abs(R[n][isym, :half][::-1]
                                      - R[n][isym, half + 1:])))
               for n in COMPS)
    print("\nasymmetry of R in +/-dz : %.4f at row %d (%.2f radii off the"
          " reference)" % (asym, isym, (y[isym] - y[iref]) / er))
    print("   0 = perfectly even, which z homogeneity requires; a large value"
          " means too few samples")

    sep_an, R_an = shape_autocorr()
    L_an = integral_scale(sep_an * er, R_an)

    print("\nintegral length scale L = int R d(sep), along dz at dy = 0:")
    for n in COMPS:
        cut = R[n][iref, half:]                     # dz >= 0 side
        print("   %-3s L = %.6f  = %.3f eddy radii" % (LABEL[n],
              integral_scale(sep_z[half:], cut),
              integral_scale(sep_z[half:], cut) / er))
    print("   analytic (shape function autocorrelation)")
    print("       L = %.6f  = %.3f eddy radii" % (L_an, L_an / er))
    # The measurement integrates a curve sampled at the grid's lags only.
    # Resampling the analytic curve the same way separates "the field is wrong"
    # from "the grid is coarse". At 3.8 cells per radius the bias is +0.3%.
    L_an_g = integral_scale(sep_z[half:],
                            np.interp(sep_z[half:] / er, sep_an, R_an,
                                      right=0.0))
    print("       resampled onto this grid's lags: %.3f radii"
          " (discretisation bias %+.1f%%)"
          % (L_an_g / er, 100.0 * (L_an_g - L_an) / L_an))
    print("   eddy_radius = %.6f,  support diameter 2r = %.6f" % (er, 2 * er))

    if gap and gap < 348:
        print("\n** frames are %d steps apart; one flow-through is 348 steps."
              % gap)
        print("** They are not independent samples -- the curves are fine but")
        print("** the effective sample count is lower than it looks.")

    # ---- split-half: has it converged, and is it stationary? -------------
    split_curves = None
    if do_split:
        if len(files) < 4:
            print("\n--split needs at least 4 frames; got %d" % len(files))
        else:
            F = len(files)
            ix = np.arange(F)
            k = F // 2
            sets = {
                "chronological": (ix[:k], ix[k:]),
                "interleaved": (ix[0::2], ix[1::2]),
            }
            L = {}
            for mode, (ia, ib) in sets.items():
                La, Ra = scales_of(par, ia, iref, nz_used, maxlag, sep_z, half)
                Lb, Rb = scales_of(par, ib, iref, nz_used, maxlag, sep_z, half)
                L[mode] = (La, Lb)
                if mode == "chronological":
                    split_curves = (Ra, Rb)   # drawn on the right panel

            print("\nsplit-half test, L in eddy radii:")
            print("   %-4s %19s %19s" % ("", "chronological", "interleaved"))
            print("   %-4s %8s %8s %6s  %8s %8s %6s"
                  % ("", "first", "second", "|d|", "odd", "even", "|d|"))
            dch, din = [], []
            for n in COMPS:
                a1, b1 = (L["chronological"][i][n] / er for i in (0, 1))
                a2, b2 = (L["interleaved"][i][n] / er for i in (0, 1))
                dch.append(abs(a1 - b1))
                din.append(abs(a2 - b2))
                print("   %-4s %8.4f %8.4f %6.4f  %8.4f %8.4f %6.4f"
                      % (LABEL[n], a1, b1, dch[-1], a2, b2, din[-1]))

            # The error bar comes from the jackknife, not the half-split above:
            # a half-split SE is one difference and is itself ~100% uncertain.
            # The split table is kept because its comparison detects drift.
            se, Ljk = jackknife_se(par, iref, nz_used, maxlag, sep_z, half)
            Lfull = integral_scale(sep_z[half:], R["uprime"][iref, half:])
            print("\n   jackknife SE(L)  = %.4f radii  (%d leave-one-out"
                  " estimates)" % (se / er, F))
            print("   half-split SE(L) = %.4f radii  (one difference, ~100%%"
                  " uncertain -- do not quote this)"
                  % (float(np.mean(din)) / 2.0))
            print("   quote L = %.3f +/- %.3f radii against analytic %.3f"
                  % (Lfull / er, se / er, L_an / er))
            nsig = abs(Lfull - L_an) / max(se, 1e-30)
            print("   -> %.1f sigma from the analytic prediction%s"
                  % (nsig, "" if nsig < 2.0 else "   ** worth chasing **"))

            ratio = float(np.mean(dch)) / max(float(np.mean(din)), 1e-12)
            print("\n   drift indicator = chronological/interleaved = %.2f"
                  % ratio)

            # Both halves need enough decorrelated realisations, and this is a
            # ratio of two noisy numbers, so it degrades faster than either.
            # Under-sampled it lands anywhere, including below 1.
            span = max(steps) - min(steps)
            nind_half = span / 2.0 / 348.0
            if nind_half < 8.0:
                print("   ** only ~%.1f decorrelated realisations per half."
                      % nind_half)
                print("   ** The indicator is NOT usable here -- read neither")
                print("   ** drift nor stationarity from it. Needs a run of")
                print("   ** at least %d steps; Stage 2's 20000 gives ~29."
                      % int(2 * 8 * 348))
            elif ratio > 2.5:
                print("   ** the chronological halves move well beyond the")
                print("   ** noise floor: the field is still DRIFTING. A")
                print("   ** longer run would average over a transient --")
                print("   ** find where it settles and discard what precedes.")
            else:
                print("   stationary: the chronological halves move no more")
                print("   than the noise, so a longer run buys sample count")
                print("   rather than a different answer. Error scales as")
                print("   1/sqrt(frames): %dx the frames -> %.4f radii."
                      % (10, se / er / np.sqrt(10.0)))

            # u' and v' are not independent estimators: both carry the sign-sum
            # P, so the three-component spread understates the noise. w' uses an
            # independent sign field, so u' vs w' is the honest pairing.
            print("   (u' and v' share the P sign-sum, so the u/v/w spread is")
            print("    not an independent noise estimate; this one is.)")

    # ---- figure ----------------------------------------------------------
    # constrained, not tight: the left panel is equal-aspect and tight_layout
    # cannot place it.
    fig = plt.figure(figsize=(12.5, 5.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    # the sketch: nested contours about the reference point
    ax = fig.add_subplot(gs[0, 0])
    ZZ, YY = np.meshgrid(sep_z, y[ilo:ihi + 1] - y[iref])
    M = R["uprime"][ilo:ihi + 1, :]
    levels = np.linspace(-0.2, 1.0, 25)
    cf = ax.contourf(ZZ, YY, M, levels=levels, cmap="RdYlBu_r", extend="both")
    cs = ax.contour(ZZ, YY, M, levels=[0.1, 0.25, 0.5, 0.75],
                    colors="k", linewidths=0.8)
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    # the shape function's own support, for scale
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(er * np.cos(th), er * np.sin(th), "k--", linewidth=1.0,
            label="eddy radius")
    ax.plot(0, 0, "k+", markersize=9)
    fig.colorbar(cf, ax=ax, fraction=0.040, pad=0.02, label="$R_{uu}$")
    ax.set_xlabel(r"$\Delta z$"); ax.set_ylabel(r"$\Delta y$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("$R_{uu}(\\Delta y, \\Delta z)$ about $y = %.5f$,"
                 " %d frames" % (y[iref], len(files)), fontsize=10)

    # the collapse test
    ax = fig.add_subplot(gs[0, 1])
    # The two chronological halves, faint and behind. If the field is
    # stationary they are indistinguishable from the full curve, and the width
    # of the band they make IS the error bar -- drawn rather than asserted.
    if split_curves is not None:
        for Rh, lab in zip(split_curves, ("first half", "second half")):
            ax.plot(sep_z[half:] / er, Rh["uprime"][iref, half:], color="0.55",
                    linewidth=1.0, alpha=0.85,
                    label=lab + " (u')" if lab == "first half" else None)
    for n, col in zip(COMPS, ("tab:blue", "tab:green", "tab:red")):
        ax.plot(sep_z[half:] / er, R[n][iref, half:], color=col, linewidth=1.6,
                label="measured " + LABEL[n])
    ax.plot(sep_an, R_an, "k--", linewidth=1.8,
            label="analytic (shape autocorrelation)")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(1.0, color="0.7", linewidth=0.8, linestyle=":")
    ax.text(1.02, 0.92, "eddy radius", fontsize=8, color="0.4")
    ax.set_xlabel(r"$\Delta z$ / eddy radius")
    ax.set_ylabel("$R$")
    ax.set_xlim(0, min(3.0, sep_z[-1] / er))
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_title("all three components must collapse onto the analytic curve",
                 fontsize=10)

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "correlation.png")
    fig.savefig(png, dpi=115)
    plt.close(fig)
    print("\nwrote", png)


if __name__ == "__main__":
    main()
