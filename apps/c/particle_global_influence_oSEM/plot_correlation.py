"""Two-point spatial correlation of the inlet field -- the SEM validation plot.

    python3 plot_correlation.py h5files_tbl/osem_output_*.h5 --outdir corr_tbl
    python3 plot_correlation.py --yref 0.0045
    python3 plot_correlation.py h5files_tbl/*.h5 --split    # has it converged?

--split is the acceptance test, and it separates two questions that a single
number cannot:

  CHRONOLOGICAL halves (first N/2 frames vs last N/2) move if the field is
  still DRIFTING -- a spin-up transient, or a slow migration of the eddy
  population. Everything after such a transient would be measuring the wrong
  thing.

  INTERLEAVED halves (odd frames vs even frames) span the same time range, so
  they cannot see drift and move only with SAMPLING NOISE.

Comparing the two is the point. Interleaved alone gives the error bar; if
chronological moves substantially more than interleaved, the extra motion is
drift, not noise, and a longer run would be averaging over a transient rather
than converging.

This is the figure that answers "does the field actually contain structures of
the size I asked for". plot_osem_h5.py checks the AMPLITUDE (rms against the
tabulated stresses) and plot_structure.py shows one instantaneous field; this
one is an ENSEMBLE statistic and needs many frames.

    R_uu(dy, dz) = <u'(yref, z) u'(yref+dy, z+dz)> / sqrt(<u'^2(yref)><u'^2>)

Contoured, it is a set of nested closed curves centred on the reference point,
and the integral length scale L = \\int R d(sep) comes straight out of it.

Kept OUT of plot_structure.py deliberately. That script maps one frame to one
picture; this one reduces every frame to a single figure, has no per-frame
output and no GIF, and carries the analytic machinery below. Two ~250-line
scripts read better than one 500-line script with two modes.

WHY THERE IS AN ANALYTIC CURVE ON THE PLOT
------------------------------------------
A measured correlation that merely looks plausible proves nothing, so the
prediction is drawn with it. It is exact, not a fit, and it follows from the
kernel:

    u' = a11 * P,   v' = a21 * P + a22 * Q,   w' = a33 * T

where P, Q, T are the three sign-weighted sums over the shape function (see
KerComputeFluct -- a31 = a32 = 0). The a_ij are evaluated at the NODE, so they
factor straight out of the sum. For two points on the SAME row they are
identical and cancel in the normalisation, which gives an exact prediction:

    R_uu = R_vv = R_ww = C(|sep|) / C(0)      along dz at dy = 0

with C the 3-D autocorrelation of the shape function. So all three components
must collapse onto ONE curve, and that curve is computable. If they collapse
onto it, the SEM machinery is doing what it was told to; any quarrel left is
with the shape function, which is a modelling choice, not an implementation.
If they miss it, there is a bug, and where they miss says something about it.

C depends only on |sep| because the shape function is spherical, so the
contours must come out CIRCULAR. Elongated contours would be a real boundary
layer; the single fixed eddy_radius here cannot produce them. That is a
limitation worth being able to show rather than assert.

SAMPLING
--------
Frames must be at least one flow-through apart -- (x_max-x_min)/(u0*dt), 348
steps as shipped -- or the same eddies are being counted repeatedly and the
error bars are fiction. The script prints the spacing it was given and warns.
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
    TRUNCATED at |r| < eddy_radius -- i.e. cut at one standard deviation, which
    is severe enough that the truncation, not the Gaussian, sets the width. So
    this is integrated numerically rather than taken from the textbook Gaussian
    result (which would be a Gaussian of sigma*sqrt(2) and is wrong here).

    Spherical symmetry reduces the 3-D overlap integral to two dimensions, in
    cylindrical coordinates about the separation axis:

        C(s) = 2*pi * \\int rho drho \\int dzeta f(r1) f(r2)
        r1 = sqrt(rho^2 + zeta^2),  r2 = sqrt(rho^2 + (zeta-s)^2)
    """
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
    circular. z is not periodic here -- the eddies reflect off the box faces --
    so the transform is zero padded and each lag is normalised by its own
    overlap count by the caller."""
    n = a.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft, axis=-1)
    B = np.fft.rfft(b, nfft, axis=-1)
    c = np.fft.irfft(np.conj(A) * B, nfft, axis=-1)
    # c[..., l] holds lag +l; negative lags live at the far end.
    return np.concatenate([c[..., -maxlag:], c[..., :maxlag + 1]], axis=-1)


def accumulate(files, iref, j0, j1, maxlag):
    """Two passes: the per-row mean first, then the correlation about it.

    Streaming rather than loading every frame, because a 200k-step run at
    201x601 is 1.6 GB of frames and this has to survive that.
    """
    nf = len(files)
    m = frame_meta(files[0])
    ny = m["ny"]

    # -- pass 1: <u'> per row. It is ~0 by construction; measured, not assumed.
    tot = {n: np.zeros(ny + 1) for n in COMPS}
    for p in files:
        d = frame_fields(p, j0, j1)
        for n in COMPS:
            tot[n] += d[n].mean(axis=1)
    mean = {n: tot[n] / nf for n in COMPS}

    # -- pass 2: raw products about that mean
    nlag = 2 * maxlag + 1
    S = {n: np.zeros((ny + 1, nlag)) for n in COMPS}   # row vs reference row
    V = {n: np.zeros(ny + 1) for n in COMPS}           # per-row variance
    for p in files:
        d = frame_fields(p, j0, j1)
        for n in COMPS:
            a = d[n] - mean[n][:, None]
            S[n] += corr_lags(np.broadcast_to(a[iref], a.shape).copy(), a,
                              maxlag)
            V[n] += (a * a).sum(axis=1)

    nz_used = j1 - j0
    lags = np.arange(-maxlag, maxlag + 1)
    npair = (nz_used - np.abs(lags)).astype(float) * nf
    nsamp = float(nz_used * nf)

    R = {}
    for n in COMPS:
        cov = S[n] / npair[None, :]
        var = V[n] / nsamp
        # The wall row is exactly zero -- a11 = sqrt(R11) and the table takes
        # R11 to 0 at y = 0, so u' vanishes there identically and the
        # normalisation is 0/0. Leave those rows undefined rather than let a
        # warning through; the contour plot simply omits them.
        den = np.sqrt(var[iref] * var)
        good = den > 1e-12 * max(den.max(), 1e-30)
        R[n] = np.full_like(cov, np.nan)
        R[n][good] = cov[good] / den[good, None]
    return R, lags, mean, {n: np.sqrt(V[n] / nsamp) for n in COMPS}


def scales_of(files, iref, j0, j1, maxlag, sep_z, half):
    """Integral length scale per component for one subset of frames."""
    R, _, _, _ = accumulate(files, iref, j0, j1, maxlag)
    return ({n: integral_scale(sep_z[half:], R[n][iref, half:])
             for n in COMPS}, R)


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

    # Trim the padded strips off both z ends. The grid spans the EDDY box, so
    # the outermost r_max in z has eddies on one side only and a variance
    # deficit to match (Defect 2 in UNDERSTANDING_oSEM.md). Correlations from
    # there are of the deficit, not of the flow.
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

    R, lags, mean, rms = accumulate(files, iref, j0, j1, maxlag)
    sep_z = lags * dz

    # ---- diagnostics -----------------------------------------------------
    print("\n<u'> per row at the reference (should be ~0 next to the rms):")
    for n in COMPS:
        print("   %-3s mean %+11.4e   rms %10.4f   ratio %.4f"
              % (LABEL[n], mean[n][iref], rms[n][iref],
                 abs(mean[n][iref]) / max(rms[n][iref], 1e-30)))

    # R must be even in dz: no direction in z is special. Tested one radius
    # OFF the reference row, not on it -- a row's correlation with ITSELF is
    # exactly even by construction (sum_z a[z]a[z+l] is the same sum as
    # sum_z a[z]a[z-l]), so measuring it there always returns 0 and tests
    # nothing. Off-row it is a real statement about the sample count.
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
            k = len(files) // 2
            sets = {
                "chronological": (files[:k], files[k:]),
                "interleaved": (files[0::2], files[1::2]),
            }
            L = {}
            for mode, (fa, fb) in sets.items():
                La, Ra = scales_of(fa, iref, j0, j1, maxlag, sep_z, half)
                Lb, Rb = scales_of(fb, iref, j0, j1, maxlag, sep_z, half)
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

            # Two independent halves of equal size: an estimate of the FULL
            # set's standard error is |a-b|/2. Taken from the interleaved
            # split, which cannot be contaminated by drift.
            se = float(np.mean(din)) / 2.0
            print("\n   SE(full-set L) ~ %.4f radii (interleaved half-split)"
                  % se)
            print("   quote L = %.3f +/- %.3f radii against analytic %.3f"
                  % (integral_scale(sep_z[half:],
                                    R["uprime"][iref, half:]) / er, se,
                     L_an / er))

            ratio = float(np.mean(dch)) / max(float(np.mean(din)), 1e-12)
            print("\n   drift indicator = chronological/interleaved = %.2f"
                  % ratio)

            # Both halves have to contain enough DECORRELATED realisations for
            # either difference to mean anything, and the indicator is a ratio
            # of two noisy numbers, so it degrades faster than either. With too
            # few, it lands anywhere -- including below 1, which no real drift
            # can produce and which is therefore a tell that the test is
            # under-sampled rather than a finding.
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
                      % (10, se / np.sqrt(10.0)))

            # u' and v' are NOT independent estimators -- both carry the same
            # sign-sum P (u' = a11*P, v' = a21*P + a22*Q), so the spread across
            # the three components understates the noise. w' is built from an
            # independent sign field, so u' vs w' is the honest pairing. The
            # split-half number above avoids the issue entirely.
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
