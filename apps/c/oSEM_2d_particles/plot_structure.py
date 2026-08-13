"""In-plane structure of the synthetic field: vectors, vorticity, divergence.

    python3 plot_structure.py                  # last frame -> structure/*.png
    python3 plot_structure.py --all --gif      # every frame, plus a GIF
    python3 plot_structure.py --full           # whole plane, not a zoom window
    python3 plot_structure.py h5files_iso/*.h5 --stats   # aggregate, no maps

Answers "is the STRUCTURE right", where plot_osem_h5.py answers "is the
AMPLITUDE right". Every panel is a derivative, so it needs >= 8 cells per eddy
radius -- the script warns below that. --outdir avoids overwriting between
runs. See the README.
"""

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

BLOCK = "osem_block"
H5DIR = "h5files"          # OSEM_OUTDIR in osem_io.h
OUTDIR = "structure"
GIF = os.path.join(OUTDIR, "structure.gif")

# rms(in-plane divergence) / rms(omega_x) for INCOMPRESSIBLE ISOTROPIC
# turbulence. Continuity forces dv/dy + dw/dz = -du/dx, so with the isotropic
# relations <(du/dx)^2> = A, <(dw/dy)^2> = 2A and <(dw/dy)(dv/dz)> = -A:
#     <div^2>     = A
#     <omega_x^2> = 2A + 2A - 2(-A) = 6A
# hence the ratio is 1/sqrt(6). A field built from blobs that each point in one
# fixed direction has no reason to satisfy this and does not.
INCOMPRESSIBLE_RATIO = 1.0 / np.sqrt(6.0)


def read_frame(path):
    with h5py.File(path, "r") as f:
        NY, NZ = int(f["NY"][0]), int(f["NZ"][0])

        # Grid dats carry a one-cell halo and are stored [z][y]; transpose so
        # rows are y and columns z, as in plot_osem_h5.py.
        def fld(name):
            return f[BLOCK][name][:][1:-1, 1:-1].T

        d = {n: fld(n) for n in ("uprime", "vprime", "wprime")}
        d.update({
            "ymin": float(f["box"][0]), "ymax": float(f["box"][1]),
            "zmin": float(f["box"][2]), "zmax": float(f["box"][3]),
            "ny": NY, "nz": NZ,
            "step": int(f["timestep"][0]),
            "time": float(f["time"][0]),
            "niter": int(f["NITER"][0]),
            "neddy": int(f["neddy"][0]),
            "ex": f["eddy_x"][:], "ey": f["eddy_y"][:], "ez": f["eddy_z"][:],
            "er": f["eddy_r"][:], "sx": f["eddy_sx"][:],
            "xplane": float(f["x_plane"][0]),
        })
    return d


def derivatives(d):
    """omega_x and the in-plane divergence, on the plane's own grid."""
    y = np.linspace(d["ymin"], d["ymax"], d["ny"] + 1)
    z = np.linspace(d["zmin"], d["zmax"], d["nz"] + 1)
    dy, dz = y[1] - y[0], z[1] - z[0]

    v, w = d["vprime"], d["wprime"]
    omx = np.gradient(w, dy, axis=0) - np.gradient(v, dz, axis=1)
    div = np.gradient(v, dy, axis=0) + np.gradient(w, dz, axis=1)
    return y, z, dy, dz, omx, div


def report(d, dy, dz, omx, div):
    r = float(d["er"][0])
    ny_cells, nz_cells = r / dy, r / dz
    print("frame            : step %d of %d,  t = %.3e s"
          % (d["step"], d["niter"], d["time"]))
    print("grid             : %d x %d nodes,  dy = %.3e  dz = %.3e"
          % (d["ny"] + 1, d["nz"] + 1, dy, dz))
    print("eddy radius      : %.5f  ->  %.1f cells in y,  %.1f cells in z"
          % (r, ny_cells, nz_cells))
    if min(ny_cells, nz_cells) < 8.0:
        print("  ** UNDER-RESOLVED for a derivative. omega_x below is largely")
        print("  ** differentiation noise. Re-run with -ny 200 -nz 600.")
    print("rms v', w'       : %.4f  %.4f" % (d["vprime"].std(),
                                             d["wprime"].std()))
    print("rms omega_x      : %.4g" % omx.std())
    print("rms in-plane div : %.4g" % div.std())
    print("  ratio div/omega: %.3f   (incompressible isotropic: %.3f)"
          % (div.std() / omx.std(), INCOMPRESSIBLE_RATIO))
    print("  A field of uniform-direction blobs is not solenoidal, so a ratio")
    print("  well above %.2f is the expected result here, not a defect. The"
          % INCOMPRESSIBLE_RATIO)
    print("  isotropic run (-rst iso) is the clean test bed: the reference")
    print("  value assumes isotropy, which the tabulated RST does not have.")


def stats_mode(files, outdir):
    """Aggregate the divergence diagnostic over many frames. The per-frame
    panel quotes rms(div)/rms(omega_x) from ONE field, which has no
    uncertainty attached and cannot show whether the number drifts."""
    rows = []
    for p in files:
        d = read_frame(p)
        _, _, dy, dz, omx, div = derivatives(d)
        rows.append((d["step"], omx.std(), div.std(), div.std() / omx.std(),
                     d["vprime"].std(), d["wprime"].std()))
    rows.sort()
    a = np.array(rows)
    step, ro, rd, rat, rv, rw = (a[:, k] for k in range(6))

    print("%8s %12s %12s %8s %9s %9s"
          % ("step", "rms omega_x", "rms div", "ratio", "rms v'", "rms w'"))
    for r in rows:
        print("%8d %12.4g %12.4g %8.4f %9.4f %9.4f" % r)

    print("\nover %d frames, steps %d..%d:" % (len(rows), step[0], step[-1]))
    print("   ratio div/omega_x = %.4f +/- %.4f  (sd over frames)"
          % (rat.mean(), rat.std(ddof=1) if len(rat) > 1 else 0.0))
    print("   incompressible isotropic reference = %.4f"
          % INCOMPRESSIBLE_RATIO)
    # Gradient ENERGY goes as the square of an rms ratio; the parentheses are
    # load bearing (** binds tighter than %, so it happens to be right, but
    # nobody should have to work that out).
    print("   -> the field carries %.1fx the dilatational gradient energy an"
          % ((rat.mean() / INCOMPRESSIBLE_RATIO) ** 2))
    print("      incompressible field would. Expected: each eddy contributes a")
    print("      blob pointing in one fixed direction, which is not solenoidal.")
    print("   NOTE the reference assumes ISOTROPY, so quote this from a")
    print("        -rst iso run; under the tabulated RST it is indicative only.")

    # A trend here would mean the field is still developing.
    if len(rows) > 3:
        k = np.polyfit(step, rat, 1)[0] * (step[-1] - step[0])
        print("   drift across the run: %+.4f in ratio (%.1f%% of the mean)"
              % (k, 100.0 * k / rat.mean()))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].plot(step, rat, "o-", color="tab:purple", linewidth=1.4,
                 markersize=3.5)
    axes[0].axhline(INCOMPRESSIBLE_RATIO, color="0.35", linestyle="--",
                    linewidth=1.2, label="incompressible isotropic")
    axes[0].axhline(rat.mean(), color="tab:purple", linestyle=":",
                    linewidth=1.0, label="mean %.3f" % rat.mean())
    axes[0].set_ylabel(r"rms div / rms $\omega_x$")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("divergence diagnostic over %d frames" % len(rows),
                      fontsize=11)

    axes[1].plot(step, rv, "o-", color="tab:green", linewidth=1.4,
                 markersize=3.5, label="rms v'")
    axes[1].plot(step, rw, "o-", color="tab:red", linewidth=1.4,
                 markersize=3.5, label="rms w'")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("rms")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("in-plane amplitude, for stationarity", fontsize=11)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "divergence_stats.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("\nwrote", png)


def window(d, dz, full):
    """Column range to draw. Equal aspect on a 1:4.7 plane gives a useless
    sliver, so default to a z-window as wide as the plane is tall."""
    if full:
        return 0, d["nz"] + 1
    span = d["ymax"] - d["ymin"]
    n = min(int(span / dz), d["nz"] + 1)
    j0 = max(0, (d["nz"] + 1 - n) // 2)
    return j0, j0 + n


def plot_frame(d, path, full, clim, outdir):
    y, z, dy, dz, omx, div = derivatives(d)
    j0, j1 = window(d, dz, full)
    zz, yy = np.meshgrid(z[j0:j1], y)
    v, w = d["vprime"][:, j0:j1], d["wprime"][:, j0:j1]

    # Equal aspect on all three panels, so the figure has to be about as wide
    # as one panel or the whole thing floats in white space.
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 13.2))

    # ---- 1. vectors over vorticity, with the eddies outlined ------------
    ax = axes[0]
    lim = clim["omx"]
    m = ax.pcolormesh(zz, yy, omx[:, j0:j1], cmap="RdBu_r", shading="auto",
                      vmin=-lim, vmax=lim)
    fig.colorbar(m, ax=ax, fraction=0.030, pad=0.006, label=r"$\omega_x$")

    # Thin the arrows to roughly 40 across, whatever the grid.
    st = max(1, (j1 - j0) // 40)
    ax.quiver(zz[::st, ::st], yy[::st, ::st], w[::st, ::st], v[::st, ::st],
              width=0.0022, alpha=0.85)

    # The eddies that actually reach the plane. Each ring of vorticity above
    # should sit on one of these circles: curl of (constant vector) x (blob)
    # peaks where the blob's gradient does, i.e. at the edge, and vanishes at
    # the centre.
    #
    # Only the eddies whose centres are NEAREST the plane are drawn. Every
    # eddy within a radius contributes something, but near the top of the box
    # they overlap into a thicket of circles over a field that is almost zero
    # there (the tabulated stresses die off in the freestream), which hides the
    # one thing the overlay exists to show.
    near = np.abs(d["ex"] - d["xplane"]) < 0.5 * d["er"]
    for i in np.nonzero(near)[0]:
        if z[j0] - d["er"][i] < d["ez"][i] < z[j1 - 1] + d["er"][i]:
            ax.add_patch(Circle((d["ez"][i], d["ey"][i]), d["er"][i],
                                fill=False, linewidth=0.7, edgecolor="0.15",
                                alpha=0.55))
    ax.set_title("$(w', v')$ over vorticity $\\omega_x$; nearest eddies"
                 " outlined", fontsize=10)

    # ---- 2. streamlines --------------------------------------------------
    ax = axes[1]
    sp = ax.streamplot(z[j0:j1], y, w, v, color=np.hypot(v, w),
                       cmap="viridis", density=1.6, linewidth=0.8)
    fig.colorbar(sp.lines, ax=ax, fraction=0.030, pad=0.006,
                 label="in-plane speed")
    ax.set_title("streamlines of the in-plane field", fontsize=10)

    # ---- 3. divergence ---------------------------------------------------
    ax = axes[2]
    lim = clim["div"]
    m = ax.pcolormesh(zz, yy, div[:, j0:j1], cmap="PuOr_r", shading="auto",
                      vmin=-lim, vmax=lim)
    fig.colorbar(m, ax=ax, fraction=0.030, pad=0.006,
                 label=r"$\partial v'/\partial y + \partial w'/\partial z$")
    ax.set_title("in-plane divergence -- rms ratio to $\\omega_x$ %.2f"
                 " (incompressible: %.2f)"
                 % (div.std() / omx.std(), INCOMPRESSIBLE_RATIO), fontsize=10)

    for ax in axes:
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(z[j0], z[j1 - 1])
        ax.set_ylim(d["ymin"], d["ymax"])

    fig.suptitle("in-plane structure -- step %d / %d,  t = %.3e s"
                 % (d["step"], d["niter"], d["time"]), fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.982])

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir,
                       os.path.basename(path).replace(".h5", "_structure.png"))
    fig.savefig(png, dpi=105)
    plt.close(fig)
    return png


def main():
    argv = sys.argv[1:]
    outdir = OUTDIR
    if "--outdir" in argv:
        k = argv.index("--outdir")
        if k + 1 >= len(argv):
            sys.exit("--outdir needs a directory name")
        outdir = argv[k + 1]
        del argv[k:k + 2]
    flags = [a for a in argv if a.startswith("--")]
    named = [a for a in argv if not a.startswith("--")]
    full = "--full" in flags

    files = named or sorted(glob.glob(os.path.join(H5DIR,
                                                   "osem_output_[0-9]*.h5")))
    if not files:
        files = sorted(glob.glob("osem_output_[0-9]*.h5"))
    if not files:
        sys.exit("no osem_output_*.h5 found in %s/ or %s"
                 % (H5DIR, os.getcwd()))
    # --stats is aggregate-only: no per-frame pictures, one summary figure.
    if "--stats" in flags:
        if len(files) < 2:
            sys.exit("--stats wants several frames; got %d" % len(files))
        stats_mode(files, outdir)
        return

    if "--all" not in flags and not named:
        files = files[-1:]          # the last frame is the interesting one

    frames = [read_frame(p) for p in files]

    # One colour scale over every frame, or the animation pulses with its own
    # autoscale and hides the evolution -- same reasoning as plot_osem_h5.py.
    om, dv = [], []
    for d in frames:
        _, _, _, _, o, g = derivatives(d)
        om.append(3.0 * o.std())
        dv.append(3.0 * g.std())
    clim = {"omx": max(om), "div": max(dv)}

    pngs = []
    for path, d in zip(files, frames):
        y, z, dy, dz, omx, div = derivatives(d)
        report(d, dy, dz, omx, div)
        pngs.append(plot_frame(d, path, full, clim, outdir))
        print("wrote", pngs[-1], "\n")

    if "--gif" in flags and len(pngs) > 1:
        from PIL import Image

        gif = os.path.join(outdir, os.path.basename(GIF))
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                for p in pngs]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=140,
                     loop=0)
        print("wrote", gif, "(%d frames)" % len(imgs))


if __name__ == "__main__":
    main()
