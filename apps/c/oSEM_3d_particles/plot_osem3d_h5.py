"""Plot the eddy cloud written by sem3d_io.h.

Each .h5 holds the eddy-box coordinate dat under the block group, the particle
dats as flat arrays, and the run constants.

    python3 plot_osem3d_h5.py                     # every osem3d_eddies*.h5
    python3 plot_osem3d_h5.py osem3d_eddies_000000.h5

Four panels:
  1. the 3D cloud, coloured by eps_x;
  2. the (y, z) plane with each eddy drawn at its own radius, which is where
     gaps and clustering are easiest to see;
  3. the (x, y) side view, showing how thin the slab is: two eddy radii in x;
  4. the wall-normal distribution, which should be flat across the box.

    ./oSEM_3d_particles_dev_seq
    python3 plot_osem3d_h5.py
"""

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

OUTDIR = "frames"


def read_frame(path):
    with h5py.File(path, "r") as f:
        n = int(f["no_particles"][0])
        pos = f["eddy_pos"][:].reshape(n, 3)
        eps = f["eddy_eps"][:].reshape(n, 3)
        rad = f["eddy_r"][:]
        ids = f["eddy_id"][:]
        out = dict(
            n=n,
            timestep=int(f["timestep"][0]),
            domain=f["domain"][:],            # xlo xhi ylo yhi zlo zhi
            delta=float(f["delta"][0]),
            radius=float(f["radius"][0]),
            span_z=float(f["span_z"][0]),
        )
    # Seeding keeps the global eddy index, so sorting makes runs at different
    # rank counts directly comparable.
    o = np.argsort(ids)
    out.update(pos=pos[o], eps=eps[o], rad=rad[o], ids=ids[o])
    return out


def disc_coverage(fr, ny=64, nz=128):
    """Fraction of the (y,z) plane over [0,delta]x[0,span_z] within one radius
    of some eddy. A crude but readable measure of how well the box is filled."""
    y = np.linspace(0.0, fr["delta"], ny)[:, None, None]
    z = np.linspace(0.0, fr["span_z"], nz)[None, :, None]
    py = fr["pos"][:, 1][None, None, :]
    pz = fr["pos"][:, 2][None, None, :]
    d2 = (y - py) ** 2 + (z - pz) ** 2
    return 100.0 * np.mean(np.any(d2 < fr["radius"] ** 2, axis=2))


def draw_box(ax, domain):
    xlo, xhi, ylo, yhi, zlo, zhi = domain
    c = np.array([[xlo, ylo, zlo], [xhi, ylo, zlo], [xhi, yhi, zlo],
                  [xlo, yhi, zlo], [xlo, ylo, zhi], [xhi, ylo, zhi],
                  [xhi, yhi, zhi], [xlo, yhi, zhi]])
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(c[a], c[b]), color="0.6", lw=0.6)


def plot_frame(fr, path):
    xlo, xhi, ylo, yhi, zlo, zhi = fr["domain"]
    pos, eps = fr["pos"], fr["eps"]
    colour = np.where(eps[:, 0] > 0, "#c1272d", "#0b6fa4")

    cov = disc_coverage(fr)
    npos = int(np.sum(eps[:, 0] > 0))

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "oSEM_3d eddy initialisation as OPS particles  --  %d eddies\n"
        "disc coverage %.1f%%   |   $\\epsilon_x$ = +1 for %d of %d"
        % (fr["n"], cov, npos, fr["n"]),
        fontsize=12)

    # --- 1. the 3D cloud ------------------------------------------------
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colour, s=14, alpha=0.85,
               edgecolors="none")
    draw_box(ax, fr["domain"])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # True aspect would squash the slab to a line: 2 radii in x against 40+ in z.
    ax.set_title("eddy cloud in its box\n"
                 "red $\\epsilon_x=+1$, blue $-1$ (axes not to scale)",
                 fontsize=10)

    # --- 2. the inlet plane (y, z) --------------------------------------
    ax = fig.add_subplot(2, 2, 2)
    for (z, y, r, c) in zip(pos[:, 2], pos[:, 1], fr["rad"], colour):
        ax.add_patch(plt.Circle((z, y), r, color=c, alpha=0.20, lw=0))
    ax.scatter(pos[:, 2], pos[:, 1], c=colour, s=8, edgecolors="none")
    # The inlet plane the box is built around: [0,delta] x [0,span_z].
    ax.add_patch(plt.Rectangle((0.0, 0.0), fr["span_z"], fr["delta"],
                               fill=False, ec="k", lw=1.4, ls="-"))
    ax.text(0.3, fr["delta"] - 0.6, "inlet plane", fontsize=8, va="top")
    ax.set_xlim(zlo, zhi); ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xlabel("z (spanwise)"); ax.set_ylabel("y (wall normal)")
    ax.set_title("circles at one eddy radius\n"
                 "white inside the box = gaps in the eddy field", fontsize=10)

    # --- 3. the slab from the side (x, y) -------------------------------
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(pos[:, 0], pos[:, 1], c=colour, s=12, edgecolors="none")
    ax.axvline(0.0, color="k", lw=1.0, ls=":")
    ax.text(0.05, yhi, " inlet plane", va="top", fontsize=9)
    ax.axhline(fr["delta"], color="k", lw=0.8, ls="--")
    ax.text(xlo, fr["delta"], " $\\delta$", va="bottom", fontsize=9)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xlabel("x (streamwise)"); ax.set_ylabel("y (wall normal)")
    ax.set_title("side view: the slab is 2 radii deep in x", fontsize=10)

    # --- 4. wall-normal distribution ------------------------------------
    ax = fig.add_subplot(2, 2, 4)
    ax.hist(pos[:, 1], bins=24, range=(ylo, yhi), color="#4a7ba7",
            edgecolor="white")
    ax.axvline(fr["delta"], color="k", lw=0.8, ls="--")
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_xlabel("y (wall normal)"); ax.set_ylabel("eddies per bin")
    ax.set_title("wall-normal spread over the box\n"
                 "(wall at y=0, $\\delta$ dashed; should be flat)", fontsize=10)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR,
                       os.path.splitext(os.path.basename(path))[0] + ".png")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote %s  (%d eddies, coverage %.1f%%)" % (out, fr["n"], cov))


def main():
    paths = sys.argv[1:] or sorted(glob.glob("osem3d_eddies*_??????.h5"))
    if not paths:
        print("no osem3d_eddies*.h5 files found", file=sys.stderr)
        return 1
    for p in paths:
        plot_frame(read_frame(p), p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
