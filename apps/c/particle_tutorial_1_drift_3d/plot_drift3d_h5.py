"""Plot the HDF5 output written by drift_io.h (3D drift).

Each drift3d_output_<timestep>.h5 holds the grid dat under the block group, the
particle dats as flat arrays, and the simulation constants.

    python3 plot_drift3d_h5.py                     # every drift3d_output_??????.h5
    python3 plot_drift3d_h5.py drift3d_output.h5   # a single file

Three panels: the 3D cloud, the view down z (where the drift and the periodic
seam read most clearly), and the view down x (where the sheet is seen face-on
and should look like a filled NPY x NPZ grid).
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

BLOCK = "tutorial_block_3d"
OUTDIR = "frames"


def read_frame(path):
    """Return particle state and the run constants."""
    with h5py.File(path, "r") as f:
        nparts = int(f["no_particles"][0])
        pos = f["position"][:].reshape(nparts, 3)
        ids = f["id"][:]

        timestep = int(f["timestep"][0])
        time = float(f["time"][0])
        domain = f["domain"][:]          # xlo xhi ylo yhi zlo zhi

    return pos, ids, timestep, time, domain


def plot_frame(path, outdir):
    pos, ids, timestep, time, dom = read_frame(path)

    fig = plt.figure(figsize=(16.0, 5.0))

    # Left: the 3D cloud. Colour by id so a particle keeps its colour across
    # frames, which is what makes the re-injection at the upstream face visible.
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=12.0, c=ids, cmap="viridis",
               edgecolors="none", depthshade=True)
    ax.set_xlim(dom[0], dom[1])
    ax.set_ylim(dom[2], dom[3])
    ax.set_zlim(dom[4], dom[5])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("t = %.3f  -  %d particles" % (time, len(pos)))

    # Middle: looking down z, which is where the drift and the periodic seam are
    # easiest to read. A correct sheet appears here as a single vertical line.
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(pos[:, 0], pos[:, 1], s=14.0, c=ids, cmap="viridis",
                edgecolors="none")
    for seam in (dom[0], dom[1]):
        ax2.axvline(seam, color="steelblue", linewidth=1.2, linestyle="--")
    ax2.set_xlim(dom[0], dom[1])
    ax2.set_ylim(dom[2], dom[3])
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("along z  -  step %d  (sheet = one line)" % timestep)

    # Right: looking down x, i.e. the sheet face-on. This is the view that shows
    # the seeding actually is an NPY x NPZ grid on the y-z plane.
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(pos[:, 1], pos[:, 2], s=18.0, c=ids, cmap="viridis",
                edgecolors="none")
    ax3.set_xlim(dom[2], dom[3])
    ax3.set_ylim(dom[4], dom[5])
    ax3.set_aspect("equal")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    ax3.set_title("along x  -  the sheet face-on")

    fig.suptitle(os.path.basename(path))

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, os.path.basename(path).replace(".h5", ".png"))
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    files = sys.argv[1:] or sorted(glob.glob("drift3d_output_[0-9]*.h5"))
    if not files:
        sys.exit("no drift3d_output_*.h5 files found in %s" % os.getcwd())

    for path in files:
        print("wrote", plot_frame(path, OUTDIR))


if __name__ == "__main__":
    main()
