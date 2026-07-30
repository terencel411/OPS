"""Plot the HDF5 output written by drift_io.h.

Each drift_output_<timestep>.h5 holds the grid dat under the block group, the
particle dats as flat arrays, and the simulation constants.

    python3 plot_drift_h5.py                  # every drift_output_??????.h5 -> frames/
    python3 plot_drift_h5.py drift_output.h5  # a single file
"""

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLOCK = "tutorial_block"
OUTDIR = "frames"


def read_frame(path):
    """Return grid coordinates, particle state and the run constants."""
    with h5py.File(path, "r") as f:
        # x_grid carries a one-cell halo in each direction and is stored
        # [y][x]; it is dim 2, so its fastest axis holds (x, y).
        NX, NY = int(f["NX"][0]), int(f["NY"][0])
        xgrid = f[BLOCK]["x_grid"][:].reshape(NY + 2, NX + 2, 2)[1:-1, 1:-1, :]

        nparts = int(f["no_particles"][0])
        pos = f["position"][:].reshape(nparts, 2)
        vel = f["velocity"][:].reshape(nparts, 2)
        ids = f["id"][:]

        timestep = int(f["timestep"][0])
        time = float(f["time"][0])
        length = float(f["LENGTH"][0])

    return xgrid, pos, vel, ids, timestep, time, length


def plot_frame(path, outdir):
    xgrid, pos, vel, ids, timestep, time, length = read_frame(path)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # The grid carries no physics in this tutorial, so draw it as a faint mesh
    # to give the particle positions a spatial reference.
    ax.plot(xgrid[..., 0], xgrid[..., 1], color="0.85", linewidth=0.4)
    ax.plot(xgrid[..., 0].T, xgrid[..., 1].T, color="0.85", linewidth=0.4)

    # Periodic seam at the left and right edges. Particles leaving through the
    # right re-enter on the left, so the seed lattice does not translate
    # rigidly and there is nothing useful to overlay. Nothing to mark top and
    # bottom: the drift is along x only, so no particle ever goes near them.
    for seam in (0.0, length):
        ax.axvline(seam, color="steelblue", linewidth=1.2, linestyle="--")

    # Colour by id so a given particle keeps its colour across frames.
    ax.scatter(pos[:, 0], pos[:, 1], s=18.0, c=ids, cmap="viridis",
               edgecolors="none")

    if np.any(vel):
        ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1],
                  color="0.4", width=0.003, scale=8.0)

    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, length)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("%s  -  step %d  -  t = %.3f  -  %d particles"
                 % (os.path.basename(path), timestep, time, len(pos)))

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, os.path.basename(path).replace(".h5", ".png"))
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    files = sys.argv[1:] or sorted(glob.glob("drift_output_[0-9]*.h5"))
    if not files:
        sys.exit("no drift_output_*.h5 files found in %s" % os.getcwd())

    for path in files:
        print("wrote", plot_frame(path, OUTDIR))


if __name__ == "__main__":
    main()
