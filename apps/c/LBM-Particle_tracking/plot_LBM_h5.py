"""Plot the HDF5 output written by LBM_particle_io.h.

Each LBM_output_<timestep>.h5 holds the lattice-Boltzmann grid dats under the
block group, the particle dats as flat arrays, and the simulation constants.

    python3 plot_LBM_h5.py                 # every LBM_output_??????.h5 -> frames/
    python3 plot_LBM_h5.py LBM_output.h5   # a single file
"""

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLOCK = "lattice-boltzmann_grid"
OUTDIR = "frames"


def read_frame(path):
    """Return grid coordinates, velocity components and particle state."""
    with h5py.File(path, "r") as f:
        grid = f[BLOCK]

        # Grid dats carry a one-cell halo in each direction and are stored
        # [y][x]; xgrid is dim 2, so its fastest axis holds (x, y).
        ny, nx = grid["ux"].shape
        xgrid = grid["xgrid"][:].reshape(ny, nx, 2)[1:-1, 1:-1, :]
        u_x = grid["ux"][1:-1, 1:-1]
        u_y = grid["uy"][1:-1, 1:-1]

        nparts = int(f["no_particles"][0])
        pos = f["part_coords"][:].reshape(nparts, 2)
        vel = f["part_vels"][:].reshape(nparts, 2)
        timestep = int(f["timestep"][0])

    return xgrid, u_x, u_y, pos, vel, timestep


def plot_frame(path, outdir):
    xgrid, u_x, u_y, pos, vel, timestep = read_frame(path)
    speed = np.hypot(u_x, u_y)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # The moving lid is an order of magnitude faster than the recirculation it
    # drives, so clip the colour scale or the interior washes out to black.
    vmax = np.percentile(speed[:-2, :], 99) or speed.max()
    mesh = ax.pcolormesh(xgrid[..., 0], xgrid[..., 1], speed,
                         shading="auto", cmap="viridis", vmin=0.0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="|u|  (clipped at 99th percentile)")

    if speed.max() > 0.0:
        ax.streamplot(xgrid[0, :, 0], xgrid[:, 0, 1], u_x, u_y,
                      color="white", linewidth=0.6, density=1.1,
                      arrowsize=0.7)

    ax.scatter(pos[:, 0], pos[:, 1], s=2.0, c="orangered",
               edgecolors="none", alpha=0.85)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("%s  -  timestep %d  -  %d particles"
                 % (os.path.basename(path), timestep, len(pos)))

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, os.path.basename(path).replace(".h5", ".png"))
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    files = sys.argv[1:] or sorted(glob.glob("LBM_output_[0-9]*.h5"))
    if not files:
        sys.exit("no LBM_output_*.h5 files found in %s" % os.getcwd())

    for path in files:
        print("wrote", plot_frame(path, OUTDIR))


if __name__ == "__main__":
    main()
