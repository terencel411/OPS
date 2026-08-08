"""Plot the HDF5 output written by influence_io.h.

Each influence_output_<timestep>.h5 holds the grid dat under the block group,
the particle state as flat gid-ordered arrays (position, velocity, strength,
influence) and the run constants.

    python3 plot_influence_h5.py                 # all frames -> frames/*.png + frames/influence.gif
    python3 plot_influence_h5.py --no-gif        # PNGs only
    python3 plot_influence_h5.py --quiver        # add velocity arrows
    python3 plot_influence_h5.py influence_output.h5   # one file

The left panel is the cloud, coloured by the all-to-all influence. The colour
scale is fixed across every frame -- that is deliberate: the whole point is to
see phi change as the cloud spreads, and a per-frame autoscale would hide
exactly that. The right panel tracks min / mean / max phi over the run, with a
dot on the frame being shown.
"""

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLOCK = "influence_block"
OUTDIR = "frames"
GIF = os.path.join(OUTDIR, "influence.gif")


def read_frame(path):
    """Return the grid, the particle state and the run constants."""
    with h5py.File(path, "r") as f:
        NX, NY = int(f["NX"][0]), int(f["NY"][0])

        # x_grid carries a one-cell halo in each direction and is stored
        # [y][x]; it is dim 2, so its fastest axis holds (x, y).
        xgrid = f[BLOCK]["x_grid"][:].reshape(NY + 2, NX + 2, 2)[1:-1, 1:-1, :]

        n = int(f["no_particles"][0])
        state = {
            "pos": f["position"][:].reshape(n, 2),
            "vel": f["velocity"][:].reshape(n, 2),
            "mass": f["strength"][:],
            "phi": f["influence"][:],
            "gid": f["gid"][:],
        }

        meta = {
            "step": int(f["timestep"][0]),
            "time": float(f["time"][0]),
            "length": float(f["LENGTH"][0]),
            "soften": float(f["soften"][0]),
            "nsteps": int(f["NSTEPS"][0]),
        }

    return xgrid, state, meta


def plot_frame(path, xgrid, state, meta, clim, history, outdir, quiver):
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    length = meta["length"]

    # Faint mesh, purely as a spatial reference: the grid carries no physics.
    ax.plot(xgrid[..., 0], xgrid[..., 1], color="0.90", linewidth=0.4)
    ax.plot(xgrid[..., 0].T, xgrid[..., 1].T, color="0.90", linewidth=0.4)

    # Marker area tracks strength, colour tracks the all-to-all influence.
    sizes = 10.0 + 55.0 * (state["mass"] - state["mass"].min()) / max(
        1e-30, np.ptp(state["mass"])
    )
    sc = ax.scatter(
        state["pos"][:, 0],
        state["pos"][:, 1],
        s=sizes,
        c=state["phi"],
        cmap="inferno",
        vmin=clim[0],
        vmax=clim[1],
        edgecolors="none",
    )

    if quiver:
        ax.quiver(
            state["pos"][:, 0],
            state["pos"][:, 1],
            state["vel"][:, 0],
            state["vel"][:, 1],
            color="0.35",
            width=0.003,
            scale=6.0,
        )

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"influence  $\phi_i=\sum_{j\neq i} m_j/(\epsilon+r_{ij})$")

    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, length)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "step %d / %d    t = %.3f    %d particles"
        % (meta["step"], meta["nsteps"], meta["time"], len(state["phi"]))
    )

    # ---- right panel: how the interaction evolves --------------------
    t, lo, mean, hi = history
    axr.fill_between(t, lo, hi, color="tab:orange", alpha=0.25,
                     label="min - max")
    axr.plot(t, mean, color="tab:red", linewidth=1.6, label="mean")

    axr.axvline(meta["time"], color="0.5", linewidth=0.9, linestyle="--")
    j = int(np.argmin(np.abs(np.asarray(t) - meta["time"])))
    axr.plot([t[j]], [mean[j]], "o", color="tab:red", markersize=6)

    axr.set_xlabel("time")
    axr.set_ylabel(r"$\phi$")
    axr.set_title("influence over the run (all ranks, all particles)")
    axr.legend(loc="best", frameon=False, fontsize=9)
    axr.grid(alpha=0.25)

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, os.path.basename(path).replace(".h5", ".png"))
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    files = args or sorted(glob.glob("influence_output_[0-9]*.h5"))
    if not files:
        sys.exit("no influence_output_*.h5 files found in %s" % os.getcwd())

    # One pass to read everything: the frames are a few kB each, and both the
    # shared colour scale and the time series need the whole run up front.
    frames = [read_frame(p) for p in files]

    phis = np.concatenate([s["phi"] for _, s, _ in frames])
    clim = (float(phis.min()), float(phis.max()))

    order = np.argsort([m["time"] for _, _, m in frames])
    history = (
        [frames[i][2]["time"] for i in order],
        [float(frames[i][1]["phi"].min()) for i in order],
        [float(frames[i][1]["phi"].mean()) for i in order],
        [float(frames[i][1]["phi"].max()) for i in order],
    )

    print("influence range over the run: %.3f .. %.3f" % clim)

    pngs = []
    for path, (xgrid, state, meta) in zip(files, frames):
        pngs.append(
            plot_frame(path, xgrid, state, meta, clim, history, OUTDIR,
                       "--quiver" in flags)
        )
        print("wrote", pngs[-1])

    if "--no-gif" not in flags and len(pngs) > 1:
        from PIL import Image

        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                for p in pngs]
        imgs[0].save(GIF, save_all=True, append_images=imgs[1:],
                     duration=120, loop=0)
        print("wrote", GIF, "(%d frames)" % len(imgs))


if __name__ == "__main__":
    main()
