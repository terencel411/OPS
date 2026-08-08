"""Plot the HDF5 output written by gather_io.h.

Each gather_output_<timestep>.h5 holds the grid dat under the block group, the
particle state as flat gid-ordered arrays, the run constants, and -- specific to
this app -- the measured cost of BOTH gathers at that step.

    python3 plot_gather_h5.py            # all frames -> frames/*.png + a GIF
    python3 plot_gather_h5.py --no-gif   # PNGs only

Two panels:

  left    the particle cloud coloured by influence. Identical physics to
          particle_global_influence -- the gather method cannot change it.
  right   what the Allgatherv costs, per step, as the run proceeds, with a
          marker on the frame being shown. Log y-axis; raw trace faint, rolling
          median bold.
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
GIF = os.path.join(OUTDIR, "gather.gif")


def read_frame(path):
    with h5py.File(path, "r") as f:
        NX, NY = int(f["NX"][0]), int(f["NY"][0])
        xgrid = f[BLOCK]["x_grid"][:].reshape(NY + 2, NX + 2, 2)[1:-1, 1:-1, :]

        n = int(f["no_particles"][0])
        state = {
            "pos": f["position"][:].reshape(n, 2),
            "mass": f["strength"][:],
            "phi": f["influence"][:],
        }
        meta = {
            "step": int(f["timestep"][0]),
            "time": float(f["time"][0]),
            "length": float(f["LENGTH"][0]),
            "nsteps": int(f["NSTEPS"][0]),
            "t_gather": float(f["t_gather_ms"][0]),
        }
    return xgrid, state, meta


def rolling_median(v, w):
    """Centred rolling median, clipped at the ends."""
    a = np.asarray(v, dtype=float)
    n = len(a)
    h = w // 2
    return np.array([np.median(a[max(0, i - h):min(n, i + h + 1)])
                     for i in range(n)])


def plot_frame(path, xgrid, state, meta, clim, history, outdir):
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    length = meta["length"]

    ax.plot(xgrid[..., 0], xgrid[..., 1], color="0.90", linewidth=0.4)
    ax.plot(xgrid[..., 0].T, xgrid[..., 1].T, color="0.90", linewidth=0.4)

    sizes = 10.0 + 55.0 * (state["mass"] - state["mass"].min()) / max(
        1e-30, np.ptp(state["mass"])
    )
    sc = ax.scatter(state["pos"][:, 0], state["pos"][:, 1], s=sizes,
                    c=state["phi"], cmap="inferno", vmin=clim[0], vmax=clim[1],
                    edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"influence  $\phi_i=\sum_{j\neq i} m_j/(\epsilon+r_{ij})$")

    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, length)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("step %d / %d    t = %.3f    %d particles"
                 % (meta["step"], meta["nsteps"], meta["time"],
                    len(state["phi"])))

    # ---- right: what the gather costs ----------------------------------
    #
    # Single-step timings are jittery -- one slow step is scheduling noise, not
    # a property of the method. So the raw trace is drawn faint and a rolling
    # median over it bold: the honest data stays visible, the trend is legible.
    t, tg = history
    axr.plot(t, tg, color="tab:green", linewidth=0.8, alpha=0.30)
    axr.plot(t, rolling_median(tg, 7), color="tab:green", linewidth=1.9,
             label="MPI_Allgatherv")

    axr.axvline(meta["time"], color="0.5", linewidth=0.9, linestyle="--")
    j = int(np.argmin(np.abs(np.asarray(t) - meta["time"])))
    axr.plot([t[j]], [tg[j]], "o", color="tab:green", markersize=6)

    axr.set_title("gather cost per step  (median %.3f ms)"
                  % float(np.median(tg)))
    axr.set_yscale("log")
    axr.set_xlabel("time")
    axr.set_ylabel("ms per step")
    axr.legend(loc="best", frameon=False, fontsize=9)
    axr.grid(alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, os.path.basename(path).replace(".h5", ".png"))
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]

    files = args or sorted(glob.glob("gather_output_[0-9]*.h5"))
    if not files:
        sys.exit("no gather_output_*.h5 files found in %s" % os.getcwd())

    frames = [read_frame(p) for p in files]

    phis = np.concatenate([s["phi"] for _, s, _ in frames])
    clim = (float(phis.min()), float(phis.max()))

    order = np.argsort([m["time"] for _, _, m in frames])
    history = (
        [frames[i][2]["time"] for i in order],
        [frames[i][2]["t_gather"] for i in order],
    )

    print("influence range : %.3f .. %.3f" % clim)
    print("MPI_Allgatherv  : median %.4f ms/step"
          % float(np.median(history[1])))

    pngs = []
    for path, (xgrid, state, meta) in zip(files, frames):
        pngs.append(plot_frame(path, xgrid, state, meta, clim, history, OUTDIR))
        print("wrote", pngs[-1])

    if "--no-gif" not in flags and len(pngs) > 1:
        from PIL import Image

        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
        imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=120,
                     loop=0)
        print("wrote", GIF, "(%d frames)" % len(imgs))


if __name__ == "__main__":
    main()
