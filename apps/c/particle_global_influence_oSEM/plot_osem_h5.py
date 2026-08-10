"""Plot the HDF5 output written by osem_io.h.

Each osem_output_<step>.h5 holds the inlet fields under the block group, the
complete id-ordered eddy list, and the run constants.

    python3 plot_osem_h5.py             # all frames -> frames/*.png + a GIF
    python3 plot_osem_h5.py --no-gif

Four panels:

  top three   u', v', w' on the inlet plane. The plane is long in z and thin in
              y, so each is a wide strip. One shared diverging scale centred on
              zero across all three components AND all frames -- a per-frame or
              per-component autoscale would hide both the evolution and the
              fact that the three are supposed to be statistically alike.

  overlaid    on EACH panel, the eddies actually contributing to it: those
              within one radius of the sampling plane, drawn at their true
              radius and coloured by the sign that drives THAT component --
              eps_x for u', eps_y for v', eps_z for w'. This is the mechanism
              made visible: every blob is one of these circles, and its colour
              matches. The three overlays use the same circles in the same
              places and differ only in colour, which is itself the point --
              the positions are shared, the signs are independent.

  bottom      rms u', v', w' against time, with the target u0*TI. This is the
              thing the scheme exists to reproduce.
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
H5DIR = "h5files"          # where the app writes frames (OSEM_OUTDIR in osem_io.h)
OUTDIR = "frames"
GIF = os.path.join(OUTDIR, "osem.gif")


def read_frame(path):
    with h5py.File(path, "r") as f:
        NY, NZ = int(f["NY"][0]), int(f["NZ"][0])

        # Grid dats carry a one-cell halo in each direction and are stored
        # [z][y]; transpose so rows are y and columns z, matching the plot.
        def fld(name):
            return f[BLOCK][name][:][1:-1, 1:-1].T

        fields = {n: fld(n) for n in ("uprime", "vprime", "wprime")}

        eddies = {
            "x": f["eddy_x"][:],
            "y": f["eddy_y"][:],
            "z": f["eddy_z"][:],
            "r": f["eddy_r"][:],
            "sx": f["eddy_sx"][:],
            "sy": f["eddy_sy"][:],
            "sz": f["eddy_sz"][:],
        }

        meta = {
            "step": int(f["timestep"][0]),
            "time": float(f["time"][0]),
            "niter": int(f["NITER"][0]),
            "u0ti": float(f["u0ti"][0]),
            "xplane": float(f["x_plane"][0]),
            "box": f["box"][:],
            "rms": f["rms"][:],
            "use_tbl": int(f["use_tbl"][0]),
            "prof": f["rms_profile"][:].reshape(-1, 3),
            "targ": f["rms_target"][:].reshape(-1, 3),
            "neddy": int(f["neddy"][0]),
            "ny": NY,
            "nz": NZ,
        }
    return fields, eddies, meta


def plot_frame(path, frame, clim, history, outdir):
    fields, eddies, meta = frame
    ymin, ymax, zmin, zmax = meta["box"]
    extent = [zmin, zmax, ymin, ymax]

    fig, axes = plt.subplots(
        4, 1, figsize=(13.0, 9.6),
        gridspec_kw={"height_ratios": [1, 1, 1, 1.35]}
    )

    near = np.abs(eddies["x"] - meta["xplane"]) < eddies["r"]
    strong = near & (np.abs(eddies["x"] - meta["xplane"]) < 0.35 * eddies["r"])
    idx = np.nonzero(near)[0]

    for ax, name, label, skey in zip(axes[:3],
                                     ("uprime", "vprime", "wprime"),
                                     ("u'", "v'", "w'"),
                                     ("sx", "sy", "sz")):
        im = ax.imshow(fields[name], origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto",
                       interpolation="bilinear")
        ax.set_ylabel("y")
        ax.text(0.006, 0.86, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2.0))
        fig.colorbar(im, ax=ax, fraction=0.020, pad=0.006)

        # Faint enough to see the field underneath; only the eddies nearest the
        # plane are outlined, or several hundred circles make it unreadable.
        sgn = eddies[skey]
        for i in idx:
            hot = strong[i]
            ax.add_patch(Circle(
                (eddies["z"][i], eddies["y"][i]), eddies["r"][i], fill=False,
                linewidth=0.7 if hot else 0.35,
                edgecolor=("#b2182b" if sgn[i] > 0 else "#2166ac"),
                alpha=0.55 if hot else 0.16))

    axes[0].set_title(
        "%d of %d eddies lie within a radius of the plane; each panel is"
        " outlined with the sign that drives it "
        "(red $+1$ / blue $-1$, bold = nearest the plane)"
        % (int(near.sum()), meta["neddy"]), fontsize=10)

    for ax in axes[:3]:
        ax.set_xlim(zmin, zmax)
        ax.set_ylim(ymin, ymax)
    axes[2].set_xlabel("z")

    # ---- bottom panel --------------------------------------------------
    axr = axes[3]

    if meta["use_tbl"]:
        # With a tabulated profile the target is a FUNCTION OF y, so a single
        # u0*TI line is meaningless -- the plane-averaged rms has nothing to be
        # compared against. Show the profile against the target instead.
        nyv = meta["prof"].shape[0] - 1
        yv = ymin + (ymax - ymin) * np.arange(nyv + 1) / float(nyv)
        for k, (lab, col) in enumerate((("u'", "tab:blue"), ("v'", "tab:green"),
                                        ("w'", "tab:red"))):
            axr.plot(yv, meta["prof"][:, k], color=col, linewidth=1.7,
                     label="rms " + lab)
            axr.plot(yv, meta["targ"][:, k], color=col, linewidth=1.1,
                     linestyle="--", alpha=0.75,
                     label=("tabulated target" if k == 0 else None))
        axr.set_xlabel("y")
        axr.set_ylabel("rms")
        axr.set_xlim(ymin, ymax)
        axr.set_ylim(bottom=0.0)
        axr.set_title("wall-normal profile: computed (solid) vs tabulated "
                      "Reynolds stresses (dashed)", fontsize=11)
        axr.legend(loc="upper right", frameon=False, fontsize=9, ncol=4)
        axr.grid(alpha=0.25)
        fig.suptitle("oSEM with OPS particles -- step %d / %d,  t = %.3e s"
                     % (meta["step"], meta["niter"], meta["time"]),
                     fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        os.makedirs(outdir, exist_ok=True)
        png = os.path.join(outdir,
                           os.path.basename(path).replace(".h5", ".png"))
        fig.savefig(png, dpi=95)
        plt.close(fig)
        return png

    t, r = history
    for k, (label, colour) in enumerate((("u'", "tab:blue"),
                                         ("v'", "tab:green"),
                                         ("w'", "tab:red"))):
        axr.plot(t, [v[k] for v in r], color=colour, linewidth=1.5,
                 label="rms " + label)
    axr.axhline(meta["u0ti"], color="0.35", linestyle="--", linewidth=1.2,
                label=r"target $u_0\,TI$")
    axr.axvline(meta["time"], color="0.6", linewidth=0.9, linestyle=":")
    axr.set_xlabel("time")
    axr.set_ylabel("rms")
    axr.set_ylim(bottom=0.0)
    axr.legend(loc="lower right", frameon=False, fontsize=9, ncol=4)
    axr.grid(alpha=0.25)

    fig.suptitle("oSEM with OPS particles -- step %d / %d,  t = %.3e s"
                 % (meta["step"], meta["niter"], meta["time"]),
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, os.path.basename(path).replace(".h5", ".png"))
    fig.savefig(png, dpi=95)
    plt.close(fig)
    return png


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]

    files = args or sorted(glob.glob(os.path.join(H5DIR, "osem_output_[0-9]*.h5")))
    if not files:
        # Fall back to the old location so frames written before the move
        # still plot, rather than failing with a bare "none found".
        files = sorted(glob.glob("osem_output_[0-9]*.h5"))
        if files:
            print("note: reading from the app root; the app now writes to %s/"
                  % H5DIR)
    if not files:
        sys.exit("no osem_output_*.h5 files found in %s/ or %s"
                 % (H5DIR, os.getcwd()))

    frames = [read_frame(p) for p in files]

    # One symmetric scale over every component and every frame.
    clim = max(float(np.abs(f[0][n]).max())
               for f in frames for n in ("uprime", "vprime", "wprime"))

    order = np.argsort([m["time"] for _, _, m in frames])
    history = ([frames[i][2]["time"] for i in order],
               [frames[i][2]["rms"] for i in order])

    print("field range   : +/- %.4g" % clim)
    print("rms at the end: u' %.4f  v' %.4f  w' %.4f  (target %.4f)"
          % (history[1][-1][0], history[1][-1][1], history[1][-1][2],
             frames[0][2]["u0ti"]))

    pngs = []
    for path, frame in zip(files, frames):
        pngs.append(plot_frame(path, frame, clim, history, OUTDIR))
        print("wrote", pngs[-1])

    if "--no-gif" not in flags and len(pngs) > 1:
        from PIL import Image

        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
        imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=120,
                     loop=0)
        print("wrote", GIF, "(%d frames)" % len(imgs))


if __name__ == "__main__":
    main()
