#!/usr/bin/env python3
"""Streamwise (x-y) and cross-plane (z-y) views of the flow from OpenSBLI HDF5 output.

  plot_streamwise.py run_dir/                        creates streamwise view of the turbulence for all
                                                     fields (rho, T, mu, etc.), selects mid-span value
                                                     for z (z_max / 2)
  plot_streamwise.py run_dir/ -z 10                  same as above but you can select a specific value
                                                     for z                                                   
  plot_streamwise.py run_dir/ -f T                   you can specify specific fields to plot as well
                                                     T is for temperature
  plot_streamwise.py run_dir/ -f T --contours 20     temperature with contour lines over it (default is 40)
  plot_streamwise.py run_dir/ -f T --cross 10 100    z-y cross-planes for temp at x = 10 and 100
  plot_streamwise.py run_dir/ --anim flow.gif        can also create an animation for it
  plot_streamwise.py a.h5 b.h5                       instead of specifying a folder - can provide 
                                                     specific h5 filenames to create the plots
"""
import sys, os, re, glob, argparse
import numpy as np
import h5py

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")

GROUP = "opensbliblock00"

# short name -> (dataset, axis label, colormap, symmetric about zero)
DIRECT = {
    "rho":   ("rho_B0",   r"$\rho$",     "turbo", False),
    "rhou0": ("rhou0_B0", r"$\rho u_0$", "viridis", False),
    "rhou1": ("rhou1_B0", r"$\rho u_1$", "RdBu_r",  True),
    "rhou2": ("rhou2_B0", r"$\rho u_2$", "RdBu_r",  True),
    "rhoE":  ("rhoE_B0",  r"$\rho E$",   "viridis", False),
    "T":     ("T_B0",     r"$T$",        "inferno", False),
    "p":     ("p_B0",     r"$p$",        "viridis", False),
    "mu":    ("mu_B0",    r"$\mu$",      "viridis", False),
}
VEL = {"u": (0, r"$u_0$", "viridis", False),
       "v": (1, r"$u_1$", "RdBu_r",  True),
       "w": (2, r"$u_2$", "RdBu_r",  True)}


SNAPSHOT = re.compile(r"opensbli_output.*\.h5$")


def targets(paths):
    """-> [(sortkey, path)] over directories and/or explicit files, in iteration order.
    Only opensbli_output*.h5 is accepted; stats_output.h5 holds averages, not a flow field."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            if not SNAPSHOT.fullmatch(os.path.basename(p)):
                sys.exit(f"not an OpenSBLI snapshot: {p}\n"
                         "  this tool plots opensbli_output*.h5 only")
            m = re.search(r"_(\d+)\.h5$", p)
            out.append((int(m.group(1)) if m else 1 << 30, p))
        elif os.path.isdir(p):
            for q in glob.glob(os.path.join(p, "opensbli_output_*.h5")):
                m = re.search(r"_(\d+)\.h5$", q)
                if m:
                    out.append((int(m.group(1)), q))
            q = os.path.join(p, "opensbli_output.h5")
            if os.path.exists(q):
                out.append((1 << 30, q))
        else:
            sys.exit(f"no such file or directory: {p}")
    if not out:
        sys.exit("no OpenSBLI snapshots found -- looked for opensbli_output*.h5")
    out.sort()
    return out


def scalar(h, name, default=None):
    return float(np.asarray(h[name]).ravel()[0]) if name in h else default


def crop(a, halo):
    return a if halo <= 0 else a[tuple(slice(halo, -halo) for _ in a.shape)]


def grid(h, halo):
    """-> (x, y, z): the one line along which each coordinate dataset varies."""
    g = h[GROUP]
    for k in ("x0_B0", "x1_B0", "x2_B0"):
        if k not in g:
            sys.exit(f"{k} not in the file -- every OpenSBLI snapshot carries coordinates")
    return (crop(np.asarray(g["x0_B0"][halo, halo, :], dtype=float), halo),
            crop(np.asarray(g["x1_B0"][halo, :, halo], dtype=float), halo),
            crop(np.asarray(g["x2_B0"][:, halo, halo], dtype=float), halo))


def slab(g, dset, halo, k=None, i=None):
    """One 2-D slice, halos stripped, rows = y.
    k -> the (y,x) plane at that z index, or the spanwise mean when k is 'mean'.
    i -> the (y,z) plane at that x index."""
    d = g[dset]
    if i is not None:
        return crop(np.asarray(d[:, :, i], dtype=float), halo).T
    if k == "mean":
        sl = slice(halo, d.shape[0] - halo) if halo else slice(None)
        return crop(np.asarray(d[sl, :, :], dtype=float).mean(axis=0), halo)
    return crop(np.asarray(d[k, :, :], dtype=float), halo)


def build(g, name, **where):
    """-> (2-D field, axis label, colormap, symmetric)."""
    def P(n):
        if n not in g:
            sys.exit(f"{n} not in the file\n  available: {', '.join(sorted(g))}")
        return slab(g, n, **where)

    if name in DIRECT:
        dset, *style = DIRECT[name]
        return (P(dset), *style)
    if name in VEL:
        n, *style = VEL[name]
        return (P(f"rhou{n}_B0") / P("rho_B0"), *style)
    dset = name if name in g else f"{name}_B0"
    return (P(dset), name, "viridis", False)


def cbticks(vmin, vmax):
    """MaxNLocator can run past the ends once contourf adds its extend triangles."""
    t = MaxNLocator(6).tick_values(vmin, vmax)
    return t[(t >= vmin) & (t <= vmax)]


def limits(arrays, sym, clim, pct):
    if clim:
        return clim
    stack = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    lo, hi = np.percentile(stack, [pct, 100 - pct])
    return (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi))) if sym else (lo, hi)


def draw(ax, a, cols, rows, jmax, cmap, vmin, vmax, ncont=0, nlev=0):
    """Filled discrete levels when nlev is set, otherwise a smooth image."""
    r, f = rows[:jmax], a[:jmax]
    if nlev:
        m = ax.contourf(cols, r, f, levels=np.linspace(vmin, vmax, nlev + 1),
                        cmap=cmap, extend="both")
        m.set_rasterized(True)
    else:
        m = ax.pcolormesh(cols, r, f, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="auto", rasterized=True)
    if ncont:
        ax.contour(cols, r, f, levels=np.linspace(vmin, vmax, ncont + 1),
                   colors="k", linewidths=0.3)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="run directory, or individual opensbli_output*.h5 files.\n"
                        "A directory is globbed for opensbli_output_*.h5 in numeric order, then\n"
                        "opensbli_output.h5 last.")
    p.add_argument("-f", "--field", default="rho",
                   help="what to plot.\n"
                        "  stored   rho rhou0 rhou1 rhou2 rhoE T p mu\n"
                        "  derived  u v w   -- velocity, = rhou{0,1,2}_B0 / rho_B0\n"
                        "  (default: rho)")
    p.add_argument("-o", "--outdir", default="frames",
                   help="directory for the PNGs, created if absent. One file per snapshot, named\n"
                        "<field>_<iter>.png, or cross_<field>_<iter>.png under --cross. <iter> is\n"
                        "read from inside the file and can differ from the HDF5 filename number.\n"
                        "(default: frames)")
    p.add_argument("-z", default=None,
                   help="which spanwise plane the x-y view cuts along. z is the direction across\n"
                        "the flow; every plane is statistically equivalent, so no single one is\n"
                        "more correct than another.\n"
                        "  omitted   mid-span, the middle z plane\n"
                        "  N         z plane N, interior index 0..nz-1 (halo added for you)\n"
                        "  mean      average of every z plane -- eddies cancel, leaving an\n"
                        "            approximate mean flow. Reads the whole 3-D field, not one slab.\n"
                        "Ignored with --cross.")
    p.add_argument("--cross", type=float, nargs="+", default=None, metavar="X",
                   help="switch from the x-y side view to z-y cross-planes - inlet plane\n"
                        "Ignored under --cross: -z, --aspect, --anim.")
    p.add_argument("--ymax", type=float, default=0.0,
                   help="crop the view above this physical y; 0 shows the full domain height.\n"
                        "(default: 0, the full height)")
    p.add_argument("--halo", type=int, default=5,
                   help="halo depth stripped off every axis before plotting. The halo holds ghost\n"
                        "data, so plotting it frames the picture in garbage.  (default: 5)")
    p.add_argument("--contours", type=int, default=40,
                   help="black contour lines drawn over the colours; 0 for none. They read well on\n"
                        "smooth fields (means, T, p) and turn to spaghetti on instantaneous\n"
                        "turbulence.  (default: 40)")
    p.add_argument("--cmap", default=None,
                   help="matplotlib colormap name, overriding the per-field default: turbo for rho,\n"
                        "RdBu_r for the signed fields (v w rhou1 rhou2), inferno for T, else viridis.")
    p.add_argument("--clim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="fix the colour limits instead of deriving them from the data.")
    p.add_argument("--pct", type=float, default=0.5,
                   help="percentile trimmed off each end when deriving the colour limits, so a few\n"
                        "outlying cells cannot flatten the whole scale. Limits are taken once over\n"
                        "every frame and panel, so a run is self-consistent.  (default: 0.5)")
    p.add_argument("--aspect", default="auto", metavar="A",
                   help="y exaggeration on the x-y plane, as a multiplier:\n"
                        "    drawn height / width = A * (y shown / x range)\n"
                        "so a fixed A changes the picture shape whenever --ymax changes.\n"
                        "  auto  pick A so the figure always has the whole domain's proportions,\n"
                        "        A = (full height) / (height shown). A --ymax 25 crop of a\n"
                        "        375x100 box is then drawn 0.27 tall, the same as the full box.\n"
                        "  1.0   true scale: one unit of y is one unit of x\n"
                        "  >1    stretch y, to make a thin layer readable\n"
                        "Cross-planes ignore this and are always true scale.  (default: auto)")
    p.add_argument("--levels", type=int, default=22,
                   help="number of filled discrete colour bands; 0 draws a smooth image instead.\n"
                        "Bands plus --contours give the usual OpenSBLI look; 0 and 0 give a\n"
                        "photograph-like view that suits turbulent snapshots.  (default: 22)")
    p.add_argument("--title", default=None,
                   help="replace the per-figure title text.")
    p.add_argument("--width", type=float, default=15.0,
                   help="figure width in inches; the height follows from --aspect.  (default: 15)")
    p.add_argument("--dpi", type=int, default=140,
                   help="output resolution in dots per inch.  (default: 140)")
    p.add_argument("--anim", default=None, metavar="GIF",
                   help="also write an animated GIF of every frame, in iteration order. Needs\n"
                        "Pillow. x-y only -- silently skipped with --cross.")
    p.add_argument("--fps", type=float, default=2.0,
                   help="frames per second for --anim.  (default: 2)")
    a = p.parse_args()

    files = targets(a.paths)
    os.makedirs(a.outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.6,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4,
                         "xtick.major.size": 5, "ytick.major.size": 5})
    x = y = z = None
    frames = []            # (iter, time, stem, [panels]) -- one panel for x-y, N for --cross
    label = cmap = None
    sym = False

    for _, path in files:
        with h5py.File(path, "r") as h:
            if GROUP not in h:
                print(f"  skipped {os.path.basename(path)}: no /{GROUP}")
                continue
            g = h[GROUP]
            if x is None:
                x, y, z = grid(h, a.halo)
                if a.cross:
                    ii = [int(np.abs(x - v).argmin()) for v in a.cross]
                    print("  cross-planes at x = " +
                          ", ".join(f"{x[i]:.2f} (i={i})" for i in ii))
            if a.cross:
                panels = [build(g, a.field, halo=a.halo, i=i + a.halo) for i in ii]
            else:
                nzi = next(iter(g.values())).shape[0] - 2 * a.halo
                k = "mean" if a.z == "mean" else \
                    a.halo + (nzi // 2 if a.z is None else int(a.z))
                if k != "mean" and not a.halo <= k < a.halo + nzi:
                    sys.exit(f"z index {a.z} outside 0..{nzi - 1}")
                panels = [build(g, a.field, halo=a.halo, k=k)]
            label, cmap, sym = panels[0][1], panels[0][2], panels[0][3]
            frames.append((int(scalar(h, "iter", -1)), scalar(h, "simulation_time", float("nan")),
                           os.path.splitext(os.path.basename(path))[0],
                           [q[0] for q in panels]))
        print(f"  read {frames[-1][2]:42s} iter {frames[-1][0]:>7d}  t = {frames[-1][1]:8.2f}")

    if not frames:
        sys.exit("nothing to plot")
    frames.sort(key=lambda f: f[0])
    if a.cmap:
        cmap = a.cmap
    vmin, vmax = limits([q for f in frames for q in f[3]], sym, a.clim, a.pct)
    jmax = max(int(np.searchsorted(y, a.ymax)) if a.ymax > 0 else len(y), 2)
    # "auto" draws the crop with the whole domain's proportions, so the picture shape
    # stays put when --ymax changes; a number is the plain y multiplier.
    aspect = ((y[-1] - y[0]) / (y[jmax - 1] - y[0]) if a.aspect == "auto"
              else float(a.aspect))
    # distinct runs can share an iteration, which would collide on the output name
    dup = len({f[0] for f in frames}) < len(frames)
    zdesc = "spanwise mean" if a.z == "mean" else ("mid-span" if a.z is None else f"z index {a.z}")
    print(f"\n  field {a.field}   colour limits [{vmin:.5g}, {vmax:.5g}]"
          f"   y <= {y[jmax - 1]:.2f} ({jmax} of {len(y)} planes)"
          f"   {'cross-plane' if a.cross else zdesc}"
          f"{'' if a.cross else f'   aspect {aspect:.3g}'}")

    written = []
    for it, t, stem, panels in frames:
        tag = f"{it:07d}_{stem}" if dup else f"{it:07d}"
        if a.cross:
            n, pw = len(panels), 3.6
            fig, axes = plt.subplots(1, n, squeeze=False, figsize=(
                pw * n + 1.5, pw * (y[jmax - 1] - y[0]) / (z[-1] - z[0]) + 1.5))
            for ax, arr, i in zip(axes[0], panels, ii):
                m = draw(ax, arr, z, y, jmax, cmap, vmin, vmax, a.contours, a.levels)
                ax.set_title(f"x = {x[i]:.1f}", fontsize=12)
                ax.set_xlabel("z")
                ax.set_aspect("equal")
            axes[0, 0].set_ylabel("y")
            fig.suptitle(a.title or
                         f"cross-plane {a.field}   iter {it}   t = {t:.2f}   {stem}", fontsize=13)
            fig.colorbar(m, ax=axes[0].tolist(), pad=0.015, fraction=0.03, label=label,
                         ticks=cbticks(vmin, vmax))
            out = os.path.join(a.outdir, f"cross_{a.field}_{tag}.png")
        else:
            hgt = a.width * aspect * (y[jmax - 1] - y[0]) / (x[-1] - x[0])
            fig, ax = plt.subplots(figsize=(a.width, hgt + 1.35))
            m = draw(ax, panels[0], x, y, jmax, cmap, vmin, vmax, a.contours, a.levels)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_aspect(aspect)
            ax.set_title(a.title or
                         (f"{a.field}   iter {it}   t = {t:.2f}   ({zdesc})"
                          + (f"   [{stem}]" if dup else "")), fontsize=13)
            fig.colorbar(m, ax=ax, pad=0.01, fraction=0.02, label=label, ticks=cbticks(vmin, vmax))
            fig.tight_layout()
            out = os.path.join(a.outdir, f"{a.field}_{tag}.png")
        fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")

    if a.anim and not a.cross:
        try:
            from PIL import Image
        except ImportError:
            sys.exit("--anim needs Pillow:  pip install pillow")
        ims = [Image.open(q).convert("P", palette=Image.ADAPTIVE) for q in written]
        ims[0].save(a.anim, save_all=True, append_images=ims[1:],
                    duration=int(1000 / a.fps), loop=0)
        print(f"  wrote {a.anim}  ({len(ims)} frames at {a.fps} fps)")
    print()


if __name__ == "__main__":
    main()
