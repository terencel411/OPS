#!/usr/bin/env python3
"""Plot slices out of an OpenSBLI/OPS HDF5 output file.

The datasets are stored as (z, y, x) with a 5-cell halo on every side, so for
block0np0/1/2 = 750/250/100 the arrays are (110, 260, 760).  Slicing
"block0np0" therefore fixes the *last* array axis and gives a y-z plane, which
is the inlet plane the synthetic eddies are injected into.

Coordinates come from the file itself (x0_B0 -> x, x1_B0 -> y, x2_B0 -> z), so
the wall-normal stretching is respected rather than being drawn as if uniform.

Examples
--------
  # one inlet plane, all three momenta, into ./slices
  ./plot_slices.py opensbli_output_000010.h5 --slices 5

  # a range, and a strided range
  ./plot_slices.py opensbli_output_000010.h5 --slices 5-25
  ./plot_slices.py opensbli_output_000010.h5 --slices 5-755:50
  ./plot_slices.py opensbli_output_000010.h5 --slices 5,10,20,100

  # velocities rather than momenta, fluctuation about the spanwise mean
  ./plot_slices.py opensbli_output_000010.h5 --slices 5-15 --velocity --fluct

  # what is in the file
  ./plot_slices.py opensbli_output_000010.h5 --list
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCK = "opensbliblock00"
DEFAULT_VARS = ["rhou0_B0", "rhou1_B0", "rhou2_B0"]

# block0npN -> axis of the stored (z, y, x) array, and the coordinate dataset
# that varies along it.
BLOCK_AXIS = {0: 2, 1: 1, 2: 0}
COORD = {0: "x0_B0", 1: "x1_B0", 2: "x2_B0"}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}
# in-plane (horizontal, vertical) block dimensions for a slice normal to each axis
IN_PLANE = {0: (2, 1), 1: (0, 2), 2: (0, 1)}


def parse_slices(spec, n):
    """'5', '5-25', '5-755:50', '5,10,20' -> sorted list of unique indices."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        step = 1
        if ":" in tok:
            tok, step_s = tok.split(":", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"step must be >= 1 in '{spec}'")
        if "-" in tok.lstrip("-"):
            lo_s, hi_s = tok.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1, step))
        else:
            out.append(int(tok))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(
            f"slice index/indices {bad[:8]} outside the available range 0..{n - 1}"
        )
    return sorted(set(out))


def load_planes(dset, axis, indices, budget_bytes=1 << 30):
    """Yield (index, plane) for the requested indices along `axis` of `dset`.

    Reads the whole contiguous span in one go when it fits in the budget - a
    strided per-slice read of a non-chunked dataset is far slower.
    """
    lo, hi = indices[0], indices[-1]
    span = hi - lo + 1
    shape = list(dset.shape)
    shape[axis] = span
    if np.prod(shape) * dset.dtype.itemsize <= budget_bytes:
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi + 1)
        block = dset[tuple(sl)]
        for i in indices:
            yield i, np.take(block, i - lo, axis=axis)
    else:
        for i in indices:
            yield i, np.take(dset, i, axis=axis)


def plane_coords(g, block_dim, idx, halo, keep_halo):
    """(horizontal, vertical) 1-D coordinate vectors for the slice plane."""
    h_dim, v_dim = IN_PLANE[block_dim]
    trim = slice(None) if keep_halo else slice(halo, -halo or None)
    sl = [0, 0, 0]
    sl[BLOCK_AXIS[block_dim]] = idx
    out = []
    for d in (h_dim, v_dim):
        s = list(sl)
        s[BLOCK_AXIS[d]] = trim
        out.append(np.asarray(g[COORD[d]][tuple(s)]))
    return out


def robust_limits(planes, percentile, symmetric):
    stacked = np.concatenate([p.ravel() for p in planes])
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(stacked, 100.0 - percentile))
    hi = float(np.percentile(stacked, percentile))
    if symmetric:
        m = max(abs(lo), abs(hi)) or 1.0
        return -m, m
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def choose_style(lo, hi, cmap_arg):
    """Diverging about zero when the field straddles zero, sequential otherwise."""
    if cmap_arg:
        return cmap_arg, False
    straddles = lo < 0 < hi and abs(lo) > 0.05 * abs(hi)
    return ("RdBu_r", True) if straddles else ("viridis", False)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Plot slices of an OpenSBLI HDF5 output file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n", 1)[-1],
    )
    p.add_argument("file", help="opensbli_output*.h5")
    p.add_argument("--slices", default="5",
                   help="index, 'lo-hi', 'lo-hi:step', or a comma-separated mix "
                        "(default: 5, i.e. the first interior plane)")
    p.add_argument("--vars", nargs="+", default=DEFAULT_VARS,
                   help=f"datasets to plot (default: {' '.join(DEFAULT_VARS)})")
    p.add_argument("--axis", type=int, choices=(0, 1, 2), default=0,
                   help="slice normal to block0npN (default: 0, the inlet plane)")
    p.add_argument("--outdir", default="slices", help="output directory (default: slices)")
    p.add_argument("--halo", type=int, default=5, help="halo depth per side (default: 5)")
    p.add_argument("--index-space", choices=("file", "grid"), default="file",
                   help="'file': index into the stored array including halo, 0..759. "
                        "'grid': index into the physical grid, 0..749 (default: file)")
    p.add_argument("--keep-halo", action="store_true",
                   help="also draw the halo cells in the slice plane")
    p.add_argument("--velocity", action="store_true",
                   help="divide rhou*_B0 by rho_B0 and plot velocities")
    p.add_argument("--fluct", action="store_true",
                   help="subtract the spanwise (z) mean at each point - isolates "
                        "the eddy structures from the mean profile")
    p.add_argument("--cmap", default=None, help="colormap (default: chosen per field)")
    p.add_argument("--clim", nargs=2, type=float, metavar=("LO", "HI"),
                   help="explicit colour limits, shared by every image")
    p.add_argument("--percentile", type=float, default=99.5,
                   help="percentile for automatic colour limits (default: 99.5)")
    p.add_argument("--separate", action="store_true",
                   help="one file per variable instead of one panel row per slice")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("LO", "HI"),
                   help="limit the horizontal axis")
    p.add_argument("--ylim", nargs=2, type=float, metavar=("LO", "HI"),
                   help="limit the vertical axis, e.g. --ylim 0 15 to zoom into "
                        "the boundary layer where the eddies live")
    p.add_argument("--aspect", choices=("equal", "auto"), default="equal",
                   help="'equal' keeps the physical proportions, 'auto' fills "
                        "the panel (default: equal)")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--list", action="store_true", help="list datasets and exit")
    args = p.parse_args(argv)

    if not os.path.exists(args.file):
        p.error(f"no such file: {args.file}")

    with h5py.File(args.file, "r") as f:
        if BLOCK not in f:
            p.error(f"{args.file} has no group '{BLOCK}'")
        g = f[BLOCK]

        if args.list:
            print(f"{args.file}")
            for k in sorted(g):
                print(f"  {BLOCK}/{k:<12} {g[k].shape} {g[k].dtype}")
            for k in ("block0np0", "block0np1", "block0np2", "iter", "simulation_time"):
                if k in f:
                    print(f"  {k:<24} {f[k][0]}")
            return 0

        missing = [v for v in args.vars if v not in g]
        if missing:
            p.error(f"not in {BLOCK}: {', '.join(missing)} "
                    f"(available: {', '.join(sorted(g))})")

        axis = BLOCK_AXIS[args.axis]
        n_stored = g[args.vars[0]].shape[axis]
        n_grid = n_stored - 2 * args.halo

        try:
            if args.index_space == "grid":
                idx = [i + args.halo for i in parse_slices(args.slices, n_grid)]
            else:
                idx = parse_slices(args.slices, n_stored)
        except ValueError as e:
            p.error(str(e))

        iter_no = int(f["iter"][0]) if "iter" in f else None
        sim_t = float(f["simulation_time"][0]) if "simulation_time" in f else None

        trim = slice(None) if args.keep_halo else slice(args.halo, -args.halo or None)
        keep = [trim, trim, trim]
        keep[axis] = slice(None)
        keep = tuple(keep)

        print(f"{args.file}: block0np{args.axis} has {n_grid} interior planes "
              f"({n_stored} stored); plotting {len(idx)} of them")

        # Pull every requested plane for every variable up front, so the colour
        # limits can be shared across the whole batch and the images stay
        # comparable when flicked through.
        data = {}
        for var in args.vars:
            dset = g[var]
            planes = {}
            for i, plane in load_planes(dset, axis, idx):
                planes[i] = np.asarray(plane)[tuple(s for k, s in enumerate(keep)
                                                   if k != axis)]
            data[var] = planes

        if args.velocity:
            rho = {}
            for i, plane in load_planes(g["rho_B0"], axis, idx):
                rho[i] = np.asarray(plane)[tuple(s for k, s in enumerate(keep)
                                                 if k != axis)]
            for var in args.vars:
                if var.startswith("rhou"):
                    for i in idx:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            data[var][i] = data[var][i] / rho[i]

        if args.fluct:
            # spanwise direction is block dim 2 (z); after slicing it is the
            # first remaining axis for an x- or y-normal slice, absent for a
            # z-normal one.
            if args.axis == 2:
                print("  --fluct ignored: a z-normal slice has no spanwise extent",
                      file=sys.stderr)
            else:
                for var in args.vars:
                    for i in idx:
                        a = data[var][i]
                        data[var][i] = a - a.mean(axis=0, keepdims=True)

        limits = {}
        for var in args.vars:
            if args.clim:
                lo, hi = args.clim
                cmap, _ = choose_style(lo, hi, args.cmap)
            else:
                sym = args.fluct or var in ("rhou1_B0", "rhou2_B0")
                lo, hi = robust_limits(list(data[var].values()), args.percentile, sym)
                cmap, _ = choose_style(lo, hi, args.cmap)
            limits[var] = (lo, hi, cmap)
            print(f"  {var}: colour limits [{lo:+.4g}, {hi:+.4g}]  cmap={cmap}")

        os.makedirs(args.outdir, exist_ok=True)
        h_dim, v_dim = IN_PLANE[args.axis]
        label = {d: f"{AXIS_NAME[d]}" for d in (h_dim, v_dim)}
        stem = os.path.splitext(os.path.basename(args.file))[0]
        kind = "u" if args.velocity else ""
        written = []

        for i in idx:
            hc, vc = plane_coords(g, args.axis, i, args.halo, args.keep_halo)
            pos = float(g[COORD[args.axis]][tuple(
                i if k == axis else 0 for k in range(3))])
            grid_i = i - args.halo
            head = f"{AXIS_NAME[args.axis]} = {pos:.4f}  (plane {grid_i}, stored index {i})"
            if iter_no is not None:
                head += f"   iter {iter_no}"
            if sim_t is not None:
                head += f"   t = {sim_t:.4f}"

            groups = [[v] for v in args.vars] if args.separate else [args.vars]
            for grp in groups:
                fig, axes = plt.subplots(
                    1, len(grp), figsize=(6.2 * len(grp), 4.6), squeeze=False,
                    constrained_layout=True)
                for ax, var in zip(axes[0], grp):
                    lo, hi, cmap = limits[var]
                    # stored plane is (first-remaining, second-remaining); pcolormesh
                    # wants C indexed (vertical, horizontal)
                    c = data[var][i]
                    if args.axis == 0:      # (z, y) -> want (y, z)
                        c = c.T
                    mesh = ax.pcolormesh(hc, vc, c, cmap=cmap, vmin=lo, vmax=hi,
                                         shading="auto", rasterized=True)
                    name = var
                    if args.velocity and var.startswith("rhou"):
                        name = var.replace("rhou", "u")
                    if args.fluct:
                        name += "'"
                    ax.set_title(name)
                    ax.set_xlabel(label[h_dim])
                    ax.set_ylabel(label[v_dim])
                    if args.xlim:
                        ax.set_xlim(*args.xlim)
                    if args.ylim:
                        ax.set_ylim(*args.ylim)
                    ax.set_aspect(args.aspect)
                    fig.colorbar(mesh, ax=ax, shrink=0.9)
                fig.suptitle(head)

                tag = grp[0] if args.separate else "all"
                out = os.path.join(
                    args.outdir,
                    f"{stem}_{AXIS_NAME[args.axis]}{grid_i:04d}_{kind}{tag}.png")
                fig.savefig(out, dpi=args.dpi)
                plt.close(fig)
                written.append(out)

        print(f"wrote {len(written)} image(s) to {args.outdir}/")
        for w in written[:6]:
            print(f"  {w}")
        if len(written) > 6:
            print(f"  ... and {len(written) - 6} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
