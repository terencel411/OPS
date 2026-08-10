import os
import glob
import re
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Setup output directory
output_dir = "paraview"
os.makedirs(output_dir, exist_ok=True)

# 2. Find and sort HDF5 files numerically based on the timestep index
h5_files = glob.glob("*.h5")

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

h5_files.sort(key=extract_number)

if not h5_files:
    print("No HDF5 files found in the current directory.")
    exit()

print(f"Found {len(h5_files)} HDF5 files. Commencing batch visualization...")

frame_paths = []

# 3. Iterate over every file and generate individual PNG frames
for file_path in h5_files:
    base_name = os.path.splitext(file_path)[0]
    output_png = os.path.join(output_dir, f"{base_name}.png")
    
    with h5py.File(file_path, "r") as f:
        # Extract metadata dimensions
        ny = int(f["/NY"][()].item())
        nz = int(f["/NZ"][()].item())
        
        # Read the raw fluctuation datasets
        u_raw = np.squeeze(f["/osem_block/uprime"][:])
        v_raw = np.squeeze(f["/osem_block/vprime"][:])
        w_raw = np.squeeze(f["/osem_block/wprime"][:])
    
    # Calculate and strip halo/ghost cells dynamically.
    #
    # NY and NZ are grid INTERVALS, so the plane has ny+1 by nz+1 NODES -- the
    # dats are 103 x 153 for the default 100 x 150, i.e. (ny+1)+2 by (nz+1)+2
    # with a halo of 1. Slicing ny by nz instead drops the last node in each
    # direction, and the dropped y row is the one at eddy_y_max, where the
    # edge deficit is strongest.
    nny, nnz = ny + 1, nz + 1

    def strip(a, g0, n0, g1, n1):
        return a[g0:g0 + n0, g1:g1 + n1]

    dz, dy = u_raw.shape[0] - nnz, u_raw.shape[1] - nny
    dy2, dz2 = u_raw.shape[0] - nny, u_raw.shape[1] - nnz

    if dz >= 0 and dy >= 0 and dz % 2 == 0 and dy % 2 == 0:
        gz, gy = dz // 2, dy // 2
        u_core = strip(u_raw, gz, nnz, gy, nny).T
        v_core = strip(v_raw, gz, nnz, gy, nny).T
        w_core = strip(w_raw, gz, nnz, gy, nny).T
    elif dy2 >= 0 and dz2 >= 0 and dy2 % 2 == 0 and dz2 % 2 == 0:
        gy, gz = dy2 // 2, dz2 // 2
        u_core = strip(u_raw, gy, nny, gz, nnz)
        v_core = strip(v_raw, gy, nny, gz, nnz)
        w_core = strip(w_raw, gy, nny, gz, nnz)
    else:
        print(f"Skipping {file_path}: Unable to parse shape {u_raw.shape}")
        continue

    # Plotting layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    components = [u_core, v_core, w_core]
    titles = [r"$u'$ Fluctuations", r"$v'$ Fluctuations", r"$w'$ Fluctuations"]
    
    for ax, data, title in zip(axes, components, titles):
        limit = max(abs(data.min()), abs(data.max()))
        if limit == 0: limit = 1.0
        
        im = ax.imshow(data.T, cmap="RdBu_r", origin="lower", 
                       vmin=-limit, vmax=limit, extent=[0, ny, 0, nz])
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(f"Y Grid Line (0 - {ny})", fontsize=11)
        ax.set_ylabel(f"Z Grid Line (0 - {nz})", fontsize=11)
        
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.75)
        cbar.ax.tick_params(labelsize=9)
        
    # Include filename as a timestamp visual cue inside the layout
    fig.suptitle(f"File Source: {file_path}", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    
    # Save the frame
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    frame_paths.append(output_png)
    print(f"   Processed frame: {output_png}")

# 4. Compile individual images into an animated GIF using PIL
print("\nCompiling frames into an animated GIF...")
gif_path = os.path.join(output_dir, "inlet_fluctuations_evolution.gif")

# Load images via PIL
images = [Image.open(frame) for frame in frame_paths]

if images:
    # duration=200 sets 200 milliseconds (0.2s) per frame. loop=0 loops infinitely.
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )
    print(f"\nSuccess! GIF generated at: {gif_path}")
else:
    print("\nError: No valid frames were generated to build the GIF.")

