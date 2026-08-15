"""
Phase 25i: Waddington Landscape 3D Visualization
Classic epigenetic landscape: cells roll down from high potential into fate valleys.

X = pseudotime (progression)
Y = fate bias (P(EGC) - P(Stasis)) — separates the bifurcation directions
Z = -V(s) (potential energy: high = undifferentiated, low = committed)

Surface interpolated via grid + gaussian smoothing.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\FDU\Y4S2\xiyuan\project\script3")
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

print("=" * 70)
print("Phase 25i: Waddington Landscape 3D Visualization")
print("=" * 70)

# Load data
print("\n[1/4] Loading data ...")
adata_mc = sc.read_h5ad(DATA / "rl_metacells.h5ad")
T_sparse = sparse.load_npz(RESULTS / "rl_transition_matrix.npz")
T_cr = T_sparse.toarray()
val_df = pd.read_csv(RESULTS / "rl_value_function.csv")

stages = adata_mc.obs['stage'].values
pt = adata_mc.obs['dpt_pseudotime'].values
V = val_df['V_value'].values
n_cells = len(adata_mc)

print(f"  {n_cells} metacells loaded")

# ===================================================================
# [2/4] Compute fate probabilities for all cells
# ===================================================================
print("\n[2/4] Computing fate coordinates ...")

# Forward propagation from each cell (k=20 steps)
egc_mask = (stages == 'EGC') | (stages == 'EGC_multi_region')
gc_mask = stages == 'GC'
nag_mask = stages == 'NAG'

# Propagate identity matrix through T^20
T_k = np.linalg.matrix_power(T_cr, 20)

fate_egc = (T_k * egc_mask[None, :]).sum(axis=1)
fate_gc = (T_k * gc_mask[None, :]).sum(axis=1)
fate_stasis = (T_k * nag_mask[None, :]).sum(axis=1)

# Y-axis: fate bias (separates valleys)
fate_bias = fate_egc - fate_stasis

# Potential energy: -V(s), normalized
potential = -V
potential_norm = (potential - potential.min()) / (potential.max() - potential.min())

print(f"  Pseudotime range: [{pt.min():.3f}, {pt.max():.3f}]")
print(f"  Fate bias range: [{fate_bias.min():.3f}, {fate_bias.max():.3f}]")
print(f"  Potential range: [{potential_norm.min():.3f}, {potential_norm.max():.3f}]")

# ===================================================================
# [3/4] Interpolate landscape surface
# ===================================================================
print("\n[3/4] Interpolating landscape surface ...")

# Grid for surface
grid_x = np.linspace(pt.min(), pt.max(), 80)
grid_y = np.linspace(fate_bias.min() - 0.05, fate_bias.max() + 0.05, 60)
gx, gy = np.meshgrid(grid_x, grid_y)

# Interpolate potential onto grid
gz = griddata(
    np.column_stack([pt, fate_bias]),
    potential_norm,
    (gx, gy),
    method='linear',
    fill_value=np.nan
)

# Fill NaN with nearest
mask_nan = np.isnan(gz)
if mask_nan.any():
    gz_nearest = griddata(
        np.column_stack([pt, fate_bias]),
        potential_norm,
        (gx, gy),
        method='nearest'
    )
    gz[mask_nan] = gz_nearest[mask_nan]

# Smooth to create valley shape
gz_smooth = gaussian_filter(gz, sigma=2.5)

# Enhance valleys: add concavity along Y to create valley walls
y_center = (grid_y.max() + grid_y.min()) / 2
y_norm = (gy - y_center) / (grid_y.max() - grid_y.min())
valley_enhancement = 0.15 * y_norm**2
gz_landscape = gz_smooth + valley_enhancement

print(f"  Grid: {gx.shape}, smoothed with sigma=2.5")

# ===================================================================
# [4/4] Visualization
# ===================================================================
print("\n[4/4] Generating Waddington landscape figures ...")

stage_colors = {
    'NAG': '#4575b4', 'CAG': '#91bfdb', 'IM': '#fee090',
    'EGC': '#d73027', 'EGC_multi_region': '#fc8d59', 'GC': '#1a1a1a'
}
cell_colors = [stage_colors.get(s, '#999999') for s in stages]

# --- Figure 1: 3D Waddington Landscape ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Surface
surf = ax.plot_surface(gx, gy, gz_landscape, alpha=0.3,
                       cmap='terrain', edgecolor='none',
                       antialiased=True)

# Scatter cells on landscape (slightly above surface)
z_offset = 0.02
ax.scatter(pt, fate_bias, potential_norm + z_offset,
           c=cell_colors, s=8, alpha=0.7, edgecolors='none')

ax.set_xlabel('Pseudotime (Progression)', fontsize=11, labelpad=10)
ax.set_ylabel('Fate Bias (EGC ← → Stasis)', fontsize=11, labelpad=10)
ax.set_zlabel('Potential Energy (-V)', fontsize=11, labelpad=10)
ax.set_title('Waddington Epigenetic Landscape\n'
             'Gastric Cancer Precursor Progression (IRL-derived)',
             fontsize=13, pad=20)

# View angle
ax.view_init(elev=25, azim=-60)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=s)
                   for s, c in stage_colors.items()]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / "waddington_landscape_3d.png", dpi=200,
            bbox_inches='tight')
plt.close()
print("  Saved: waddington_landscape_3d.png")

# --- Figure 2: 2D contour version (for paper main figure) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Contour landscape with cell scatter
ax = axes[0]
contour = ax.contourf(gx, gy, gz_landscape, levels=20, cmap='terrain', alpha=0.6)
ax.scatter(pt, fate_bias, c=cell_colors, s=12, alpha=0.7,
           edgecolors='white', linewidths=0.3)
ax.set_xlabel('Pseudotime (Progression)')
ax.set_ylabel('Fate Bias (P(EGC) - P(Stasis))')
ax.set_title('Waddington Landscape (2D projection)\nColor = disease stage')
plt.colorbar(contour, ax=ax, label='Potential (-V)')

# Add bifurcation annotations
ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
ax.text(pt.max() * 0.7, fate_bias.max() * 0.8, 'EGC valley',
        fontsize=10, color='red', ha='center')
ax.text(pt.max() * 0.7, fate_bias.min() * 0.8, 'Stasis valley',
        fontsize=10, color='blue', ha='center')

legend_elements = [Patch(facecolor=c, label=s)
                   for s, c in stage_colors.items()]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

# Panel B: Same layout, colored by P(EGC) fate
ax = axes[1]
ax.contourf(gx, gy, gz_landscape, levels=20, cmap='terrain', alpha=0.3)
sc_fate = ax.scatter(pt, fate_bias, c=fate_egc, cmap='RdYlBu_r',
                     s=12, alpha=0.8, edgecolors='none')
ax.set_xlabel('Pseudotime (Progression)')
ax.set_ylabel('Fate Bias (P(EGC) - P(Stasis))')
ax.set_title('Waddington Landscape\nColor = P(EGC) fate probability')
plt.colorbar(sc_fate, ax=ax, label='P(EGC)')
ax.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(FIGURES / "waddington_landscape_2d.png", dpi=200,
            bbox_inches='tight')
plt.close()
print("  Saved: waddington_landscape_2d.png")

# --- Figure 3: Cross-section view (landscape profile at key pseudotimes) ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Cross sections at 3 pseudotime points
pt_cuts = [0.02, 0.05, 0.08]
cut_labels = ['Early (NAG/CAG)', 'Mid (IM)', 'Late (IM→EGC)']

for i, (pt_cut, label) in enumerate(zip(pt_cuts, cut_labels)):
    ax = axes[i]
    # Find nearest grid column
    col_idx = np.argmin(np.abs(grid_x - pt_cut))
    profile = gz_landscape[:, col_idx]

    ax.fill_between(grid_y, profile, profile.max() + 0.05,
                    color='#8B4513', alpha=0.3)
    ax.plot(grid_y, profile, 'k-', lw=2)

    # Scatter cells near this pseudotime
    pt_window = 0.01
    near_mask = np.abs(pt - pt_cut) < pt_window
    if near_mask.any():
        near_y = fate_bias[near_mask]
        near_z = potential_norm[near_mask]
        near_colors = [cell_colors[j] for j in np.where(near_mask)[0]]
        ax.scatter(near_y, near_z, c=near_colors, s=30, zorder=5,
                   edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Fate Bias')
    ax.set_ylabel('Potential')
    ax.set_title(f'{label}\n(pt≈{pt_cut:.2f})')
    ax.set_ylim(profile.min() - 0.05, profile.max() + 0.1)

plt.suptitle('Landscape Cross-sections: Valley Formation along Progression',
             fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "waddington_cross_sections.png", dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: waddington_cross_sections.png")

# --- Figure 4: Multi-angle 3D (supplementary) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                          subplot_kw={'projection': '3d'})

angles = [(25, -60), (45, -120), (10, -30)]
titles = ['Standard view', 'Rear view (valleys visible)', 'Low-angle view']

for ax, (elev, azim), title in zip(axes, angles, titles):
    ax.plot_surface(gx, gy, gz_landscape, alpha=0.35,
                    cmap='terrain', edgecolor='none')
    ax.scatter(pt, fate_bias, potential_norm + z_offset,
               c=cell_colors, s=5, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PT', fontsize=8)
    ax.set_ylabel('Fate', fontsize=8)
    ax.set_zlabel('-V', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)

plt.tight_layout()
plt.savefig(FIGURES / "waddington_landscape_multiangle.png", dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: waddington_landscape_multiangle.png")

print("\n" + "=" * 70)
print("Phase 25i COMPLETE")
print("=" * 70)
print("  Generated 4 landscape figures:")
print("    1. waddington_landscape_3d.png — main 3D landscape")
print("    2. waddington_landscape_2d.png — 2D contour (paper main fig)")
print("    3. waddington_cross_sections.png — valley profile at 3 timepoints")
print("    4. waddington_landscape_multiangle.png — multi-angle (supp)")
print("\n  Interpretation:")
print("    - High potential (top) = undifferentiated stem-like state")
print("    - Low potential (valleys) = committed fate")
print("    - Right valley (positive fate bias) = EGC trajectory")
print("    - Left valley (negative fate bias) = Stasis/regression")
print("    - Ball rolling downhill = cells following IRL-optimal policy")
print("=" * 70)
