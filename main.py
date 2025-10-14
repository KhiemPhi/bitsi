import argparse
import open3d as o3d 
import sys
import numpy as np 
import matplotlib.pyplot as plt
from bitsi_slicer.bitsi_slicer import slice_object_bbox, bitsi_metric, build_cloud_object
from numpy.polynomial import Chebyshev

def fit_best_chebyshev(x, y, min_deg=2, max_deg=None, criterion="bic"):
    """
    Fit Chebyshev polynomials of increasing degree and pick the best one
    based on the chosen information criterion ('bic', 'aic', 'aicc', or 'adj_r2').

    Returns:
        best_coeffs, best_degree, best_yfit, best_metrics
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points for fitting.")

    if max_deg is None:
        max_deg = min(8, n - 2)
    max_deg = max(max_deg, min_deg)

    # scale x → [-1, 1] for Chebyshev stability
    xmin, xmax = np.min(x), np.max(x)
    if xmax == xmin:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1.0

    def metrics(y_true, y_pred, k):
        rss = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
        r2 = 1 - rss / tss
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
        sigma2 = rss / max(n, 1)
        eps = 1e-18
        aic = n * np.log(sigma2 + eps) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        bic = n * np.log(sigma2 + eps) + k * np.log(max(n, 1))
        return dict(r2=r2, adj_r2=adj_r2, aic=aic, aicc=aicc, bic=bic, rss=rss)

    best = None
    all_models = []

    for deg in range(min_deg, max_deg + 1):
        try:
            ch = Chebyshev.fit(x_scaled, y, deg)
            y_fit = ch(x_scaled)
            m = metrics(y, y_fit, k=deg + 1)
            m["degree"] = deg
            m["model"] = ch
            m["y_fit"] = y_fit
            all_models.append(m)
        except np.linalg.LinAlgError:
            continue

    if not all_models:
        raise RuntimeError("Chebyshev fitting failed for all degrees.")

    if criterion.lower() in ("bic", "aic", "aicc"):
        key = criterion.lower()
        best = min(all_models, key=lambda m: m[key])
    elif criterion.lower() == "adj_r2":
        best = max(all_models, key=lambda m: m["adj_r2"])
    else:
        raise ValueError("criterion must be one of 'bic','aic','aicc','adj_r2'")

    best_ch = best["model"]
    coeffs = best_ch.coef
    degree = best["degree"]
    y_fit = best["y_fit"]
    return coeffs, degree, y_fit, best

def visualize_bitsi_with_extrema(bitsi_x, bitsi_y, bitsi_z):
    # Axis conventions
    axis_names = ["X", "Y", "Z"]
    axis_colors = {"X": "r", "Y": "g", "Z": "b"}
    all_bitsi = [bitsi_x, bitsi_y, bitsi_z]

    # --- Find dominant direction (largest variation)
    variances = [np.var(b) for b in all_bitsi]
    dominant_idx = int(np.argmax(variances))
    dominant_axis = axis_names[dominant_idx]

    y = np.array(all_bitsi[dominant_idx], dtype=float)
    x = np.arange(len(y))

    # --- Fit best Chebyshev polynomial
    coeffs, degree, y_fit, metrics = fit_best_chebyshev(x, y, min_deg=2, max_deg=5, criterion="bic")
    print(f"🎯 Chosen Chebyshev degree {degree}, adj R²={metrics['adj_r2']:.3f}, BIC={metrics['bic']:.2f}")

    # --- Derivative and extrema
    deriv = np.gradient(y_fit)
    extrema_idx = np.where(np.diff(np.sign(deriv)) != 0)[0]

    # --- Find nearest integer slice indices for extrema
    slice_indices = [int(round(i)) for i in extrema_idx]

    # --- Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, "o", color=axis_colors[dominant_axis], label=f"BITSI ({dominant_axis}-axis)")
    plt.plot(x, y_fit, "--", color="k", alpha=0.7, label=f"Chebyshev deg {degree} fit")

    # Mark and label extrema
    for i in slice_indices:
        kind = "min" if y_fit[i] < np.mean(y_fit) else "max"
        plt.scatter(i, y_fit[i], color="orange", s=60, zorder=3)
        plt.text(i, y_fit[i] + 0.05, f"{kind} (slice {i})", ha="center", fontsize=10, color="orange")

    plt.title(f"BITSI Curve Along Dominant Axis ({dominant_axis}-axis)")
    plt.xlabel("Slice Index")
    plt.ylabel("BITSI Value")
    plt.legend()
    plt.grid(True)

    print(f"📍 Slice indices near extrema: {slice_indices}")
    plt.show()

    return dominant_axis, slice_indices, coeffs

def visualize_bitsi(bitsi_x, bitsi_y, bitsi_z):
    """
    Visualize BITSI metric values along x, y, z directions as scatter plots.
    Each value corresponds to a slice along that axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    directions = ['X', 'Y', 'Z']
    bitsi_values = [bitsi_x, bitsi_y, bitsi_z]
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for ax, vals, dir_label, color in zip(axes, bitsi_values, directions, colors):
        vals = np.array(vals)
        ax.scatter(np.arange(len(vals)), vals, s=50, color=color, alpha=0.8, edgecolor='k')
        ax.set_title(f'BITSI Along {dir_label}-axis', fontsize=12)
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('BITSI Value')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, len(vals))
        ax.axhline(y=np.mean(vals), color='gray', linestyle=':', label='Mean')
        ax.legend()

    plt.suptitle("BITSI Metric Distribution by Axis", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.show()

def pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx):
    axis_names = ["X", "Y", "Z"]
    axis_colors = {"X": "\033[91m", "Y": "\033[92m", "Z": "\033[94m"}  # red, green, blue
    reset = "\033[0m"

    # Compute mean IBR values for each axis
    mean_vals = [np.mean(bitsi_x), np.mean(bitsi_y), np.mean(bitsi_z)]

    print("\n✅ BITSI Metrics Summary:")
    print("-" * 40)
    for i, (axis, mean_val) in enumerate(zip(axis_names, mean_vals)):
        color = axis_colors[axis]
        print(f"{color}{axis}-axis:{reset} mean IBR = {mean_val:.3f}")

    # Determine the dominant (smallest mean) axis
    dominant_axis_idx = int(np.argmin(mean_vals))
    dominant_axis = axis_names[dominant_axis_idx]

    print("-" * 40)
    print(f"🧭 Dominant slicing direction: {axis_colors[dominant_axis]}{dominant_axis}-axis{reset}")
    print(f"   (lowest mean IBR = {mean_vals[dominant_axis_idx]:.3f})")

def auto_thickness(cloud_object, scale=0.01):
    bbox_min = np.min(cloud_object.points, axis=0)
    bbox_max = np.max(cloud_object.points, axis=0)
    diag_length = np.linalg.norm(bbox_max - bbox_min)
    return scale * diag_length

def main():
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
    # Create an ArgumentParser Object
    parser = argparse.ArgumentParser(description='Task-Oriented Grasp Synthesis')

    # Add a command-line argument for the input filenamep
    parser.add_argument('--object', type=str, help='Name of the ycb object', default='toy')
    parser.add_argument('--width', type=float, help='gripper width', default=0.13)
    parser.add_argument('--height', type=float, help='gripper height', default=0.07)      
    parser.add_argument('--partial', action='store_true', help='use partial or not partial data')
    parser.add_argument('--epsilon', type=float, default=5e-3 )
    args = parser.parse_args()
    gripper_width = args.width 
    gripper_height = args.height  
    epsilon = args.epsilon 

    # -------------------------------------------------------------------------
    # Directory management
    # -------------------------------------------------------------------------
    if args.partial:
        parent_dir = '/home/khiemphi/SBU/Grasp-Sim/output_data/sim_data/'  #'/home/khiemphi/SBU/Grasp-Sim/output_data/sim_data/' #'ycb_point_cloud/' 
    else: 
        parent_dir = '/home/khiem/Point-Cloud-Decomposition-for-Grasping/ycb_point_cloud/' 
    
    # -------------------------------------------------------------------------
    # Loading Point Cloud
    # -------------------------------------------------------------------------
    object_path = args.object
    print(f"📦 Loading object: {args.object}")
    pcd = o3d.io.read_point_cloud(f"{parent_dir}{object_path}/nontextured.ply")
    pcd.paint_uniform_color([0, 0, 1]) # --> use only pc color 
    if pcd.is_empty():
        sys.exit(f"❌ Error: Loaded point cloud is empty at {object_path}")

    # -------------------------------------------------------------------------
    # Calculate BITSI Metric
    # -------------------------------------------------------------------------
    cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = auto_thickness(cloud_object, scale=0.02)
    print(f"📏 Adaptive thickness selected: {thickness:.6f} (scale=5%)")
    print("⚙️  Computing BITSI metrics...")
    bitsi_x, bitsi_y, bitsi_z, slice_idx = bitsi_metric(cloud_object, epsilon=epsilon, thickness=thickness)
    pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx)
    #visualize_bitsi(bitsi_x, bitsi_y, bitsi_z)

    # -------------------------------------------------------------------------
    # Segmentation With BITSI Metric
    # -------------------------------------------------------------------------
    visualize_bitsi_with_extrema(bitsi_x, bitsi_y, bitsi_z)


if __name__ == "__main__":
    main()