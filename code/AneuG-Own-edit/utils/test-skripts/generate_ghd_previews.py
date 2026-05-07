import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import numpy as np

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_mesh(ax, mesh, title, color, alpha):
    ax.set_title(title)
    
    # Extract vertices and faces
    v = mesh.vertices
    f = mesh.faces
    
    # Plot using trisurf
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color=color, alpha=alpha, edgecolor='none', shade=True)
    
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Set equal aspect ratio
    set_axes_equal(ax)

def generate_preview(target_path, warped_path, save_path, folder_name):
    # Load meshes
    try:
        mesh_target = trimesh.load(target_path, force='mesh')
        mesh_warped = trimesh.load(warped_path, force='mesh')
    except Exception as e:
        print(f"Error loading meshes for {folder_name}: {e}")
        return

    fig = plt.figure(figsize=(14, 7))
    
    # Plot Target (Left)
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh(ax1, mesh_target, "target", color='lightgrey', alpha=0.3)
    
    # Plot Warped (Right)
    ax2 = fig.add_subplot(122, projection='3d')
    plot_mesh(ax2, mesh_warped, "warped", color='deepskyblue', alpha=0.8)

    fig.suptitle(f"Fitting preview | {folder_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved image: {save_path}")

def process_directories(source_root, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Walk through the directory
    for entry in os.listdir(source_root):
        subdir_path = os.path.join(source_root, entry)
        
        # Check if directory
        if os.path.isdir(subdir_path):
            # Paths to OBJ files
            # Adjust path if structure is different. Assuming vanilla/viz based on previous context.
            viz_dir = os.path.join(subdir_path, 'vanilla', 'viz')
            target_obj = os.path.join(viz_dir, 'target.obj')
            warped_obj = os.path.join(viz_dir, 'warped_epoch_14800.obj')
            
            if os.path.exists(target_obj) and os.path.exists(warped_obj):
                save_path = os.path.join(output_dir, f"{entry}.png")
                
                # Check if image already exists to avoid re-generating (optional)
                # if not os.path.exists(save_path):
                generate_preview(target_obj, warped_obj, save_path, entry)
            else:
                # Silent skip or verify logic as needed
                # print(f"Skipping {entry}: Missing OBJ files.")
                pass

if __name__ == "__main__":
    # Define root paths
    # Assuming script is run from project root
    ROOT_DIR = os.getcwd() # Or specific path
    SOURCE_DIR = os.path.join(ROOT_DIR, "checkpoints", "ghd_fitting_output_15000")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "checkpoints", "ghd_fitting_output_15000_previews")
    
    if os.path.exists(SOURCE_DIR):
        print(f"Processing directories in {SOURCE_DIR}...")
        process_directories(SOURCE_DIR, OUTPUT_DIR)
        print("Done.")
    else:
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")