import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    input_dir = Path("zero-aneurysmen")
    output_file = Path("canonical_average.obj")
    
    obj_files = list(input_dir.glob("*/aneurysm_aligned.obj"))
    if not obj_files:
        logging.error("No aligned aneurysm meshes found.")
        return
        
    logging.info(f"Found: {len(obj_files)} aligned aneurysm meshes.")
    
    # 1. Template Generation
    logging.info("Generating spherical template...")
    template = trimesh.creation.icosphere(subdivisions=5, radius=1.0)

    # subdivisions=0: 12 vertices
    # subdivisions=1: 42 vertices
    # subdivisions=2: 162 vertices
    # subdivisions=3: 642 vertices
    # subdivisions=4: 2,562 vertices
    # subdivisions=5: 10,242 vertices
    # subdivisions=6: 40,962 vertices
    
    sample_size = min(50, len(obj_files))
    logging.info(f"Calculating average centroid from a sample of {sample_size} meshes...")
    centers = []
    for f in obj_files[:sample_size]:
        try:
            m = trimesh.load(f, force='mesh')
            centers.append(m.centroid)
        except Exception as e:
            logging.warning(f"Error loading {f} for centroid calculation: {e}")
            
    if centers:
        ray_origin = np.mean(centers, axis=0)
    else:
        ray_origin = np.array([0.0, 0.0, 0.0])
        
    logging.info(f"Using Ray Origin (Center): {ray_origin}")
    
    ray_directions = template.vertices.copy()
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
    
    num_rays = len(ray_directions)
    ray_origins = np.tile(ray_origin, (num_rays, 1))
    
    ray_distances = [[] for _ in range(num_rays)]
    
    # 2. Ray-Casting Loop over all meshes
    for obj_file in tqdm(obj_files, desc="Ray-casting Meshes"):
        try:
            mesh = trimesh.load(obj_file, force='mesh')
            
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=True
            )
            
            if len(locations) == 0:
                continue
                
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
            
            max_dist_per_ray = {}
            for r_idx, dist in zip(index_ray, distances):
                if r_idx not in max_dist_per_ray or dist > max_dist_per_ray[r_idx]:
                    max_dist_per_ray[r_idx] = dist
                    
            for r_idx, max_dist in max_dist_per_ray.items():
                ray_distances[r_idx].append(max_dist)
                
        except Exception as e:
            logging.warning(f"Error processing {obj_file}: {e}")
            
    # 4. Averaging
    logging.info("Calculating average distances...")
    final_distances = np.zeros(num_rays)
    valid_rays = np.zeros(num_rays, dtype=bool)
    
    min_hits_required = len(obj_files) * 0.1
    
    for i in range(num_rays):
        dists = ray_distances[i]
        if len(dists) > min_hits_required:
            final_distances[i] = np.median(dists)
            valid_rays[i] = True
        else:
            final_distances[i] = 0.0
            valid_rays[i] = False
            
    # 5. Reconstruction
    logging.info("Reconstructing Canonical Mesh...")
    new_vertices = ray_origin + ray_directions * final_distances[:, np.newaxis]
    canonical_mesh = trimesh.Trimesh(vertices=new_vertices, faces=template.faces)
    
    valid_faces_mask = valid_rays[canonical_mesh.faces].all(axis=1)
    canonical_mesh.update_faces(valid_faces_mask)
    canonical_mesh.remove_unreferenced_vertices()
    
    # 6. Cylinder cut based on average ostium points
    logging.info("Calculating cylinder radius from ostium points...")
    all_ostium_points = []
    for ostium_file in input_dir.glob("*/04_subpointclouds/subpointcloud_label_2.ply"):
        try:
            pc = trimesh.load(ostium_file)
            if hasattr(pc, 'vertices') and len(pc.vertices) > 0:
                all_ostium_points.append(pc.vertices)
        except:
            pass
            
    if all_ostium_points:
        combined_ostium = np.vstack(all_ostium_points)
        
        # Since the meshes are aligned (Ostium at Z~0, Normal=[0,0,1]),
        # we can calculate the radius in the XY plane.
        xy_distances = np.linalg.norm(combined_ostium[:, :2], axis=1)
        
        # We take the 50th percentile to ensure the cylinder is large enough
        # to capture the funnel, but not too large to damage the walls.
        cylinder_radius = np.percentile(xy_distances, 50)
        logging.info(f"Calculated cylinder radius (50th percentile): {cylinder_radius:.4f}")
        
        # We only cut in the lower part of the aneurysm (e.g., Z < 0.12),
        # so we don't accidentally cut off the top of the aneurysm if it's narrow.
        z_threshold = 0.12
        
        # Find vertices of the Canonical Mesh that are INSIDE the cylinder AND AT THE BOTTOM
        mesh_xy_dist = np.linalg.norm(canonical_mesh.vertices[:, :2], axis=1)
        mesh_z = canonical_mesh.vertices[:, 2]
        
        vertices_to_remove = (mesh_xy_dist < cylinder_radius) & (mesh_z < z_threshold)
        
        # Keep only faces that DO NOT contain a vertex that should be removed
        faces_to_keep = ~vertices_to_remove[canonical_mesh.faces].any(axis=1)
        
        canonical_mesh.update_faces(faces_to_keep)
        canonical_mesh.remove_unreferenced_vertices()
        logging.info(f"Cylinder cut performed. Removed {np.sum(vertices_to_remove)} vertices.")
        
    else:
        logging.warning("No ostium points found. Skipping cylinder cut.")
    
    # Apply light Laplace smoothing
    logging.info("Applying Laplace smoothing...")
    trimesh.smoothing.filter_laplacian(canonical_mesh, iterations=5)
    
    # Create output directory structure
    output_dir = Path("canonical_model")
    (output_dir / "05_submeshes").mkdir(parents=True, exist_ok=True)
    (output_dir / "04_subpointclouds").mkdir(parents=True, exist_ok=True)
    (output_dir / "07_other").mkdir(parents=True, exist_ok=True)
    
    # Export Mesh
    mesh_output_file = output_dir / "05_submeshes" / "canonical_average.obj"
    logging.info(f"Exporting to {mesh_output_file}...")
    canonical_mesh.export(mesh_output_file)
    
    # --- Extract Ostium Pointcloud and Metadata ---
    logging.info("Extracting ostium points (boundary vertices)...")
    edges = canonical_mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_vertices_indices = np.unique(boundary_edges)
    boundary_vertices = canonical_mesh.vertices[boundary_vertices_indices]
    
    if len(boundary_vertices) > 0:
        # 1. Save pointcloud
        pc = trimesh.points.PointCloud(boundary_vertices)
        pc_file = output_dir / "04_subpointclouds" / "subpointcloud_label_2.ply"
        pc.export(pc_file)
        logging.info(f"Ostium pointcloud saved to {pc_file}")
        
        # 2. Calculate and save centroid
        centroid_ostium = np.mean(boundary_vertices, axis=0)
        centroid_file = output_dir / "07_other" / "centroid_ostium.npy"
        np.save(centroid_file, centroid_ostium)
        logging.info(f"Centroid saved to {centroid_file}")
        
        # 3. Save normal vector (since aligned, it is [0, 0, 1])
        normal_vector = np.array([0.0, 0.0, 1.0])
        normal_file = output_dir / "07_other" / "normal_vector.npy"
        np.save(normal_file, normal_vector)
        logging.info(f"Normal vector saved to {normal_file}")
    else:
        logging.warning("Could not find boundary vertices for the ostium!")
        
    # Also save a copy in the main directory for easy access
    canonical_mesh.export("canonical_average.obj")
    
    logging.info("Done!")

if __name__ == "__main__":
    main()
