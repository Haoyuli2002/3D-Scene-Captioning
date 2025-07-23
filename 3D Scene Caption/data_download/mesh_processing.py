import open3d as o3d
import numpy as np
import json
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import json
import matplotlib.cm as cm
import k3d

# Define paths to the mesh and segmentation files
scene_dir = 'ScanNet_Data/data/0a7cc12c0e'
mesh_file_dir = scene_dir + '/scans/mesh_aligned_0.05_semantic.ply'
segments_file = scene_dir + '/scans/segments.json'
anno_file = scene_dir + '/scans/segments_anno.json'

def load_mesh(mesh_file=mesh_file_dir):
    """
    Loads a mesh from a PLY file.

    Args:
        mesh_file (str): Path to the .ply mesh file.

    Returns:
        vertices (np.ndarray): (N, 3) array of vertex coordinates.
        triangles (np.ndarray): (M, 3) array of triangle vertex indices.
    """
    print(f"Loading mesh from {mesh_file}")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    print(f"Loaded {len(vertices)} vertices and {len(triangles)} triangles.")
    return vertices, triangles


def parse_segmentation_and_annotation(segments_file=segments_file, anno_file=anno_file):
    """
    Parses segmentation data and instance annotations.

    Args:
        segments_file (str): Path to JSON with 'segIndices' per vertex.
            - The corresponding vertices of each segment.
        anno_file (str): Path to JSON with 'segGroups' metadata.
            - Contains 'objectId', 'label', and 'segments' for each instance.
            - Specify the underlying segments for each instance.

    Returns:
        segment_ids (np.ndarray): (N,) segment ID for each vertex.
        segment_to_object (dict): segment ID → instance ID mapping.
        object_to_label (dict): instance ID → semantic label.
        unique_instances (set): All unique instance IDs found.
    """
    print(f"Loading segment IDs from {segments_file}")
    with open(segments_file) as f:
        segment_ids = np.array(json.load(f)['segIndices'])

    print(f"Loading annotations from {anno_file}")
    with open(anno_file) as f:
        anno_data = json.load(f)['segGroups']

    segment_to_object = {}
    object_to_label = {}

    for obj in anno_data:
        instance_id = obj['objectId']
        label = obj['label']
        for seg_id in obj['segments']:
            segment_to_object[seg_id] = instance_id
        object_to_label[instance_id] = label

    unique_instances = set(segment_to_object.values())
    print(f"Found {len(unique_instances)} unique instances.")
    return segment_ids, segment_to_object, object_to_label, unique_instances


def individualual_mesh_extraction(
    vertices,
    triangles,
    segment_ids,
    segment_to_object,
    object_to_label,
    unique_instances,
    output_dir,
    relaxed=False
):
    """
    Extracts per-instance meshes from a full scene.

    Strategy:
        - For each instance, selects triangles whose all vertices match the instance.
            - If relaxed, selects triangles where at least 2 vertices match the instance.
        - Builds a sub-mesh for the instance.
        - Saves each extracted mesh as a PLY file with semantic labels.

    Args:
        vertices (np.ndarray): (N, 3) scene vertices.
        triangles (np.ndarray): (M, 3) triangle indices.
        segment_ids (np.ndarray): (N,) segment ID per vertex.
            - Every vertex belongs to a segment.
        segment_to_object (dict): segment ID → instance ID.
            - Every segment belongs to an instance.
        object_to_label (dict): instance ID → semantic label.
        unique_instances (set): Set of unique instance IDs.
        output_dir (str): Directory to save .ply meshes.

    Returns:
        None. Saves PLY files to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting meshes to {output_dir}")

    for instance_id in tqdm(unique_instances, desc="Instances"):
        instance_face_indices = []

        # STEP 1: Iterate all triangles and find those that match the instance
        for i, tri in enumerate(triangles):
            segs = [segment_ids[v_idx] for v_idx in tri]
            if relaxed:
                # Relaxed: at least 2 vertices must match the instance
                count = sum([segment_to_object.get(seg) == instance_id for seg in segs])
                if count >= 2:
                    instance_face_indices.append(i)
            else:
                # Strict: all 3 vertices must match the instance
                if all(segment_to_object.get(seg) == instance_id for seg in segs):
                    instance_face_indices.append(i)            
        # If no faces found for this instance, skip
        if not instance_face_indices:
            print(f"Skipping instance {instance_id} (no faces found).")
            continue

        # STEP 2: Extract local submesh
        sub_triangles_global = triangles[instance_face_indices]
        # flatten the triangle into vertices and dedupliate the vertices.
        # return_inverse=True: returns the indices that can be used to reconstruct the original array.
        unique_vertex_indices, inverse_indices = np.unique(sub_triangles_global.flatten(), return_inverse=True)
        sub_vertices = vertices[unique_vertex_indices]
        sub_triangles_local = inverse_indices.reshape(sub_triangles_global.shape)

        # STEP 3: Build Open3D mesh
        submesh = o3d.geometry.TriangleMesh()
        submesh.vertices = o3d.utility.Vector3dVector(sub_vertices)
        submesh.triangles = o3d.utility.Vector3iVector(sub_triangles_local)
        submesh.compute_vertex_normals()

        # STEP 4: Save
        label = object_to_label.get(instance_id, "unknown")
        filename = os.path.join(output_dir, f"object_{instance_id}_{label}.ply")
        o3d.io.write_triangle_mesh(filename, submesh, write_ascii=False)
        print(f"Saved {filename}")

def load_individual_meshes(mesh_dir):
    """
    Loads and concatenates all individual .ply meshes in a directory.

    Args:
        mesh_dir (str): Directory containing individual object .ply meshes.

    Returns:
        all_vertices (list of np.ndarray): List of (Ni, 3) vertex arrays.
        all_faces (list of np.ndarray): List of (Mi, 3) face arrays (with global indices).
        offsets (list of int): Cumulative vertex offsets for indexing (specify which vertices belong to which object).
        object_names (list of str): Filenames of individual objects.
    """
    all_vertices = []
    all_faces = []
    offsets = [0]  # Start with 0 offset
    object_names = os.listdir(mesh_dir)

    for filename in tqdm(object_names, desc="Loading individual meshes"):
        mesh_path = os.path.join(mesh_dir, filename)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        all_vertices.append(vertices)
        all_faces.append(faces + offsets[-1])  # Offset faces by the current total，because faces are defined in the individual mesh
        offsets.append(offsets[-1] + len(vertices))  # Update offset for next mesh
    
    return all_vertices, all_faces, offsets, object_names


def vertex_region_ids(offsets, region_assignments):
    """
    Maps each vertex to its corresponding region ID based on offsets and assignments.

    Args:
        offsets (list of int): Cumulative vertex offsets for indexing.
        region_assignments (list of int): Region IDs for each individual mesh.

    Returns:
        vertex_region_ids (np.ndarray): (N,) array of region IDs for all vertices.
    """
    num_vertices = offsets[-1]
    vertex_region_ids = np.zeros(num_vertices, dtype=int)

    for i in range(len(region_assignments)):
        start = offsets[i]
        end = offsets[i + 1]
        region = region_assignments[i]
        vertex_region_ids[start:end] = region

    return vertex_region_ids

from matplotlib import cm

def coloring(vertex_region_ids, colormap_name='Set3'):
    """
    Maps region IDs to RGB colors for each vertex using a qualitative colormap.

    Args:
        vertex_region_ids (np.ndarray): (N,) array of region IDs.
        colormap_name (str): Matplotlib colormap to use.

    Returns:
        (N, 3) np.ndarray of uint8 RGB colors.
    """
    unique_regions = np.unique(vertex_region_ids)
    num_regions = len(unique_regions)
    print(f"Number of unique regions: {num_regions}")

    # Use qualitative colormap (good for categorical regions)
    cmap = cm.get_cmap(colormap_name, num_regions)
    region_colors = (cmap(range(num_regions))[:, :3] * 255).astype(np.uint8)

    region_id_to_color = {
        rid: region_colors[i]
        for i, rid in enumerate(unique_regions)
    }

    vertex_colors = np.zeros((vertex_region_ids.shape[0], 3), dtype=np.uint8)
    for i in range(vertex_region_ids.shape[0]):
        rid = vertex_region_ids[i]
        vertex_colors[i] = region_id_to_color.get(rid, [150, 150, 150])

    return vertex_colors

def coloring_rainbow(vertex_region_ids):
    """
    Maps region IDs to softer rainbow colors for each vertex.

    Args:
        vertex_region_ids (np.ndarray): (N,) array of region IDs.

    Returns:
        (N, 3) np.ndarray of uint8 RGB colors.
    """
    unique_regions = np.unique(vertex_region_ids)
    num_regions = len(unique_regions)
    print(f"Number of unique regions: {num_regions}")

    # Define strong rainbow colors (RGB)
    rainbow_colors = np.array([
        [255, 0, 0],       # Red
        [255, 127, 0],     # Orange
        [255, 255, 0],     # Yellow
        [0, 255, 0],       # Green
        [0, 255, 255],     # Cyan
        [0, 0, 255],       # Blue
        [139, 0, 255]      # Purple
    ], dtype=np.uint8)

    # Soften colors by blending with white
    blend_factor = 0.4  # 0 = pure white, 1 = original color
    rainbow_colors_soft = (rainbow_colors * blend_factor + 255 * (1 - blend_factor)).astype(np.uint8)

    # Assign region colors
    region_id_to_color = {
        rid: rainbow_colors_soft[i % len(rainbow_colors_soft)]
        for i, rid in enumerate(sorted(unique_regions))
    }

    # Map every vertex to its region's color
    vertex_colors = np.zeros((vertex_region_ids.shape[0], 3), dtype=np.uint8)
    for i in range(vertex_region_ids.shape[0]):
        rid = vertex_region_ids[i]
        vertex_colors[i] = region_id_to_color.get(rid, [200, 200, 200])  # fallback to light gray

    return vertex_colors


def plot_colored_mesh_k3d(vertices, faces, vertex_colors):
    """
    Plots a colored triangular mesh in K3D with per-vertex colors.

    Args:
        vertices (np.ndarray): (N, 3) vertex coordinates.
        faces (np.ndarray): (M, 3) triangle indices.
        vertex_colors (np.ndarray): (N, 3) RGB uint8 colors.

    Returns:
        k3d.plot.Plot: The K3D plot object.
    """
    def rgb_to_uint32(r, g, b):
        return (255 << 24) | (int(r) << 16) | (int(g) << 8) | int(b)

    vertex_colors_uint32 = np.array([
        rgb_to_uint32(r, g, b) for r, g, b in vertex_colors
    ], dtype=np.uint32)

    plot = k3d.plot()
    mesh_plot = k3d.mesh(
        vertices.astype(np.float32),
        faces.astype(np.uint32),
        colors=vertex_colors_uint32
    )
    plot += mesh_plot
    # plot.display()
    return plot

import k3d
import numpy as np

def plot_colored_mesh_k3d_with_labels(vertices, faces, vertex_colors, region_assignments, offsets, object_names):
    """
    Plots the combined mesh in K3D and adds per-region text labels at approximate centroids.
    
    - vertices: (N,3)
    - faces: (M,3)
    - vertex_colors: (N,3)
    - region_assignments: list[int] (per object)
    - offsets: cumulative vertex offsets
    - object_names: filenames (or custom captions)
    """

    def rgb_to_uint32(r, g, b):
        return (255 << 24) | (int(r) << 16) | (int(g) << 8) | int(b)

    vertex_colors_uint32 = np.array([
        rgb_to_uint32(r, g, b) for r, g, b in vertex_colors
    ], dtype=np.uint32)

    plot = k3d.plot()
    
    # Add the mesh
    mesh_plot = k3d.mesh(
        vertices.astype(np.float32),
        faces.astype(np.uint32),
        colors=vertex_colors_uint32
    )
    plot += mesh_plot

    # Compute region centroids
    label_positions = []
    label_strings = []

    for i in range(len(region_assignments)):
        start = offsets[i]
        end = offsets[i + 1]
        region_vertices = vertices[start:end]
        if region_vertices.shape[0] == 0:
            continue
        center = region_vertices.mean(axis=0)
        label_positions.append(center)
        label_strings.append(object_names[i].split('.ply')[0])

    if len(label_positions) > 0:
        label_positions = np.array(label_positions, dtype=np.float32)
        plot += k3d.text(
            '\n'.join(label_strings),
            position=[0, 0, 0],
            color=0x000000,
            size=1
        )
        for i, pos in enumerate(label_positions):
            plot += k3d.label(
                text=label_strings[i],
                position=pos.tolist(),
                color=0x000000,
                size=1
            )

    return plot