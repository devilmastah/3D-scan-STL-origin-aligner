import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

def load_stl(filename):
    return pv.read(filename)

def calculate_rotation_matrix(points, target_normal):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    plane_normal = np.cross(v1, v2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    axis = np.cross(plane_normal, target_normal)
    angle = np.arccos(np.dot(plane_normal, target_normal))
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def apply_transformation(mesh, R):
    mesh.points = np.dot(mesh.points, R.T)
    return mesh

def average_nearest_points(mesh, picked_point, k=5):
    tree = cKDTree(mesh.points)
    distances, indices = tree.query(picked_point, k=k)
    average_position = np.mean(mesh.points[indices], axis=0)
    return average_position

def calculate_best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    cov_matrix = np.cov(points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[:, 0]
    return normal_vector, centroid

def calculate_best_fit_vector(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    direction_vector = eigenvectors[:, -1]
    return direction_vector

def align_plane_or_vector(mesh, target_type, output_filename):
    selected_points = []
    spheres = []
    plane_actor = None

    def print_status():
        print(f"\n[STATUS] Selected Points: {len(selected_points)}")
        print("Press B to undo the last point, Space to confirm, or keys 1-9 to remove specific points.")

    def update_plane_xy():
        nonlocal plane_actor
        if len(selected_points) >= 3 and target_type == "XY":
            selected_points_array = np.array(selected_points)
            normal_vector, centroid = calculate_best_fit_plane(selected_points_array)
            plane_size = mesh.bounds[1] - mesh.bounds[0]
            plane = pv.Plane(center=centroid, direction=normal_vector, i_size=plane_size, j_size=plane_size)
            if plane_actor:
                plotter.remove_actor(plane_actor)
            plane_actor = plotter.add_mesh(plane, color="green", opacity=0.5, pickable=False)

    def update_plane_z_rotation():
        nonlocal plane_actor
        if len(selected_points) >= 3 and target_type == "XZ":
            selected_points_array = np.array(selected_points)
            normal_vector, centroid = calculate_best_fit_plane(selected_points_array)
            plane_size = mesh.bounds[1] - mesh.bounds[0]
            plane = pv.Plane(center=centroid, direction=normal_vector, i_size=plane_size, j_size=plane_size)
            if plane_actor:
                plotter.remove_actor(plane_actor)
            plane_actor = plotter.add_mesh(plane, color="orange", opacity=0.5, pickable=False)

    def point_picking_callback(picked_point, picker):
        nonlocal selected_points, spheres
        averaged_point = average_nearest_points(mesh, picked_point, k=5)
        selected_points.append(averaged_point)
        print(f"\n[INFO] Point added.")
        sphere = pv.Sphere(radius=1, center=averaged_point)
        spheres.append(plotter.add_mesh(sphere, color="blue"))
        if target_type == "XY":
            update_plane_xy()
        elif target_type == "XZ":
            update_plane_z_rotation()
        print_status()

    def undo_last_point():
        nonlocal selected_points, spheres
        if selected_points:
            selected_points.pop()
            last_sphere = spheres.pop()
            plotter.remove_actor(last_sphere)
            if target_type == "XY":
                update_plane_xy()
            elif target_type == "XZ":
                update_plane_z_rotation()
            print("\n[INFO] Last point removed.")
            print_status()
        else:
            print("\n[INFO] No points to remove.")

    def confirm_points():
        nonlocal selected_points, plotter
        if target_type == "XY" and len(selected_points) >= 3:
            print("\n[INFO] Points confirmed. Proceeding with transformation...")
            selected_points_array = np.array(selected_points)
            normal_vector, centroid = calculate_best_fit_plane(selected_points_array)
            R = calculate_rotation_matrix(selected_points_array, np.array([0, 0, 1]))
            transformed_mesh = apply_transformation(mesh, R)
            transformed_mesh.save(output_filename)
            print(f"\n[SUCCESS] Transformed mesh saved as {output_filename}")
            plotter.update()
        elif target_type == "XZ" and len(selected_points) >= 3:
            print("\n[INFO] Points confirmed. Proceeding with transformation...")
            selected_points_array = np.array(selected_points)
            projected_points = selected_points_array.copy()
            projected_points[:, 2] = 0
            best_fit_vector = calculate_best_fit_vector(projected_points)
            angle = np.arctan2(best_fit_vector[1], best_fit_vector[0])
            cos_angle = np.cos(-angle)
            sin_angle = np.sin(-angle)
            R = np.array([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            transformed_mesh = apply_transformation(mesh, R)
            transformed_mesh.save(output_filename)
            print(f"\n[SUCCESS] Transformed mesh saved as {output_filename}")
            plotter.update()
        else:
            print("\n[ERROR] Insufficient points selected. Please select the required points.")

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightblue", pickable=True)
    plotter.add_mesh(pv.Sphere(radius=1, center=(0, 0, 0)), color="red", label="Origin")
    plotter.show_axes()
    plotter.show_grid()
    plotter.enable_point_picking(callback=point_picking_callback, use_picker="cell")
    plotter.add_key_event("b", undo_last_point)
    plotter.add_key_event("space", confirm_points)

    print_status()
    plotter.show()

def main():
    filename = "INPUTSTLFILE.stl"
    mesh = load_stl(filename)
    print("\nAligning to XY plane... select 3 or more points to align")
    align_plane_or_vector(mesh, target_type="XY", output_filename="alignedXY.stl")
    print("\nRotating around Z-axis to align with X-axis... select three or more points")
    aligned_mesh = load_stl("alignedXY.stl")
    align_plane_or_vector(aligned_mesh, target_type="XZ", output_filename="alignedXYZ.stl")

if __name__ == "__main__":
    main()
