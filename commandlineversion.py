import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree  # Import KDTree for nearest-neighbor search

# Load an STL file
def load_stl(filename):
    return pv.read(filename)

# Calculate the rotation matrix for aligning a plane to a target normal
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

# Apply transformation
def apply_transformation(mesh, R):
    mesh.points = np.dot(mesh.points, R.T)
    return mesh

# Average the 25 nearest neighbors of a picked point
def average_nearest_points(mesh, picked_point, k=25):
    """
    Compute the average position of the k-nearest neighbors of a picked point.

    Parameters:
        mesh (pyvista.PolyData): The mesh being analyzed.
        picked_point (numpy.ndarray): The 3D coordinates of the selected point.
        k (int): The number of nearest neighbors to average.

    Returns:
        numpy.ndarray: The average position of the k-nearest neighbors.
    """
    # Build a KDTree for the mesh points
    tree = cKDTree(mesh.points)

    # Find the indices of the k-nearest neighbors
    distances, indices = tree.query(picked_point, k=k)

    # Compute the average position of these neighbors
    average_position = np.mean(mesh.points[indices], axis=0)
    return average_position

# Compute the best-fit plane from a set of points
def calculate_best_fit_plane(points):
    """
    Compute the best-fit plane for a set of points.

    Parameters:
        points (numpy.ndarray): An Nx3 array of points.

    Returns:
        tuple: A tuple containing the plane normal (numpy.ndarray) and a point on the plane (numpy.ndarray).
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(points - centroid, rowvar=False)

    # Perform eigen decomposition to find the normal vector
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[:, 0]  # The eigenvector corresponding to the smallest eigenvalue

    return normal_vector, centroid

# Compute the best-fit direction vector from a set of points
def calculate_best_fit_vector(points):
    """
    Compute the best-fit direction vector for a set of points.

    Parameters:
        points (numpy.ndarray): An Nx3 array of points.

    Returns:
        numpy.ndarray: A unit vector representing the best-fit direction.
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Subtract the centroid to get relative positions
    centered_points = points - centroid

    # Compute the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Perform eigen decomposition to find the principal direction
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The eigenvector corresponding to the largest eigenvalue is the principal direction
    direction_vector = eigenvectors[:, -1]  # Largest eigenvalue

    return direction_vector

def align_plane_or_vector(mesh, target_type, output_filename):
    selected_points = []
    spheres = []  # Store sphere actors for each selected point

    def print_status():
        """Print the current status to the command line."""
        print(f"\n[STATUS] Selected Points: {len(selected_points)}")
        print("Press B to undo the last point, or Space to confirm.")

    def point_picking_callback(picked_point, picker):
        nonlocal selected_points, spheres

        # Compute the averaged position of the 25 nearest neighbors
        averaged_point = average_nearest_points(mesh, picked_point, k=25)

        selected_points.append(averaged_point)
        print(f"\n[INFO] Point added.")

        # Add a blue sphere at the averaged point
        sphere = pv.Sphere(radius=1, center=averaged_point)
        spheres.append(plotter.add_mesh(sphere, color="blue"))

        print_status()

    def undo_last_point():
        nonlocal selected_points, spheres
        if selected_points:
            # Remove the last selected point and its corresponding sphere
            selected_points.pop()
            last_sphere = spheres.pop()
            plotter.remove_actor(last_sphere)
            print("\n[INFO] Last point removed.")
            print_status()
        else:
            print("\n[INFO] No points to remove.")

    def confirm_points():
        nonlocal selected_points, plotter
        if target_type == "XY" and len(selected_points) >= 3:
            print("\n[INFO] Points confirmed. Proceeding with transformation...")

            # Calculate the best-fit plane
            selected_points_array = np.array(selected_points)
            normal_vector, centroid = calculate_best_fit_plane(selected_points_array)

            # Use the normal vector to align to the XY plane
            R = calculate_rotation_matrix(selected_points_array, np.array([0, 0, 1]))

            transformed_mesh = apply_transformation(mesh, R)
            transformed_mesh.save(output_filename)
            print(f"\n[SUCCESS] Transformed mesh saved as {output_filename}")
            print(f"\nPlease close the current 3d viewer window to proceed to Rotation alignment")
            plotter.update()
        elif target_type == "XZ" and len(selected_points) >= 2:
            print("\n[INFO] Points confirmed. Proceeding with transformation...")

            # Calculate the best-fit direction vector
            selected_points_array = np.array(selected_points)
            best_fit_vector = calculate_best_fit_vector(selected_points_array)

            # Use the best-fit vector to align to the X-axis
            axis = np.cross(best_fit_vector, np.array([1, 0, 0]))
            angle = np.arccos(np.dot(best_fit_vector, np.array([1, 0, 0])))

            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            else:
                R = np.eye(3)  # Already aligned

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

    # Enable point picking
    plotter.enable_point_picking(
        callback=point_picking_callback,
        use_picker="cell",  # Use cell picker to ensure surface selection
    )

    # Add key event handlers
    plotter.add_key_event("b", undo_last_point)  # Undo last point
    plotter.add_key_event("space", confirm_points)  # Confirm points

    print_status()  # Initial status
    plotter.show()


# Main function
def main():
    # Step 1: Align to XY plane
    filename = "INPUTSTLFILE.stl"  # Replace with your STL file
    mesh = load_stl(filename)
    print("\nAligning to XY plane... select 3 or more points to align")
    align_plane_or_vector(mesh, target_type="XY", output_filename="alignedXY.stl")

    # Step 2: Rotate around Z-axis to align with X-axis
    print("\nRotating around Z-axis to align with X-axis... select two or more points that should be parrallel to the X-Axis")
    aligned_mesh = load_stl("alignedXY.stl")
    align_plane_or_vector(aligned_mesh, target_type="XZ", output_filename="alignedXYZ.stl")

if __name__ == "__main__":
    main()
