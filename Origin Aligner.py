import sys
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree  # Import KDTree for nearest-neighbor search

# Load an STL file
def load_stl(filename):
    """
    Load an STL file using PyVista.
    
    Parameters:
        filename (str): Path to the STL file.
        
    Returns:
        pyvista.PolyData: The loaded STL mesh.
    """
    return pv.read(filename)

# Calculate the rotation matrix for aligning a plane to a target normal
def calculate_rotation_matrix(points, target_normal):
    """
    Calculate a rotation matrix to align a plane defined by three points to a target normal vector.
    
    Parameters:
        points (numpy.ndarray): 3x3 array containing three points that define a plane.
        target_normal (numpy.ndarray): The target normal vector for alignment.
        
    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    plane_normal = np.cross(v1, v2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    axis = np.cross(plane_normal, target_normal)
    angle = np.arccos(np.dot(plane_normal, target_normal))
    if np.linalg.norm(axis) < 1e-6:  # If the normals are already aligned
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis

    # Use Rodrigues' rotation formula to compute the rotation matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

# Apply transformation
def apply_transformation(mesh, R):
    """
    Apply a transformation (rotation) to a mesh.
    
    Parameters:
        mesh (pyvista.PolyData): The input mesh.
        R (numpy.ndarray): The 3x3 rotation matrix.
        
    Returns:
        pyvista.PolyData: The transformed mesh.
    """
    mesh.points = np.dot(mesh.points, R.T)
    return mesh

# Average the k nearest neighbors of a picked point
def average_nearest_points(mesh, picked_point, k=25):
    """
    Compute the average position of the k-nearest neighbors of a picked point on a mesh.
    
    Parameters:
        mesh (pyvista.PolyData): The input mesh.
        picked_point (numpy.ndarray): The picked 3D point.
        k (int): The number of nearest neighbors to average.
        
    Returns:
        numpy.ndarray: The average position of the k-nearest neighbors.
    """
    tree = cKDTree(mesh.points)
    distances, indices = tree.query(picked_point, k=k)
    return np.mean(mesh.points[indices], axis=0)

# Compute the best-fit plane from a set of points
def calculate_best_fit_plane(points):
    """
    Compute the best-fit plane for a set of points using PCA.
    
    Parameters:
        points (numpy.ndarray): An Nx3 array of points.
        
    Returns:
        tuple: A tuple containing the plane normal (numpy.ndarray) and a point on the plane (numpy.ndarray).
    """
    centroid = np.mean(points, axis=0)
    cov_matrix = np.cov(points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[:, 0]  # The eigenvector corresponding to the smallest eigenvalue
    return normal_vector, centroid

# Compute the best-fit direction vector from a set of points
def calculate_best_fit_vector(points):
    """
    Compute the best-fit direction vector for a set of points using PCA.
    
    Parameters:
        points (numpy.ndarray): An Nx3 array of points.
        
    Returns:
        numpy.ndarray: A unit vector representing the best-fit direction.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvectors[:, -1]  # The principal direction

# Handle the plane or vector alignment process
def align_plane_or_vector(mesh, target_type, output_filename):
    """
    Align a plane or vector based on selected points on the mesh.
    
    Parameters:
        mesh (pyvista.PolyData): The input mesh.
        target_type (str): The alignment type ("XY" for plane alignment, "XZ" for vector alignment).
        output_filename (str): The filename to save the transformed mesh.
    """
    selected_points = []
    spheres = []

    def print_status():
        """Display the current alignment status in the terminal."""
        print(f"\n[STATUS] Selected Points: {len(selected_points)}")
        print("Press B to undo the last point, or Space to confirm.")

    def point_picking_callback(picked_point, picker):
        """Callback for point picking."""
        nonlocal selected_points, spheres
        averaged_point = average_nearest_points(mesh, picked_point, k=25)
        selected_points.append(averaged_point)
        sphere = pv.Sphere(radius=1, center=averaged_point)
        spheres.append(plotter.add_mesh(sphere, color="blue"))
        print_status()

    def undo_last_point():
        """Undo the last selected point."""
        nonlocal selected_points, spheres
        if selected_points:
            selected_points.pop()
            last_sphere = spheres.pop()
            plotter.remove_actor(last_sphere)
            print("\n[INFO] Last point removed.")
            print_status()
        else:
            print("\n[INFO] No points to remove.")

    def confirm_points():
        """Confirm selected points and apply the transformation."""
        nonlocal selected_points, plotter
        if target_type == "XY" and len(selected_points) >= 3:
            selected_points_array = np.array(selected_points)
            normal_vector, centroid = calculate_best_fit_plane(selected_points_array)
            R = calculate_rotation_matrix(selected_points_array, np.array([0, 0, 1]))
            transformed_mesh = apply_transformation(mesh, R)
            transformed_mesh.save(output_filename)
            print(f"\n[SUCCESS] Transformed mesh saved as {output_filename}")
            plotter.update()
        elif target_type == "XZ" and len(selected_points) >= 2:
            selected_points_array = np.array(selected_points)
            best_fit_vector = calculate_best_fit_vector(selected_points_array)
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
                R = np.eye(3)
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

# Main function
def main():
    """
    Main entry point for the script. Aligns the mesh to the XY plane and then to the X-axis.
    """
    if len(sys.argv) < 2:
        print("Usage: Drag and drop an STL file onto this executable.")
        return

    filename = sys.argv[1]
    try:
        mesh = load_stl(filename)
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return

    print("\nAligning to XY plane... select 3 or more points to align.")
    align_plane_or_vector(mesh, target_type="XY", output_filename="alignedXY.stl")

    print("\nRotating around Z-axis to align with X-axis... select 2 or more points that should be parallel to the X-axis.")
    aligned_mesh = load_stl("alignedXY.stl")
    align_plane_or_vector(aligned_mesh, target_type="XZ", output_filename="alignedXYZ.stl")

if __name__ == "__main__":
    main()
