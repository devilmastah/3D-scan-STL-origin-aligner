import sys
import pyvista as pv
import numpy as np

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

# Calculate the rotation matrix for aligning a vector in the XY plane to the X-axis
def calculate_z_rotation_matrix(points):
    v = points[1] - points[0]
    v[2] = 0
    v = v / np.linalg.norm(v)

    angle = np.arctan2(v[1], v[0])
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)

    R = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return R

# Apply transformation
def apply_transformation(mesh, R):
    mesh.points = np.dot(mesh.points, R.T)
    return mesh

def align_plane_or_vector(mesh, target_type, output_filename):
    selected_points = []

    def point_picking_callback(picked_point, picker):
        nonlocal selected_points
        selected_points.append(picked_point)
        print(f"Picked point: {picked_point}")

        if target_type == "XY" and len(selected_points) == 3:
            print("3 points selected. Calculating transformation for XY plane...")
            R = calculate_rotation_matrix(np.array(selected_points), np.array([0, 0, 1]))
            transformed_mesh = apply_transformation(mesh, R)
            transformed_mesh.save(output_filename)
            print(f"Transformed mesh saved as {output_filename}")

        elif target_type == "XZ" and len(selected_points) == 2:
            print("2 points selected. Calculating rotation around Z-axis...")
            R = calculate_z_rotation_matrix(np.array(selected_points))
            transformed_mesh = apply_transformation(mesh, R)
            transformed_mesh.save(output_filename)
            print(f"Transformed mesh saved as {output_filename}")

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightblue")
    plotter.add_mesh(pv.Sphere(radius=10, center=(0, 0, 0)), color="red", label="Origin")
    plotter.show_axes()
    plotter.show_grid()
    plotter.enable_point_picking(callback=point_picking_callback, use_picker="mesh")
    plotter.show()

# Main function
def main():
    # Check if a filename is provided
    if len(sys.argv) < 2:
        print("Usage: Drag and drop an STL file onto this executable.")
        return

    # Get the filename from the command-line arguments
    filename = sys.argv[1]
    try:
        mesh = load_stl(filename)
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return

    # Step 1: Align to XY plane
    print("Aligning to XY plane...")
    align_plane_or_vector(mesh, target_type="XY", output_filename="alignedXY.stl")

    # Step 2: Rotate around Z-axis to align with X-axis
    print("Rotating around Z-axis to align with X-axis...")
    aligned_mesh = load_stl("alignedXY.stl")
    align_plane_or_vector(aligned_mesh, target_type="XZ", output_filename="alignedXYZ.stl")

if __name__ == "__main__":
    main()
