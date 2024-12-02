import pyvista as pv
import numpy as np

# Load an STL file
def load_stl(filename):
    return pv.read(filename)

# Calculate the rotation matrix for aligning a plane to a target normal
def calculate_rotation_matrix(points, target_normal):
    # Points should be 3x3 matrix with selected points
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    plane_normal = np.cross(v1, v2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize

    # Rotation to align plane_normal to target_normal
    axis = np.cross(plane_normal, target_normal)
    angle = np.arccos(np.dot(plane_normal, target_normal))
    if np.linalg.norm(axis) < 1e-6:  # Already aligned
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)  # Normalize axis

    # Rotation matrix (Rodrigues' rotation formula)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

# Calculate the rotation matrix for aligning a vector in the XY plane to the X-axis
def calculate_z_rotation_matrix(points):
    # Vector in the XY plane
    v = points[1] - points[0]
    v[2] = 0  # Ensure the vector lies in the XY plane
    v = v / np.linalg.norm(v)  # Normalize

    # Angle to rotate around Z-axis to align with the X-axis
    angle = np.arctan2(v[1], v[0])  # atan2(y, x) gives angle from X-axis
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)

    # Rotation matrix around Z-axis
    R = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return R

# Apply transformation
def apply_transformation(mesh, R):
    mesh.points = np.dot(mesh.points, R.T)  # Apply rotation matrix
    return mesh

def align_plane_or_vector(mesh, target_type, output_filename):
    selected_points = []

    def point_picking_callback(picked_point, picker):
        nonlocal selected_points
        selected_points.append(picked_point)
        print(f"Picked point: {picked_point}")
        
        if target_type == "XY" and len(selected_points) == 3:
            print("3 points selected. Calculating transformation for XY plane...")
            
            # Calculate rotation matrix for the XY plane
            R = calculate_rotation_matrix(np.array(selected_points), np.array([0, 0, 1]))
            
            # Apply transformation
            transformed_mesh = apply_transformation(mesh, R)
            
            # Save transformed mesh
            transformed_mesh.save(output_filename)
            print(f"Transformed mesh saved as {output_filename}")


        elif target_type == "XZ" and len(selected_points) == 2:
            print("2 points selected. Calculating rotation around Z-axis...")
            
            # Calculate rotation matrix for Z-axis alignment
            R = calculate_z_rotation_matrix(np.array(selected_points))
            
            # Apply transformation
            transformed_mesh = apply_transformation(mesh, R)
            
            # Save transformed mesh
            transformed_mesh.save(output_filename)
            print(f"Transformed mesh saved as {output_filename}")


    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightblue")

    # Add a marker for the origin
    origin = pv.Sphere(radius=10, center=(0, 0, 0))
    plotter.add_mesh(origin, color="red", label="Origin")

    # Show axes
    plotter.show_axes()

    # Enable a grid for better orientation
    plotter.show_grid()

    # Enable point picking
    plotter.enable_point_picking(callback=point_picking_callback, use_picker="mesh")
    plotter.show()


# Main function
def main():
    # Step 1: Align to XY plane
    filename = "INPUTSTLFILE.stl"  # Replace with your STL file
    mesh = load_stl(filename)
    print("Aligning to XY plane...")
    align_plane_or_vector(mesh, target_type="XY", output_filename="alignedXY.stl")

    # Step 2: Rotate around Z-axis to align with X-axis
    print("Rotating around Z-axis to align with X-axis...")
    aligned_mesh = load_stl("alignedXY.stl")
    align_plane_or_vector(aligned_mesh, target_type="XZ", output_filename="alignedXYZ.stl")

if __name__ == "__main__":
    main()