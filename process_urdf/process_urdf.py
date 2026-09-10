#!/usr/bin/env python
import numpy as np
import json
import os 
import tf_ros as tf
import time
from utils_urdf import *
from utils_archive import *
import argparse

import collections
try:
    collections.Iterable
except AttributeError:
    import collections.abc
    collections.Iterable = collections.abc.Iterable

def make_parser():
    """ Input Parser """
    parser = argparse.ArgumentParser(description='Process urdf and create the sphere controller.')
    parser.add_argument('--robot_path', type=str, help='Path to urdf')
    parser.add_argument('--base_link', type=str, default=None,
                        help='Father link of all fingers in robot.urdf. If omitted, the URDF root is detected.')
    parser.add_argument('--thumb_anchor', type=float, help='Theta Position to place the thumb in', default=1.571)
    parser.add_argument('--verbose', type=bool, help='Running Program in verbose mode',
                        default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--left', type=bool, help='Invert anchors for Left hands',
                        default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--correct_axes', type=bool, help='Correct the joint positions to center to links',
                        default=True, action = argparse.BooleanOptionalAction)
    return parser

if __name__ == "__main__":
    # Load params and set Variables
    args = make_parser().parse_args()
    robot_dir = args.robot_path
    verbosity = args.verbose
    left = args.left
    base_link = args.base_link
    correct_joint_axis = args.correct_axes
    thumb_anchor = args.thumb_anchor
    anchor_values = [0.0, 0.0, 0.0, 0.0, thumb_anchor] 
    merge_fingers = [0, 1]
    control_sphere_samples = 20000
    num_skeleton_points = 200
    anchor_values = np.array(anchor_values)

    # Robot folder Dir
    robot_folder = os.path.dirname(robot_dir)

    # Load config.json if it exists
    config_path = os.path.join(robot_folder, "config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        q_0_dict = config.get("opened_dofs", None)
    except FileNotFoundError:
        print(f"Config File not Found at {config_path}")
        q_0_dict = None
    
    with open(config_path, 'r') as f2:
        config2 = json.load(f2)
        print(f"Config file found at {config_path}")
        q_opened = config2.get("opened_dofs", None)
    print()

    # Load URDF
    robot = load_urdf(robot_dir)

    link_names = {link.name for link in robot.links}
    if base_link is None:
        base_link = detect_base_link(robot)
        print(f"Auto-detected base_link: {base_link}")
    elif base_link not in link_names:
        raise ValueError(
            f"base_link '{base_link}' is not a link in the URDF. "
            f"Available links: {sorted(link_names)}"
        )

    # Load meshes
    link_meshes = load_link_meshes(robot_dir, robot)

    # Example: Print mesh info and optionally visualize
    if verbosity:
        for link_name, meshes in link_meshes.items():
            print(f"Link: {link_name}")

            for mesh_type in ["visual", "collision"]:
                if meshes[mesh_type]:
                    for i, (mesh, scale, origin) in enumerate(meshes[mesh_type]):
                        print(f"  {mesh_type.capitalize()} Mesh {i}:")
                        print(f"    Vertices: {len(mesh.vertices)}")
                        print(f"    Faces: {len(mesh.faces)}")
                        print(f"    Transform: {origin}")
                        print(f"    Scale: {scale}")
            print()
                
        print()
        

    
    # Analyze joints
    joint_info, kin_chains, vertices_dict, meshes_dict = get_unfixed_kin_chains(robot_dir, link_meshes, base_link, correct_joint_axis)

    # Verbosity 
    for joint in joint_info.keys():
        print(joint, ": ", joint_info[joint])

    # Get all finger chains
    finger_chains = [ # Finger chains 
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]
    n_fingers = len(finger_chains)
    print()
    print(n_fingers, "fingers in gripper")
    print()


    # Add fingertip points
    for finger in finger_chains:
        fingertip = finger[-1]
        z = np.linspace(0, 5, 20) * 0.01  
        ft_points = np.stack((np.zeros(z.flatten().shape), np.zeros(z.flatten().shape), z.flatten()), axis=1)
        y = np.linspace(0, 1.25, 20) * 0.01  
        ft_points2 = np.stack((np.zeros(y.flatten().shape), y.flatten(), np.zeros(y.flatten().shape)), axis=1)
        x = np.linspace(0, 2.5, 20) * 0.01  
        ft_points3 = np.stack((x.flatten(), np.zeros(x.flatten().shape),np.zeros(x.flatten().shape)), axis=1)
        vertices_dict[fingertip] = np.vstack((ft_points, ft_points2, ft_points3))
    
    if verbosity:
        print("Visualizing meshes at initial configuration:")
        visualize_meshes_in_base_frame(meshes_dict, joint_info, kin_chains, q_0_dict, "Initial Mesh Configuration", base_link)
        print()
    construct_plane(vertices_dict, joint_info, kin_chains, base_link)

    new_dict = dict()
    for chain in kin_chains:
        new_dict[chain[-1]] = vertices_dict[chain[-1]]

    # Allign palm normal frame to root finger plane
    viz_dict = vertices_dict.copy()
    viz_dict["palm_normal"] = viz_dict["palm_normal"][0:20]
    if verbosity:
        print("Palm Normal and Fingertip Visualization:")
        visualize_vertices_in_base_frame(viz_dict, joint_info, kin_chains, q_0_dict, "Palm Normal and Fingertip Visualization", base_link)
        print()

    joint_type_info = {}
    # Process joint types ABC
    type_ABC_info = process_type_ABC_joints(finger_chains, joint_info, kin_chains, vertices_dict, meshes_dict, q_0_dict) 
    joint_type_info.update(type_ABC_info)

    print("Solving Max Reach IK")
    q_max_dict = solve_inverse_kinematics(finger_chains, joint_info, kin_chains, vertices_dict, joint_type_info, q_opened)
    print()

    if verbosity:
        print("Visualizing meshes at Q Max configuration:")
        visualize_meshes_in_base_frame(meshes_dict, joint_info, kin_chains, q_max_dict, "Flexed Hand Configuration (Phi Max)", base_link)
        print()

    n_axis_points = construct_sphere_with_palm(vertices_dict, joint_info, kin_chains, q_0_dict, q_max_dict, base_link)

    # Reallign, reprocess ABC
    del vertices_dict["palm_normal"]

    if verbosity:
        print("Showing sphere with Initial Hand configuration:")
        visualize_vertices_in_base_frame(vertices_dict, joint_info, kin_chains, q_0_dict, "Sphere Created", base_link)

    viz_dict = dict()
    viz_dict["sphere_frame"] = vertices_dict["sphere_frame"][:-n_axis_points]

    p = sample_fibonacci_points(1000)
    theta, phi = p[:, 0], p[:, 1]
    radius = joint_info["sphere_frame"][6]  # Assuming radius is the 7th element
    phi_breaks = [0, 3 * np.pi / 4, np.pi]
    radius_offsets = [-1 , np.sqrt(2), 2]
    sphere_points = compute_axisymmetric_deformed_points(phi_breaks, radius_offsets, p, base_radius=radius)

    # Generate sphere x-axis points in sphere frame
    radius_offset = np.linspace(1, 1.5, 10) * radius  # 5 samples for radius scaling
    phi_plane = np.linspace(0, np.pi, 20)  # 20 samples for polar angle
    phi_plane, radius_offset = np.meshgrid(phi_plane, radius_offset)
    x = radius_offset * np.sin(phi_plane)
    z = radius_offset * np.cos(phi_plane)
    axis_points = np.stack((x.flatten(), np.zeros(z.flatten().shape), z.flatten()), axis=1)
    
    # Corresponding theta for axis_points (all zeros since they lie in x-z plane)
    theta_axis = np.zeros_like(phi_plane.flatten())
    
    # Center point
    center_point = np.array([[0, 0, 0]])
    theta_center = np.array([0])  # Assign theta=0 for center
    phi_center = np.array([0])    # Assign phi=0 for center

    # Combine all points, theta, and phi
    all_points = np.vstack((center_point, sphere_points))
    all_theta = np.concatenate((theta_center, theta))
    all_phi = np.concatenate((phi_center, phi))
    
    # Create finger_print_dict
    viz_finger_print_dict = {
        "sphere_frame": {
            "points": all_points,
            "theta": all_theta,
            "phi": all_phi
        }
    }
    theta_list = [0, np.deg2rad(60), np.deg2rad(120), np.deg2rad(180), np.deg2rad(240), np.deg2rad(300)]
    theta_list = [2*np.pi-0.6, 2*np.pi-0.2, 0.2, 0.6, 1.7]
    phi_list = [0, np.pi/4,3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4]

    # Compute Main joints
    main_joints, theta_ranges = compute_main_joints(joint_info, kin_chains, q_0_dict, q_max_dict, joint_type_info, base_link)
    print("Finger Chains", finger_chains)
    print("Main Joints: ", main_joints)
    print("Theta Ranges: ", theta_ranges)
    print()

    # Allign sphere
    theta_alignment, finger_chain_controller_indices = compute_finger_chain_order_translation(joint_info, kin_chains, theta_ranges, q_0_dict, left, base_link)

    # Merged Fingers Override
    chain_indices = []
    c = 0
    for chain in finger_chains:
        if (merge_fingers[0] not in finger_chain_controller_indices) and finger_chain_controller_indices[c]==merge_fingers[1]:
            chain_indices.append(merge_fingers)
        elif (merge_fingers[1] not in finger_chain_controller_indices) and finger_chain_controller_indices[c]==merge_fingers[0]:
            chain_indices.append(merge_fingers)
        else:
            chain_indices.append([finger_chain_controller_indices[c]])
        c+=1 
    print()

    print("Finger Chain order", finger_chain_controller_indices)
    print("Finger Chain indices", chain_indices)
    middle_index = finger_chain_controller_indices.index(2) 

    # Process fingertips
    process_bounding_boxes(finger_chains, vertices_dict, joint_info, joint_type_info, q_0_dict, main_joints)
    print()
    print()

    if True:
        print("Joint type info")
        for joint in joint_type_info.keys():
            print(joint, ": ", joint_type_info[joint])
        print() 
    

    # Show sphere alignment
    if verbosity:
        print("Showing alligned sphere with Initial Hand configuration:")
        visualize_vertices_in_base_frame(vertices_dict, joint_info, kin_chains, q_0_dict, "Alligned Sphere", base_link)
        print() 
    print() 

    # Process Type D joints
    type_D_info = process_type_D_joints(finger_chains, main_joints, joint_type_info)
    for joint in type_D_info.keys():
        joint_type_info[joint].update(type_D_info[joint])

    print("------------------------------")

    # Create Finger Print
    finger_print_dict = spherical_raycasts(
        meshes_dict,
        joint_info,
        kin_chains,
        q_max_dict,
        1000,
        base_link
    )

    fp_dict = {}
    for key in finger_print_dict.keys():
        if finger_print_dict[key]:
            fp_dict[key] = finger_print_dict[key]["points"]

    fp_7_point_sample = compute_7_point_sample_skeleton(finger_print_dict, joint_info, kin_chains, q_0_dict)
    fp_dict_7 = {k: np.array(v) for k, v in fp_7_point_sample.items() if len(v) > 0}

    if verbosity:
        visualize_vertices_in_base_frame(fp_dict_7, joint_info, kin_chains, q_0_dict, "7-Point Fingerprint Sample (Homogeneous Observations)", base_link)

    # Get transformation matrices
    T_base_to_sphere = tf.quaternion_matrix(joint_info["sphere_frame"][1])
    T_base_to_sphere[:3, 3] = joint_info["sphere_frame"][0]
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)

    T_base_to_palm = tf.quaternion_matrix(joint_info["palm_normal"][1])
    T_base_to_palm[:3, 3] = joint_info["palm_normal"][0]
    T_palm_to_base = np.linalg.inv(T_base_to_palm)

    # Get sphere points to define fp -> Build lookup dictionaries, 
    radius = joint_info["sphere_frame"][6]
    p = sample_fibonacci_points(4000)
    theta, phi = p[:, 0], p[:, 1]
    phi_breaks = [0, np.pi/2, 3 * np.pi / 4, np.pi]
    radius_values = [0, 0, np.sqrt(2), 2]
    sphere_points = compute_axisymmetric_deformed_points(phi_breaks, radius_values, p, base_radius=radius)

    fp_dict["sphere_frame"] = sphere_points

    del fp_dict["sphere_frame"]

    middle_mj = main_joints[middle_index]

    if True:
        print("Joint type info")
        for joint in joint_type_info.keys():
            print(joint, ": ", joint_type_info[joint])
        print() 
    theta_anchors = process_anchors(finger_chain_controller_indices, anchor_values)

    sphere_control_dict = dict()
    projected_fp_dict = dict()
    q_sphere = q_0_dict.copy()
    q_max_phi = q_0_dict.copy()
    c = 0 
    theta_min = np.zeros_like(anchor_values)
    theta_max = np.zeros_like(anchor_values)
    for joint in main_joints:
        chain_idx = chain_indices[c]
        chain_anchor_values = anchor_values[chain_idx]
        merge_anchor = None
        
        if len(chain_idx) == 1:
            if chain_idx[0] in merge_fingers:
                m_list = merge_fingers.copy()
                print("m_list", m_list)
                m_list.remove(chain_idx[0])
                merge_anchor = anchor_values[m_list]
        print("chain_idx", chain_idx)

        projected_fp_sample, sphere_control, q_finger, q_phi, min_offset, max_offset = build_lookup_dict_with_fp_and_phi(
            fp_7_point_sample,
            joint,
            q_0_dict,
            joint_info,
            kin_chains,
            joint_type_info,
            main_joints,
            chain_idx,
            merge_anchor,
            anchor_override = chain_anchor_values,
            realign_sphere = False,
            resolution = 0.005,
            N = 200,
            base_link = base_link,
        )
        sphere_control_dict.update(sphere_control)
        projected_fp_dict.update(projected_fp_sample)
        chain = next((c for c in kin_chains if joint in c), None)
        theta_min[chain_idx] = min_offset
        theta_max[chain_idx] = max_offset
        for j in chain[1:-1]:
            q_sphere[j] = q_finger[j]
            q_max_phi[j] = q_phi[j]
        c+=1

    phi_roots = np.zeros_like(anchor_values)
    phi_rs = np.zeros_like(anchor_values)
    c = 0 
    T_base_to_sphere = tf.quaternion_matrix(joint_info["sphere_frame"][1])
    T_base_to_sphere[:3, 3] = joint_info["sphere_frame"][0]
    T_sphere_to_base = np.linalg.inv(T_base_to_sphere)
    for chain in finger_chains:
        chain_idx = chain_indices[c]
        joint = chain[1]
        if joint in main_joints:
            joint = chain[2]
        T_base_to_root = compute_transform_to_joint(joint, kin_chains, joint_info, q_0_dict, base_link=base_link)
        p_base_root = T_base_to_root[:3, 3]
        p_sphere_root = (T_sphere_to_base @ np.append(p_base_root, 1))[:3]
        r_root = np.linalg.norm(p_sphere_root)
        joint_phi = np.arccos(p_sphere_root[2] / r_root)
        phi_roots[chain_idx] = joint_phi
        phi_rs[chain_idx] = r_root
        c+=1
        print(joint)
        print(p_base_root)
        print(p_sphere_root)

    print()
    print("Anchor Ranges:")
    print("Min:", theta_min)
    print("Max:", theta_max)
    print()


    print("Max Phi joint values:", q_max_phi)
    if verbosity:
        visualize_meshes_in_base_frame(meshes_dict, joint_info, kin_chains, q_sphere, "Sphere control", base_link)
        visualize_meshes_in_base_frame(meshes_dict, joint_info, kin_chains, q_max_phi, "Positive Theta Rot", base_link)
    

    print()

    # Create Finger Print
    radius = joint_info["sphere_frame"][6]  
    finger_print_dict = spherical_raycasts(
        meshes_dict,
        joint_info,
        kin_chains,
        q_sphere,
        control_sphere_samples,
        base_link
    )

    ft_dict = {}
    for key in finger_print_dict.keys():
        if finger_print_dict[key]:
            ft_dict[key] = finger_print_dict[key]["points"]

    if verbosity:
        visualize_vertices_in_base_frame(ft_dict, joint_info, kin_chains, q_sphere, f"Computed Robot Finger Print", base_link)
    print()

    p = sample_fibonacci_points(2000)
    theta, phi = p[:, 0], p[:, 1]
    radius = joint_info["sphere_frame"][6]  # Assuming radius is the 7th element
    phi_breaks = [0, np.pi/2, 3 * np.pi / 4, np.pi]
    radius_values = [0, 0, np.sqrt(2), 2]
    sphere_points = compute_axisymmetric_deformed_points(phi_breaks, radius_values, p, base_radius=radius)
    
    # Generate sphere x-axis points in sphere frame
    radius_offset = np.linspace(1, 1.5, 10) * radius  # 5 samples for radius scaling
    phi_plane = np.linspace(0, np.pi, 20)  # 20 samples for polar angle
    phi_plane, radius_offset = np.meshgrid(phi_plane, radius_offset)
    x = radius_offset * np.sin(phi_plane)
    z = radius_offset * np.cos(phi_plane)
    axis_points = np.stack((x.flatten(), np.zeros(z.flatten().shape), z.flatten()), axis=1)
    
    # Corresponding theta for axis_points (all zeros since they lie in x-z plane)
    theta_axis = np.zeros_like(phi_plane.flatten())
    
    # Center point
    center_point = np.array([[0, 0, 0]])
    theta_center = np.array([0])  # Assign theta=0 for center
    phi_center = np.array([0])    # Assign phi=0 for center

    # Combine all points, theta, and phi
    all_points = np.vstack((center_point, sphere_points))
    all_theta = np.concatenate((theta_center, theta))
    all_phi = np.concatenate((phi_center, phi))
    
    # Create finger_print_dict
    viz_finger_print_dict =finger_print_dict.copy()
    viz_finger_print_dict["sphere_frame"] = {
            "points": all_points,
            "theta": all_theta,
            "phi": all_phi
        }
    
    # Example usage of the lookup function
    print("Testing lookups")
    for j in main_joints: #
        if j is None:
            continue
        try:
            # print(joint_type_info[j])
            for i in np.arange(-1, 1, 0.5):
                q_A = get_q_A_from_offset(i, j, joint_type_info)
                print(f"q_A for theta_offset {i} on joint {j}: {q_A}")
        except ValueError as e:
            print(f"Lookup error: {e}")

    print()
    print()

    # Sphere samples
    n_spheres = 5

    # Initial theta of driving planes
    d_planes_theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) 
    
    # Define driving plane values with spheres as rows
    d_plane_offsets = np.array([
        [0, 0, 0, 0],  # Sphere 0: offsets for driving planes 0, 10, 15, 25
        [np.pi/4, 0.0, 0.0, 0.0],  # Sphere 1
        [np.pi/4, 0.0, 0.0, 0.0],  # Sphere 2
        [np.pi/6, -np.pi/8, -np.pi/4, np.pi/6],  # Sphere 4
        [0.0, np.pi/6, 0.0,  -np.pi/3]  # Sphere 3
    ]).T  # Shape: (n_spheres, n_driving_planes) = (5, 4)

    d_vectors_phi = np.array([np.pi/4, np.pi/2, 3*np.pi/4]) #constant phi

    # Define driving vector values with spheres as the first dimension
    d_vector_offsets = np.array([ # offset in r
        # Sphere 0
        [
            [0.0, 0.0, 0.0],  # Driving plane 0, vectors 3, 7, 11
            [0.0, 0.0, 0.0],  # Driving plane 10
            [0.0, 0.0, 0.0],  # Driving plane 15
            [0.0, 0.0, 0.0]  # Driving plane 25
        ],
        # Sphere 1
        [
            [0.0, 0.0, 0.0],  # Driving plane 0, vectors 3, 7, 11
            [0.0, 0.0, 0.0],  # Driving plane 10
            [0.0, 0.0, 0.0],  # Driving plane 15
            [0.0, 0.0, 0.0]  # Driving plane 25
        ],
        # Sphere 2
        [
            [-0.25,  0.4, 0.55],
            [0,  0.0, 0],
            [0,  0.0, 0.0],
            [0,  0.0, 0.0]
        ],
        # Sphere 3
        [
            [-0.25,  0.4, 0.55],
            [0.25,  0.3, 0.45],
            [0.15,  0.0, -0.3],
            [0.2, -0.3, -0.1]
        ],
        # Sphere 4
        [
            [-0.3,  0.0, 0.0],
            [-0.25,  1.25, -0.25],
            [-0.15,  -0.25, -0.25],
            [0.7, -0.7, -0.7]
        ],
        
    ])  # Shape: (n_spheres, n_driving_planes, n_driving_vectors) = (5, 4, 3)

    # Transpose to match expected shape (n_driving_planes, n_driving_vectors, n_spheres)
    d_vector_offsets = np.transpose(d_vector_offsets, (1, 2, 0))  # Shape: (4, 3, 5)

    # Poles values input
    s_pole_values = np.array([0.0, 0.0, 0.0, 0.25, -0.75])
    n_pole_values = np.zeros(len(s_pole_values))  # Shape: (n_spheres,), all 0.0
    n_pole_values[3]=0.25
    # Compute sphere surface points
    start_time = time.perf_counter()

    points = sample_fibonacci_points(10000)

    points, phi, theta = compute_deformed_sphere_points(
        d_planes_theta,      # 1D array of driving plane angles in radians
        d_plane_offsets,     # 2D array of theta offsets (n_driving_planes, n_spheres)
        d_vectors_phi,       # 1D array of driving vector phi angles in radians
        d_vector_offsets,    # 3D array of radius offsets (n_driving_planes, n_driving_vectors, n_spheres)
        n_pole_values,       # 1D array of north pole radius offsets for each sphere
        s_pole_values,       # 1D array of south pole radius offsets for each sphere
        points,       # Number of points to sample per sphere (default: 1000)
        base_radius=joint_info["sphere_frame"][6]      # Base radius of the sphere (default: 1.0)
    )

    centroids = np.mean(points, axis=0) 
    centroids_expanded = np.expand_dims(centroids, axis=0)
    points = np.concatenate([points, centroids_expanded], axis=0)
    points_h = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=2)

    # Record the end time
    end_time = time.perf_counter()
    
    # Calculate the duration
    duration = end_time - start_time

    print(f"Vector interpolation time: {duration} seconds")

    # Set up as gripper information 
    # Get all finger chains
    finger_chains = [ # Finger chains 
        chain for chain in kin_chains
        if "palm_normal" not in chain and "sphere_frame" not in chain
    ]

    non_finger_chains = [ # Finger chains 
        chain for chain in kin_chains
        if "palm_normal" in chain or "sphere_frame" in chain
    ]

    joints = []
    for chain in finger_chains:
        for j in chain[1:-1]: # Exclude base_link and tips
            joints.append(j)
    print("Gripper joints: ", joints)

    joint_index_dict = {element: index for index, element in enumerate(joints)}
    
    # Get init q_0 for all spheres
    q_0 = np.array([q_0_dict[j] for j in joints])
    print("q_0 position: ", q_0)
    q_0_all = np.tile(q_0, (n_spheres, 1))
    
    # Get finger chain indices
    finger_chain_indices = []
    for chain in finger_chains:
        indices = [joint_index_dict[x] for x in chain[1:-1]]
        finger_chain_indices.append(indices)
    print("finger chain indices: \n", finger_chain_indices)
    
    #Get type A information
    type_A_joints = [q for q in joint_type_info.keys() if joint_type_info[q]["type"]=="A"]
    v_dict = viz_finger_print_dict.copy()
    v_dict["sphere_frame"] = {
            "points": points[:-1, 0, :],
            "theta": theta,
            "phi": phi
        }

    q_dict = q_0_dict.copy()
    
    for joint in main_joints:
        q_0_dict[joint] = q_sphere[joint]

    # Solve inverse kinematics for all sphere samples at the same time
    img = 0
    if len(type_A_joints)>1:
        solve_for_A_joints(
            type_A_joints, 
            joint_type_info, 
            d_planes_theta, 
            d_plane_offsets, 
            joint_index_dict, 
            q_0_all
        )
        for joint in type_A_joints:
            print("Solved for:", joint)
            print("Result", q_0_all[:, joint_index_dict[joint]])

            q_dict[joint] = q_0_all[0, joint_index_dict[joint]]
        

    for chain in finger_chains:
        for i in range(len(chain)-2): # solve from root joint to finger tip (not including fingertip or baselink)
            # Record the start time
            start_time = time.perf_counter()
            
            j_idx = i+1
            joint = chain[j_idx]
            
            # If type A continue
            if (joint_type_info[joint]["type"] == "A") | (joint_type_info[joint]["type"] == "C"):
                continue
            elif (joint_type_info[joint]["type"] == "B") | (joint_type_info[joint]["type"] == "D"):
                # Solve for type B
                print("Solving for:", joint)
                T_sphere_to_joint = compute_sphere_to_single_joint_transforms(joint, q_0_all, joint_index_dict, joint_info, chain)
                T_joint_to_sphere = np.linalg.inv(T_sphere_to_joint)
                # print(T_joint_to_sphere)
                transformed_points_h = np.einsum('sij,psj->psi', T_joint_to_sphere, points_h)
                joint_points = transformed_points_h[:-1,:,:3]
                centroids = transformed_points_h[-1,:,:3]

                # Joint Phi
                # Extract the position of the joint origin in the sphere frame
                joint_pos_in_sphere = T_sphere_to_joint[:, :3, 3]  # Shape: (n_spheres, 3)
                r_joint = np.linalg.norm(joint_pos_in_sphere, axis=1)  # Shape: (n_spheres,)
                joint_phi = np.arccos(joint_pos_in_sphere[:, 2] / r_joint)  # Shape: (n_spheres,)

                # Compute point_mask for finger width filter
                # Extract parameters from joint_type_info
                box_min = np.array(joint_type_info[joint]["box_min"]) # [x_min, y_min, z_min] 
                box_max = np.array(joint_type_info[joint]["box_max"]) # [x_max, y_max, z_max] 

                ft = np.array(joint_type_info[joint]["nj"])               # Magnitude threshold 
                l = float(np.linalg.norm(ft[:2]))
                z_values = joint_points[..., 2]                                  # Shape: (n_points, n_spheres)
                
                
                z_values = joint_points[:, :, 2]
                # z_mask = (z_values >= box_min[2]) & (z_values <= box_max[2])
                # Add phi mask
                l = np.linalg.norm(joint_type_info[j]["nj"][:2])
                points_norm = np.linalg.norm(joint_points[:,:,:3], axis = 2)
                x_values = joint_points[:, :, 0] 
                if joint == chain[1]:
                    # range_mask = (mask & (x_values >= l))
                    range_mask = (points_norm >= l)
                elif j == chain[-2]:
                    range_mask = (points_norm >= l)
                else:
                    # range_mask = ((x_values > 0.0) & (x_values < l)) 
                    range_mask = (points_norm >= l) 
                

                
                z_mask = np.abs(z_values)<= 0.10 * radius

                # l = np.tile(l_ft, (n_spheres,))
                x_values = joint_points[:, :, 0] 
                x_mask = x_values > 0.0
                point_mask = z_mask & range_mask & x_mask
                
                q_0_all[:, joint_index_dict[joint]] = solve_for_B_joint(
                    q_0_all, 
                    joint, 
                    joint_index_dict, 
                    centroids,
                    joint_points,
                    joint_info,
                    point_mask,
                    l,
                    joint_type_info
                )
                print("Result", q_0_all[:, joint_index_dict[joint]])
                # q_dict = dict(zip(joints, q_0_all[0]))
                # vertices_dict["sphere_frame"] = points[:,0,:]
                # visualize_vertices_in_base_frame(vertices_dict, joint_info, kin_chains, q_dict, f"Joint_check {joint}")
            
            q_dict[joint] = q_0_all[0, joint_index_dict[joint]]
            
            # Record the end time
            end_time = time.perf_counter()
            
            # Calculate the duration
            duration = end_time - start_time

            print(f"Joint {joint} solving time: {duration} seconds")
    
    q_dict = dict(zip(joints, q_0_all[0]))

    if verbosity:
        visualize_vertices_in_base_frame(vertices_dict, joint_info, kin_chains, q_0_dict, "Initial Opened palm q", base_link)
        print("final_qs", q_0_all)
        for i in range(n_spheres):
            q_dict = dict(zip(joints, q_0_all[i]))
            vertices_dict["sphere_frame"] = points[:,i,:]
            visualize_vertices_in_base_frame(vertices_dict, joint_info, kin_chains, q_dict, f"Sphere {i}", base_link)
            visualize_meshes_and_fingerprints(meshes_dict, joint_info, kin_chains, finger_print_dict, q=q_dict, title="Sphere Sample", base_link=base_link)
    
    z = np.linspace(0, 10, 200) * 0.01  
    ft_points = np.stack((np.zeros(z.flatten().shape), np.zeros(z.flatten().shape), z.flatten()), axis=1)
    y = np.linspace(0, 10, 200) * 0.01  
    ft_points2 = np.stack((np.zeros(y.flatten().shape), y.flatten(), np.zeros(y.flatten().shape)), axis=1)
    x = np.linspace(0, 10, 200) * 0.01  
    ft_points3 = np.stack((x.flatten(), np.zeros(x.flatten().shape), np.zeros(x.flatten().shape)), axis=1)
    sphere_points = np.vstack((ft_points, ft_points2, ft_points3))
    vertices_dict["sphere_frame"] = sphere_points

    viz_dofs = dict()
    viz_dofs["joint_names"] = joints
    viz_dofs["values"] = q_0_all.copy()

    # Save everyting
    cik_json_path = os.path.join(robot_folder, "sphere_cik.json")

    # --- Save sphere_cik ---
    cik_structured = {
        "joint_info": joint_info,
        "kin_chains": kin_chains,
        "sphere_n_samples": control_sphere_samples,
        "chain_order": finger_chain_controller_indices,
        "chain_indices": chain_indices,
        "joint_type_info": joint_type_info,
        "q_open_palm": q_0_dict,
        "q_sphere": q_sphere,
        "plane_anchors": anchor_values, 
        "min_offsets": theta_min,
        "max_offsets": theta_max,
        "finger_print_dict": {
            k: {
                "points": v["points"],
                "theta": v["theta"],
                "phi": v["phi"],
                "indices": v["indices"],
                # "points_in_sphere": v["points_in_sphere"]
            } for k, v in finger_print_dict.items() if v
        },
        "fp_sample_7": {
            k: v for k, v in fp_7_point_sample.items()
        },
        "fp_projected": {
            k: v for k, v in projected_fp_dict.items()
        },
        "sphere_control": {
            k: v for k, v in sphere_control_dict.items()
        },
    }
    save_to_json(cik_json_path, cik_structured) 
    print(f"Saved sphere_cik data at {cik_json_path}")
