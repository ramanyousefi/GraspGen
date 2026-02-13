#!/usr/bin/env python3

import argparse
import os
import struct
import sys
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import actionlib
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal

def read_ply_header(filepath):
    vertex_count = 0
    face_count = 0
    properties = []
    is_binary = False
    byte_order = '<'

    with open(filepath, 'rb') as f:
        line = f.readline().decode('utf-8').strip()
        if line != 'ply':
            raise RuntimeError("Not a PLY file")

        while True:
            line = f.readline().decode('utf-8').strip()

            if line == 'end_header':
                break

            if line.startswith('format'):
                if 'binary_little_endian' in line:
                    is_binary = True
                    byte_order = '<'
                elif 'binary_big_endian' in line:
                    is_binary = True
                    byte_order = '>'

            elif line.startswith('element'):
                parts = line.split()
                if parts[1] == 'vertex':
                    vertex_count = int(parts[2])
                    properties = []
                elif parts[1] == 'face':
                    face_count = int(parts[2])

            elif line.startswith('property') and vertex_count > 0:
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[2], parts[1]))

    return vertex_count, face_count, properties, is_binary, byte_order


def read_ply_vertices(filepath, vertex_count, properties, byte_order):
    type_map = {
        'float': ('f',4),
        'float32': ('f',4),
        'double': ('d',8),
        'int': ('i',4),
        'uint': ('I',4),
    }

    fmt = byte_order + ''.join(type_map[t][0] for _, t in properties)
    size = sum(type_map[t][1] for _, t in properties)

    xyz_idx = {name: i for i, (name, _) in enumerate(properties) if name in ['x', 'y', 'z']}
    if len(xyz_idx) != 3:
        raise RuntimeError("PLY missing x/y/z")

    vertices = []

    with open(filepath, 'rb') as f:
        while f.readline().decode('utf-8').strip() != "end_header":
            pass

        for _ in range(vertex_count):
            data = f.read(size)
            vals = struct.unpack(fmt, data)
            vertices.append([
                vals[xyz_idx["x"]],
                vals[xyz_idx["y"]],
                vals[xyz_idx["z"]]
            ])

    return np.asarray(vertices, dtype=np.float32)


def read_ply_faces(filepath, vertex_count, face_count, byte_order):
    faces = []

    with open(filepath, "rb") as f:
        while f.readline().decode('utf-8').strip() != "end_header":
            pass

        # Skip vertex data (x,y,z = 3 floats = 12 bytes)
        f.seek(vertex_count * 12, os.SEEK_CUR)

        for _ in range(face_count):
            n = struct.unpack(byte_order + "B", f.read(1))[0]
            idx = struct.unpack(byte_order + "I" * n, f.read(4 * n))
            if n == 3:
                faces.append(idx)

    return np.asarray(faces, dtype=np.int32)


def sample_mesh_surface(vertices, faces, num_samples=12000):
    v0 =vertices[faces[:, 0]]
    v1 =vertices[faces[:, 1]]
    v2 =vertices[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / np.sum(areas)

    chosen = np.random.choice(len(faces), size=num_samples, p=probs)

    r1 = np.sqrt(np.random.rand(num_samples))
    r2 = np.random.rand(num_samples)

    a =1-r1
    b =r1*(1-r2)
    c =r1*r2

    samples = (a[:, None] * v0[chosen] + b[:, None] * v1[chosen] + c[:, None] * v2[chosen])

    return np.vstack([vertices, samples])


def rotate_points(points):#rotation needed bcs then it takes the slide as table
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    return points @ R.T


def lift(points):#lifting the slide so its above the thing
    up = 0.02
    minZ = points[:, 2].min()
    points[:, 2] += (up - minZ)
    return points


def thicken_object(points): #also thicken bcs lib ignores object if too thin
    thickness = 0.003
    offsets = np.linspace(-thickness / 2, thickness / 2, 3)
    return np.vstack([points + [0, 0, o] for o in offsets])


def create_pointcloud2_msg(points, frame_id):
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    rgb = np.uint32((255 << 16) | (255 << 8) | 255)
    cloud = [[x, y, z, rgb] for x, y, z in points]

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.UINT32, 1),
    ]

    return pc2.create_cloud(header, fields, cloud)


def plan_grasps(points, frame_id):
    pub = rospy.Publisher('/head_camera/depth_registered/points', PointCloud2, queue_size=1)

    client = actionlib.SimpleActionClient(
        'basic_grasping_perception/find_objects',
        FindGraspableObjectsAction)

    #rospy.loginfo("Waiting for grasp server...")
    client.wait_for_server()

    cloud = create_pointcloud2_msg(points, frame_id)

    goal = FindGraspableObjectsGoal()
    goal.plan_grasps = True
    client.send_goal(goal)

    start = rospy.Time.now()
    while rospy.Time.now() - start < rospy.Duration(10):
        pub.publish(cloud)
        rospy.sleep(0.05)

    client.wait_for_result()
    result = client.get_result()

    if not result or not result.objects:
        rospy.logerr("couldnt find object")
        return False

    rospy.loginfo("Grasp planning SUCCESS")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_file")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    rospy.init_node('ply_to_ptcld')

    vc, fc, props, is_bin, endian = read_ply_header(args.ply_file)
    vertices = read_ply_vertices(args.ply_file, vc, props, endian)
    faces = read_ply_faces(args.ply_file, vc, fc, endian)

    rospy.loginfo(f"Loaded {len(vertices)} vertices, {len(faces)} faces")

    points = sample_mesh_surface(vertices, faces)
    points = rotate_points(points)
    points = thicken_object(points)
    points = lift(points)
    points *= args.scale

    rospy.loginfo(f"Publishing {len(points)} points")
    plan_grasps(points, args.frame_id)


if __name__ == '__main__':
    main()
