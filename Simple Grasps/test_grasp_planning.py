#!/usr/bin/env python3
#FOR TESTING TO SEE IF EVERYTING WORKS, THE NODES CONNECT
import rospy
import actionlib
import numpy as np
import json
from datetime import datetime
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal


def create_synthetic_point_cloud(object_type, frame_id='base_link'):
    points = []
    table_size = 1.0
    table_resolution = 0.02
    for x in np.arange(-table_size/2, table_size/2, table_resolution):
        for y in np.arange(-table_size/2, table_size/2, table_resolution):
            points.append([x, y, 0.0])
    
    if object_type == 'cylinder':
        radius = 0.03
        height = 0.08
        center = np.array([0.5, 0.0, 0.0])
        
        # Generate points on cylinder surface
        num_points = 150
        angles = np.linspace(0, 2*np.pi, num_points)
        heights = np.linspace(center[2], center[2] + height, 15)
        
        for h in heights:
            for angle in angles:
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                points.append([x, y, h])
    
    elif object_type == 'sphere':
        radius = 0.04
        center = np.array([0.5, 0.0, radius])
        
        num_points = 250
        for i in range(num_points):
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            
            if z >= 0:
                points.append([x, y, z])
    
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    
    points_with_color = []
    for p in points:
        rgb = 255 << 16 | 255 << 8 | 255  # White: (R=255, G=255, B=255)
        points_with_color.append([p[0], p[1], p[2], rgb])
    
    fields = [
        pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
        pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1),
    ]
    
    cloud_msg = pc2.create_cloud(header, fields, points_with_color)
    return cloud_msg


def publish_grasp_markers(grasps, frame_id='base_link', marker_pub=None):
    if marker_pub is None:
        marker_pub = rospy.Publisher('/grasp_markers', MarkerArray, queue_size=10)
    
    marker_array = MarkerArray()
    
    for i, grasp in enumerate(grasps):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        pos = grasp.grasp_pose.pose.position
        marker.pose.position = pos
        marker.pose.orientation = grasp.grasp_pose.pose.orientation
        
        marker.scale.x = 0.02  #length
        marker.scale.y = 0.005  #width
        marker.scale.z = 0.005
        
        quality = grasp.grasp_quality
        marker.color.r = 1.0 - quality  # Red if low quality
        marker.color.g = quality        # Green if high quality
        marker.color.b = 0.5
        marker.color.a = 0.8
        
        marker_array.markers.append(marker)
    
    #rospy.loginfo(f"Publishing {len(grasps)} grasp markers...")
    marker_pub.publish(marker_array)


# def save_grasp_results(object_type, result, output_dir='/tmp'):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{output_dir}/grasp_results_{object_type}_{timestamp}.json"
    
#     data = {
#         'timestamp': timestamp,
#         'object_type': object_type,
#         'num_objects': len(result.objects),
#         'objects': []
#     }
    
#     for obj_idx, obj in enumerate(result.objects):
#         obj_data = {
#             'object_id': obj_idx,
#             'num_grasps': len(obj.grasps),
#             'grasps': []
#         }
        
#         for grasp_idx, grasp in enumerate(obj.grasps):
#             grasp_data = {
#                 'grasp_id': grasp_idx,
#                 'quality': float(grasp.grasp_quality),
#                 'position': {
#                     'x': float(grasp.grasp_pose.pose.position.x),
#                     'y': float(grasp.grasp_pose.pose.position.y),
#                     'z': float(grasp.grasp_pose.pose.position.z)
#                 },
#                 'orientation': {
#                     'x': float(grasp.grasp_pose.pose.orientation.x),
#                     'y': float(grasp.grasp_pose.pose.orientation.y),
#                     'z': float(grasp.grasp_pose.pose.orientation.z),
#                     'w': float(grasp.grasp_pose.pose.orientation.w)
#                 }
#             }
#             obj_data['grasps'].append(grasp_data)
        
#         data['objects'].append(obj_data)
    
#     try:
#         with open(filename, 'w') as f:
#             json.dump(data, f, indent=2)
#         rospy.loginfo(f"Grasp results saved to: {filename}")
#         return filename
#     except Exception as e:
#         rospy.logerr(f"Failed to save grasp results: {e}")
#         return None


def test_grasp_planning(object_type):
    rospy.loginfo(f"Testing grasp planning with synthetic {object_type}...")
    
    # put the comments on rviz topics first one at pointcloud2 2nd one in marker arrays
    cloud_pub = rospy.Publisher("/head_camera/depth_registered/points", PointCloud2, queue_size=1)
    marker_pub = rospy.Publisher("/grasp_markers", MarkerArray, queue_size=10)
    
    rospy.sleep(0.5)
    
    rospy.loginfo("Connecting to basic_grasping_perception/find_objects action server...")
    find_objects_client = actionlib.SimpleActionClient(
        "basic_grasping_perception/find_objects",
        FindGraspableObjectsAction
    )
    
    if not find_objects_client.wait_for_server(timeout=rospy.Duration(5.0)):
        rospy.logerr("FAIL")
        return False
    
    #rospy.loginfo("Connected to server")
    
    #send goal
    goal = FindGraspableObjectsGoal()
    goal.plan_grasps = True
    
    #rospy.loginfo("Sending goal to find_objects action server")
    find_objects_client.send_goal(goal)
    
    #rospy.loginfo(f"Publishing synthetic {object_type} point cloud...")
    start_time = rospy.Time.now()
    publish_duration = rospy.Duration(12.0)
    
    while rospy.Time.now() - start_time < publish_duration and not rospy.is_shutdown():
        cloud_msg = create_synthetic_point_cloud(object_type, frame_id="base_link")
        cloud_pub.publish(cloud_msg)
        rospy.sleep(0.05)  # 20hz
    
    #rospy.loginfo("Waiting for result from action server...")
    if not find_objects_client.wait_for_result(timeout=rospy.Duration(5.0)):
        rospy.logerr("timeout")
        return False
    
    result = find_objects_client.get_result()
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("RESULTS")
    rospy.loginfo("=" * 60)
    
    if result is None or len(result.objects) == 0:
        rospy.logwarn("WARNING: couldnt detect objects.")
        return False
        
    total_grasps = 0
    all_grasps = []
    for i, obj in enumerate(result.objects):
        rospy.loginfo(f"\nObject {i + 1}:")
        rospy.loginfo(f"  Grasps found: {len(obj.grasps)}")
        total_grasps += len(obj.grasps)
        all_grasps.extend(obj.grasps)
        
        if len(obj.grasps) > 0:
            rospy.loginfo(f"  Top 3 grasps")
            for j, grasp in enumerate(obj.grasps[:3]):
                pos = grasp.grasp_pose.pose.position
                rospy.loginfo(f"    Grasp {j + 1}:")
                rospy.loginfo(f"      Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
                rospy.loginfo(f"      Quality: {grasp.grasp_quality:.4f}")
    
    if total_grasps > 0:
        publish_grasp_markers(all_grasps, frame_id='base_link', marker_pub=marker_pub)
        
        rospy.loginfo("AAAAAAAAAA Grasps generated")
        return True
    else:
        rospy.logwarn("WARNING: No grasps generated")
        return False


def main():
    rospy.init_node('test_grasp_planning')
    
    try:
        results = {}
        for object_type in ['cylinder', 'sphere']:
            rospy.loginfo(f"\n{'=' * 60}")
            rospy.loginfo(f"Testing {object_type}...")
            rospy.loginfo(f"{'=' * 60}\n")
            
            success = test_grasp_planning(object_type)
            results[object_type] = success
            
            rospy.sleep(2.0)
        
        # Summary
        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("FINAL SUMMARY")
        rospy.loginfo("=" * 60)
        
        for obj_type, success in results.items():
            status = "--PASSED--" if success else "--FAILED--"
            rospy.loginfo(f"{obj_type}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            rospy.loginfo("\nAll tests passed!!!")
        else:
            rospy.logwarn("\nSome tests failed.")
        
        return all_passed
    
    except KeyboardInterrupt:
        rospy.loginfo("Test interrupted by user")
        return False
    except Exception as e:
        rospy.logerr(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
