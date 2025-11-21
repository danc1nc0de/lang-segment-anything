from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility, view_points

VERSION = 'v1.0-mini'
DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes/'
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_json_wheel_direction(cam_sensor):
    wheel_result = {}
    json_wheel_path = os.path.join(DATA_ROOT, VERSION, 'json_wheel_direction', cam_sensor,
                                   'sample_wheel_direction.json')
    with open(json_wheel_path, 'r') as f:
        wheel_result = json.load(f)
    return wheel_result


def get_boxes_3d_ego(nusc, sample, sensor_name):
    sample_data_token = sample['data'][sensor_name]
    boxes_3d_world = nusc.get_boxes(sample_data_token)  # boxes in world coordinate
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    boxes_3d_ego = []
    for box_3d_world in boxes_3d_world:
        box_3d_ego = box_3d_world.copy()
        box_3d_ego.translate(-np.array(ego_pose['translation']))
        box_3d_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        boxes_3d_ego.append(box_3d_ego)
    return boxes_3d_ego


def filtering_non_vehicles(boxes_3d):
    boxes_3d_output = []
    for box_3d in boxes_3d:
        if box_3d.name.split('.')[0] == 'vehicle':
            boxes_3d_output.append(box_3d)
    return boxes_3d_output


def get_img_wheel_path(img_name, cam_sensor):
    img_wheel_dir = os.path.join(DATA_ROOT, 'samples_wheel_calc_yaw', cam_sensor)
    if not os.path.exists(img_wheel_dir):
        os.makedirs(img_wheel_dir)
    img_wheel_name = img_name + '_wheel.jpg'
    img_wheel_name_notxt = img_name + '_wheel_notxt.jpg'
    img_wheel_path = os.path.join(img_wheel_dir, img_wheel_name)
    img_wheel_path_notxt = os.path.join(img_wheel_dir, img_wheel_name_notxt)
    return img_wheel_path, img_wheel_path_notxt


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    yaw_diff_dict = {}
    yaw_diff_lst_tot = []
    for cam_sensor in tqdm(CAM_SENSORS, desc="cam_sensor"):
        annos_wheel_direction = load_json_wheel_direction(cam_sensor)
        yaw_diff_dict[cam_sensor] = []
        for scene in tqdm(nusc.scene, desc="scene"):
            sample_token_lst = []
            first_sample_token = scene['first_sample_token']
            nxt_sample_token = first_sample_token
            while nxt_sample_token != '':
                sample_token_lst.append(nxt_sample_token)
                sample = nusc.get('sample', nxt_sample_token)
                nxt_sample_token = sample['next']
            for sample_token in tqdm(sample_token_lst, desc="sample_token"):
                sample = nusc.get('sample', sample_token)
                cam_data = nusc.get('sample_data', sample['data'][cam_sensor])
                img_name = cam_data['filename'].split('/')[-1].split('.')[0]
                if img_name not in annos_wheel_direction:
                    continue
                if len(annos_wheel_direction[img_name]) == 0:
                    continue
                # get boxes_3d
                _, boxes_3d_sensor_tot, _ = nusc.get_sample_data(sample['data'][cam_sensor],
                                                                 box_vis_level=BoxVisibility.ANY)
                boxes_3d_sensor_veh = filtering_non_vehicles(boxes_3d_sensor_tot)
                for box_3d_sensor in boxes_3d_sensor_veh:
                    if box_3d_sensor.token not in annos_wheel_direction[img_name]:
                        continue
                    box_sensor_yaw, _, _ = R.from_matrix(box_3d_sensor.rotation_matrix).as_euler('zyx', degrees=False)
                    wheel_direction_sensor = annos_wheel_direction[img_name][box_3d_sensor.token]['wheel_direction']
                    wheel_yaw = np.arctan2(wheel_direction_sensor[2], wheel_direction_sensor[0])
                    yaw_ego_diff = wheel_yaw - box_sensor_yaw
                    while yaw_ego_diff > np.pi:
                        yaw_ego_diff -= 2 * np.pi
                    while yaw_ego_diff < -np.pi:
                        yaw_ego_diff += 2 * np.pi
                    if abs(np.rad2deg(yaw_ego_diff)) > 15:
                        continue
                    yaw_diff_dict[cam_sensor].append(np.rad2deg(yaw_ego_diff))
                    yaw_diff_lst_tot.append(np.rad2deg(yaw_ego_diff))
                    # if 10 < abs(np.rad2deg(yaw_ego_diff)) < 15:
                    #     img_wheel_path, img_wheel_path_notxt = get_img_wheel_path(img_name, cam_sensor)
                    #     os.system(f"cp {img_wheel_path} {img_wheel_path_notxt} /home/danc1nc0de/Pictures/1015/")

    plt.figure(figsize=(8, 5))
    plt.hist(yaw_diff_lst_tot, bins=100, edgecolor='black')  # bins=10 表示分10个区间
    xticks = plt.xticks()[0]
    plt.xticks(xticks, [f"{d}°" for d in xticks])
    plt.xlabel("Difference in yaw angle")
    plt.ylabel("Counts")
    plt.title(VERSION)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
