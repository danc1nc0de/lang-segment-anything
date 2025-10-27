from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
import json

VERSION = 'v1.0-mini'
DATA_ROOT = os.path.join('/home/danc1nc0de/Datasets/nuScenes', VERSION)
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def update_box_wheel_direction(wheel_direction_ego, wheel_yaw, box_ego):
    wheel_yaw_valid = -1
    if np.abs(wheel_yaw - box_ego.orientation.angle) < np.radians(5.0):
        wheel_yaw_valid = 0
    elif np.abs(wheel_yaw + np.pi - box_ego.orientation.angle) < np.radians(5.0):
        wheel_yaw_valid = 1
    elif np.abs(wheel_yaw - np.pi - box_ego.orientation.angle) < np.radians(5.0):
        wheel_yaw_valid = 2
    else:
        wheel_yaw_valid = -1

    if wheel_yaw_valid >= 0:
        box_ego.wheel_direction_valid = True
        if box_ego.wheel_direction is None:
            box_ego.wheel_direction = []
        if box_ego.wheel_yaw is None:
            box_ego.wheel_yaw = []
        if wheel_yaw_valid == 0:
            box_ego.wheel_direction.append(wheel_direction_ego)
            box_ego.wheel_yaw.append(wheel_yaw)
        elif wheel_yaw_valid == 1:
            box_ego.wheel_direction.append(-wheel_direction_ego)
            box_ego.wheel_yaw.append(wheel_yaw + np.pi)
        elif wheel_yaw_valid == 2:
            box_ego.wheel_direction.append(-wheel_direction_ego)
            box_ego.wheel_yaw.append(wheel_yaw - np.pi)
        else:
            assert False, 'invalid wheel yaw'


def get_wheel_ground_point(wheel_anno):
    u_grd, v_grd = -1.0, -1.0
    for v, u in wheel_anno['mask']:
        if v > v_grd:
            u_grd, v_grd = u, v
    return u_grd, v_grd


def update_wheel_direction(boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data):
    # calc calibration
    R_sensor_to_ego = Quaternion(calibrated_sensor_data['rotation']).rotation_matrix
    K = np.array(calibrated_sensor_data['camera_intrinsic'])
    U = R_sensor_to_ego @ np.linalg.inv(K)
    u = U[2, :]
    for box_ego in boxes_ego:
        wheel_anno_lst = wheel_annos[sensor_name][box_ego.token]
        for i in range(len(wheel_anno_lst)):
            wheel_anno_0 = wheel_anno_lst[i]
            u_grd_0, v_grd_0 = get_wheel_ground_point(wheel_anno_0)
            P_i_0 = np.array([u_grd_0, v_grd_0, 1.0])
            for j in range(i + 1, len(wheel_anno_lst)):
                wheel_anno_1 = wheel_anno_lst[j]
                u_grd_1, v_grd_1 = get_wheel_ground_point(wheel_anno_1)
                P_i_1 = np.array([u_grd_1, v_grd_1, 1.0])
                wheel_direction_ego = U @ ((u.T @ P_i_0) * P_i_1 - (u.T @ P_i_1) * P_i_0)
                wheel_yaw = np.arctan2(-wheel_direction_ego[1], wheel_direction_ego[0])
                wheel_yaw_deg = np.degrees(wheel_yaw)
                update_box_wheel_direction(wheel_direction_ego, wheel_yaw, box_ego)


def filtering_not_valid_wheels(wheel_annos_lst):
    wheel_annos_lst_output = []
    for wheel_annos in wheel_annos_lst:
        u_min, v_min, u_max, v_max = wheel_annos['box']
        width, height = u_max - u_min, v_max - v_min
        if height < width * 0.95:
            continue
        wheel_annos_lst_output.append(wheel_annos)
    return wheel_annos_lst_output


def filtering_boxes(boxes_ego, wheel_annos, sensor_name):
    boxes_ego_output = []
    for box_ego in boxes_ego:
        # filtering non-vehicle
        if box_ego.name.split('.')[0] != 'vehicle':
            continue
        # filtering no wheels
        if box_ego.token not in wheel_annos[sensor_name]:
            continue
        # filtering less than 2 wheels
        if len(wheel_annos[sensor_name][box_ego.token]) < 2:
            continue
        # filtering not valid wheels
        wheel_annos[sensor_name][box_ego.token] = filtering_not_valid_wheels(wheel_annos[sensor_name][box_ego.token])
        # filtering less than 2 wheels
        if len(wheel_annos[sensor_name][box_ego.token]) < 2:
            continue
        boxes_ego_output.append(box_ego)
    return boxes_ego_output


def get_boxes_ego(nusc, sample, sensor_name):
    sample_data_token = sample['data'][sensor_name]
    boxes_world = nusc.get_boxes(sample_data_token)  # boxes in world coordinate
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    boxes_ego = []
    for box_world in boxes_world:
        box_ego = box_world.copy()
        box_ego.translate(-np.array(ego_pose['translation']))
        box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        boxes_ego.append(box_ego)
    return boxes_ego


def load_wheel_annos():
    wheel_annos_json_path = os.path.join(DATA_ROOT, VERSION, 'wheel_annotation.json')
    wheel_annos = {}
    with open(wheel_annos_json_path, 'r') as f:
        wheel_annos_json = json.load(f)
        for wheel_anno in wheel_annos_json:
            sample_annotation_token = wheel_anno['sample_annotation_token']
            sensor_name = wheel_anno['sensor_name']
            if sensor_name not in wheel_annos:
                wheel_annos[sensor_name] = {}
            if sample_annotation_token not in wheel_annos[sensor_name]:
                wheel_annos[sensor_name][sample_annotation_token] = []
            wheel_annos[sensor_name][sample_annotation_token].append(wheel_anno)
    return wheel_annos


def main():
    wheel_annos = load_wheel_annos()
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
            for sensor_name in CAM_SENSORS:
                sample_sensor_data = nusc.get('sample_data', sample['data'][sensor_name])
                # calibration data of sensor to ego
                calibrated_sensor_data = nusc.get('calibrated_sensor', sample_sensor_data['calibrated_sensor_token'])
                boxes_ego = get_boxes_ego(nusc, sample, sensor_name)
                boxes_ego = filtering_boxes(boxes_ego, wheel_annos, sensor_name)
                update_wheel_direction(boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data)


if __name__ == '__main__':
    main()
