from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
import json

VERSION = 'v1.0-mini'
DATA_ROOT = os.path.join('/home/danc1nc0de/Datasets/nuScenes', VERSION)
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

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
                boxes_ego = get_boxes_ego(nusc, sample, sensor_name)
                boxes_ego = filtering_boxes(boxes_ego, wheel_annos, sensor_name)
                pass


if __name__ == '__main__':
    main()
