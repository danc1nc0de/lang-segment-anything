from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np

VERSION = 'v1.0-trainval'
DATA_ROOT = '/mnt/sunyinghao/datasets/nuscenes/'
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_json_wheel(cam_sensor):
    wheel_result = {}
    json_wheel_path = os.path.join(DATA_ROOT, VERSION, 'json_wheel_assoc_n_filtering_fix', cam_sensor,
                                   'sample_wheel_annotation.json')
    with open(json_wheel_path, 'r') as f:
        wheel_result = json.load(f)
    return wheel_result


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    stat_num_dict = {}
    for cam_sensor in tqdm(CAM_SENSORS, desc="cam_sensor"):
        annos_wheel = load_json_wheel(cam_sensor)
        for img_name in annos_wheel:
            if 'assoc_box_tokens' not in annos_wheel[img_name]:
                continue
            for assoc_box_token in annos_wheel[img_name]['assoc_box_tokens']:
                anno_info = nusc.get(table_name='sample_annotation', token=assoc_box_token)
                # stat count
                if anno_info['category_name'] not in stat_num_dict:
                    stat_num_dict[anno_info['category_name']] = 0
                stat_num_dict[anno_info['category_name']] += 1
    print(stat_num_dict)


if __name__ == '__main__':
    main()
