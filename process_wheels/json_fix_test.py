import os
import json
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

VERSION = ['v1.0-mini']
DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes/'
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def save_json_wheel(annos_wheel, json_wheel_path):
    with open(json_wheel_path, 'w') as f:
        json.dump(annos_wheel, f, indent=2)


def load_json_wheel(cam_sensor):
    wheel_result = {}
    json_wheel_path = os.path.join(DATA_ROOT, 'json_wheel', cam_sensor, 'sample_wheel_annotation.json')
    with open(json_wheel_path, 'r') as f:
        wheel_result = json.load(f)
    return wheel_result


def get_save_json_path(cam_sensor, version):
    json_wheel_dir = os.path.join(DATA_ROOT, version, 'json_wheel_assoc_n_filtering_fix', cam_sensor)
    if not os.path.exists(json_wheel_dir):
        os.makedirs(json_wheel_dir)
    json_wheel_path = os.path.join(json_wheel_dir, 'sample_wheel_annotation.json')
    return json_wheel_path, os.path.exists(json_wheel_path)


def main():
    for version in tqdm(VERSION, desc='version'):
        nusc = NuScenes(version=version, dataroot=DATA_ROOT, verbose=True)
        for cam_sensor in tqdm(CAM_SENSORS, desc="cam_sensor"):
            json_wheel_path, flag_path_exists = get_save_json_path(cam_sensor, version)
            if flag_path_exists:
                continue
            annos_wheel = load_json_wheel(cam_sensor)
            annos_wheel_output = {}
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
                    annos_wheel_output.update(
                        {img_name: annos_wheel[img_name]}
                    )
            save_json_wheel(annos_wheel_output, json_wheel_path)


if __name__ == '__main__':
    main()
