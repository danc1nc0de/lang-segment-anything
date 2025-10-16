import json
import os

DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes'
VERSION = 'v1.0-mini'


def main():
    data_tot = []
    json_num = 8
    for idx in range(json_num):
        json_path = os.path.join(DATA_ROOT, VERSION, VERSION, 'wheel_annotation_' + str(idx) + '.json')
        with open(json_path, 'r') as f:
            data_iter = json.load(f)
            data_tot.extend(data_iter)
    json_path = os.path.join(DATA_ROOT, VERSION, VERSION, 'wheel_annotation.json')
    with open(json_path, 'w') as f:
        json.dump(data_tot, f, indent=2)


if __name__ == '__main__':
    main()
