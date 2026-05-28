import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from pygments.styles.dracula import background
from tqdm import tqdm
import numpy as np

VERSION = 'v1.0-mini'
DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes/'
RADAR_SENSORS = ['RADAR_FRONT']


def main():
    object_dict = {} # class : distance : [rcs0, rcs1, ...]
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']

            cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

            radar_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
            calibrated_sensor_radar = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            _, boxes_3d_rad, _ = nusc.get_sample_data(sample['data']['RADAR_FRONT'])

            ptc_path = os.path.join(DATA_ROOT, radar_data['filename'])
            ptc = RadarPointCloud.from_file(ptc_path)
            for pt_idx in range(ptc.points.shape[1]):
                x_rad, y_rad = ptc.points[:, pt_idx][0], ptc.points[:, pt_idx][1]
                vx_comp, vy_comp = ptc.points[:, pt_idx][8], ptc.points[:, pt_idx][9]
                rcs = ptc.points[:, pt_idx][5]
                dis_min = float('inf')
                box_match_idx = -1
                for box_idx, box_3d_rad in enumerate(boxes_3d_rad):
                    box_3d_rad_center = box_3d_rad.center
                    dis = np.sqrt((x_rad - box_3d_rad_center[0]) ** 2 + (y_rad - box_3d_rad_center[1]) ** 2)
                    if dis < dis_min:
                        dis_min = dis
                        box_match_idx = box_idx
                if box_match_idx < 0:
                    continue
                width, length = boxes_3d_rad[box_match_idx].wlh[0], boxes_3d_rad[box_match_idx].wlh[1]
                if dis_min > 0.5 * np.sqrt(width ** 2 + length ** 2):
                    continue
                pass


if __name__ == '__main__':
    main()
