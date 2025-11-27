from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import cv2

CAM_SENSORS = ['CAM_FRONT_ZOOMED', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
               'CAM_BACK_RIGHT']


def load_orig_img(img_name):
    img_dir = '/home/danc1nc0de/Datasets/Lyft/images'
    img_path = os.path.join(img_dir, img_name + '.jpeg')
    img = Image.open(img_path).convert("RGB")
    return img


def draw_boxes_3d(img, boxes_3d, calibrated_sensor_data, colors_map):
    img_output = np.asarray(img).copy()
    cam_intrinsic = np.array(calibrated_sensor_data['camera_intrinsic'])
    for idx, box in enumerate(boxes_3d):
        color = colors_map[idx]
        box_corners = view_points(box.corners(), cam_intrinsic, normalize=True)[:2, :]
        for i in range(4):
            cv2.line(img_output,
                     (int(box_corners.T[i][0]), int(box_corners.T[i][1])),
                     (int(box_corners.T[i + 4][0]), int(box_corners.T[i + 4][1])),
                     color, 2)
            cv2.line(img_output,
                     (int(box_corners.T[i][0]), int(box_corners.T[i][1])),
                     (int(box_corners.T[(i + 1) % 4][0]), int(box_corners.T[(i + 1) % 4][1])),
                     color, 2)
            cv2.line(img_output,
                     (int(box_corners.T[i + 4][0]), int(box_corners.T[i + 4][1])),
                     (int(box_corners.T[(i + 1) % 4 + 4][0]), int(box_corners.T[(i + 1) % 4 + 4][1])),
                     color, 2)
    return Image.fromarray(np.uint8(img_output)).convert("RGB")


def get_colors_map(cnt):
    colors_map = []
    for i in range(cnt):
        colors_map.append(np.random.choice(range(256), size=3).tolist())
    return colors_map


def main():
    lyft = LyftDataset(data_path='/home/danc1nc0de/Datasets/Lyft/', json_path='/home/danc1nc0de/Datasets/Lyft/data',
                       verbose=True)
    for scene in tqdm(lyft.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = lyft.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
            for cam_sensor in CAM_SENSORS:
                cam_data = lyft.get('sample_data', sample['data'][cam_sensor])
                calibrated_sensor_data = lyft.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                _, boxes_3d, camera_intrinsic = lyft.get_sample_data(sample['data'][cam_sensor],
                                                                     box_vis_level=BoxVisibility.ANY)
                colors_map = get_colors_map(len(boxes_3d))
                img_name = cam_data['filename'].split('/')[-1].split('.')[0]
                img = load_orig_img(img_name)
                img = draw_boxes_3d(img, boxes_3d, calibrated_sensor_data, colors_map)
                img_dir = os.path.join('/home/danc1nc0de/Datasets/Lyft/images_output', cam_sensor)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                img_path = os.path.join(img_dir, img_name + '.jpg')
                img.save(img_path)


if __name__ == '__main__':
    main()
