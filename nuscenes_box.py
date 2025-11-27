from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import cv2

VERSION = 'v1.0-mini'
DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes/'
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def load_orig_img(img_name, cam_sensor):
    img_dir = os.path.join(DATA_ROOT, 'samples', cam_sensor)
    img_path = os.path.join(img_dir, img_name + '.jpg')
    img = load_image(img_path)
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
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
            for cam_sensor in CAM_SENSORS:
                cam_data = nusc.get('sample_data', sample['data'][cam_sensor])
                calibrated_sensor_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                _, boxes_3d, camera_intrinsic = nusc.get_sample_data(sample['data'][cam_sensor],
                                                                     box_vis_level=BoxVisibility.ANY)
                colors_map = get_colors_map(len(boxes_3d))
                img_name = cam_data['filename'].split('/')[-1].split('.')[0]
                if img_name != 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151613362404':
                    continue
                img = load_orig_img(img_name, cam_sensor)
                img = draw_boxes_3d(img, boxes_3d, calibrated_sensor_data, colors_map)
                img_dir = os.path.join('/home/danc1nc0de/Datasets/nuScenes/samples_box', cam_sensor)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                img_path = os.path.join(img_dir, img_name + '.jpg')
                img.save(img_path)
                img_crop = np.asarray(img).copy()
                img_crop = img_crop[420:620, 730:1050]
                img_crop = Image.fromarray(np.uint8(img_crop)).convert("RGB")
                original_size = img_crop.size  # 获取原始尺寸 (width, height)
                new_size = (original_size[0] * 2, original_size[1] * 2)  # 计算新尺寸
                img_crop = img_crop.resize(new_size, Image.LANCZOS)
                img_crop = Image.fromarray(np.uint8(img_crop)).convert("RGB")
                img_crop.save(img_path.replace('.jpg', '_crop.jpg'))

                img = np.asarray(img).copy()
                img[0:400, 960:] = img_crop
                img = Image.fromarray(np.uint8(img)).convert("RGB")
                img.save(img_path)


if __name__ == '__main__':
    main()
