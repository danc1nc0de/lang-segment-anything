import os
from nuscenes.nuscenes import NuScenes
import json
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from lang_sam.utils import draw_image, load_image
import numpy as np
import cv2
from PIL import Image


VERSION = 'v1.0-mini'
DATA_ROOT = os.path.join('/home/danc1nc0de/Datasets/nuScenes', VERSION)
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_wheel_annos():
    json_path = os.path.join(DATA_ROOT, VERSION, 'wheel_annotation.json')
    wheel_annos_dict = {}
    with open(json_path, 'r') as f:
        wheel_annos = json.load(f)
        for wheel_anno in wheel_annos:
            sample_annotation_token = wheel_anno['sample_annotation_token']
            sensor_name = wheel_anno['sensor_name']
            if sensor_name not in wheel_annos_dict:
                wheel_annos_dict[sensor_name] = {}
            if sample_annotation_token not in wheel_annos_dict[sensor_name]:
                wheel_annos_dict[sensor_name][sample_annotation_token] = []
            wheel_annos_dict[sensor_name][sample_annotation_token].append(wheel_anno)
    return wheel_annos_dict


def get_boxes_2d(boxes_3d, camera_intrinsic, img_size):
    height, width = img_size
    boxes_2d = []
    for box in boxes_3d:
        corners_3d = box.corners()
        corners_img = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
        u_min, u_max = corners_img[0].min(), corners_img[0].max()
        v_min, v_max = corners_img[1].min(), corners_img[1].max()
        u_min, u_max = np.clip(u_min, 0, width), np.clip(u_max, 0, width)
        v_min, v_max = np.clip(v_min, 0, height), np.clip(v_max, 0, height)
        boxes_2d.append(np.array([u_min, v_min, u_max, v_max]))  # xyxy
    return boxes_2d


def vis_wheels(wheel_annos_dict, nusc):
    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
            for sensor_name in CAM_SENSORS:
                # img
                cam_data = nusc.get('sample_data', sample['data'][sensor_name])
                img_size = (cam_data['height'], cam_data['width'])

                # boxes_2d
                _, boxes_3d, camera_intrinsic = nusc.get_sample_data(sample['data'][sensor_name],
                                                                     box_vis_level=BoxVisibility.ANY)
                boxes_2d = get_boxes_2d(boxes_3d, camera_intrinsic, img_size)
                save_result_img(cam_data, wheel_annos_dict, boxes_2d, boxes_3d, sensor_name)


def draw_boxes(img, boxes_2d):
    img_draw = img.copy()
    for box_2d in boxes_2d:
        cv2.rectangle(img_draw, (box_2d[0].astype(np.int32), box_2d[1].astype(np.int32)),
                      (box_2d[2].astype(np.int32), box_2d[3].astype(np.int32)), (0, 255, 0), 2)
    return img_draw


def save_result_img(img_data, wheel_annos_dict, boxes_2d, boxes_3d, sensor_name):
    img_size = (img_data['height'], img_data['width'])
    img_path = os.path.join(DATA_ROOT, img_data['filename'])
    image_pil = load_image(img_path)

    output_img = image_pil.copy()
    image_array = np.asarray(image_pil)

    vehicle_boxes_2d = []
    wheel_scores = []
    wheel_boxes = []
    wheel_masks = []
    wheel_labels = []
    for box_2d, box_3d in zip(boxes_2d, boxes_3d):
        if box_3d.name.split('.')[0] == 'vehicle':
            vehicle_boxes_2d.append(box_2d)
            token = box_3d.token
            if token in wheel_annos_dict[sensor_name]:
                for wheel in wheel_annos_dict[sensor_name][token]:
                    wheel_boxes.append(wheel['box'])
                    wheel_scores.append(wheel['box_score'])
                    wheel_labels.append('wheel')
                    wheel_mask = np.zeros(img_size)
                    for x, y in wheel['mask']:
                        wheel_mask[x, y] = 1
                    wheel_masks.append(wheel_mask)

    image_array = draw_boxes(
        image_array,
        vehicle_boxes_2d
    )
    if len(wheel_boxes):
        wheel_boxes = np.array(wheel_boxes)
        wheel_masks = np.array(wheel_masks)
        wheel_scores = np.array(wheel_scores)
        image_array = draw_image(
            image_array,
            wheel_masks,
            wheel_boxes,
            wheel_scores,
            wheel_labels,
        )
    output_img = Image.fromarray(np.uint8(image_array)).convert("RGB")
    output_path = os.path.join(DATA_ROOT, 'wheels', sensor_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_name = img_data['filename'].split('/')[-1].split('.')[0]
    output_path = os.path.join(output_path, img_name + '_wheels.jpg')
    output_img.save(output_path)


def main():
    wheel_annos_dict = load_wheel_annos()
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)
    vis_wheels(wheel_annos_dict, nusc)


if __name__ == '__main__':
    main()
