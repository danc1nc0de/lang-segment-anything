import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image, load_image
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import uuid

DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes/v1.0-mini'
TXT_PROMPT = 'wheel.'
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def seg_wheels(model, img_data):
    img_path = os.path.join(DATA_ROOT, img_data['filename'])
    image_pil = load_image(img_path)
    return model.predict([image_pil], [TXT_PROMPT])[0]


def draw_boxes(img, boxes_2d):
    img_draw = img.copy()
    for box_2d in boxes_2d:
        cv2.rectangle(img_draw, (box_2d[0].astype(np.int32), box_2d[1].astype(np.int32)), (box_2d[2].astype(np.int32), box_2d[3].astype(np.int32)), (0, 255, 0), 2)
    return img_draw


def save_result_img(img_data, result, boxes_2d, sensor_name):
    img_path = os.path.join(DATA_ROOT, img_data['filename'])
    image_pil = load_image(img_path)
    output_img = image_pil.copy()
    if result['labels']:
        image_array = np.asarray(image_pil)
        image_array = draw_boxes(
            image_array,
            boxes_2d
        )
        output_img = draw_image(
            image_array,
            result["masks"],
            result["boxes"],
            result["scores"],
            result["labels"],
        )
        output_img = Image.fromarray(np.uint8(output_img)).convert("RGB")
    output_path = os.path.join(DATA_ROOT, 'wheels', sensor_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_name = img_data['filename'].split('/')[-1].split('.')[0]
    output_path = os.path.join(output_path, img_name + '_wheels.jpg')
    output_img.save(output_path)


def del_small_wheels(wheels_result, min_area=400):
    if len(wheels_result['labels']) == 0:
        return wheels_result
    idx_keep = []
    for idx in range(len(wheels_result['labels'])):
        u_min, v_min, u_max, v_max = wheels_result['boxes'][idx]
        area = (u_max - u_min) * (v_max - v_min)
        if area > min_area:
            idx_keep.append(idx)
    if len(wheels_result['labels']) == len(idx_keep):
        return wheels_result
    if not idx_keep:
        wheels_result['scores'] = np.empty(0)
        wheels_result['boxes'] = np.empty((0, 4))
        wheels_result['text_labels'] = []
        wheels_result['labels'] = []
        wheels_result['masks'] = []
        wheels_result['mask_scores'] = []
        return wheels_result
    wheels_result['scores'] = wheels_result['scores'][idx_keep]
    wheels_result['boxes'] = wheels_result['boxes'][idx_keep]
    wheels_result['text_labels'] = [wheels_result['text_labels'][idx] for idx in idx_keep]
    wheels_result['labels'] = [wheels_result['labels'][idx] for idx in idx_keep]
    wheels_result['masks'] = wheels_result['masks'][idx_keep]
    wheels_result['mask_scores'] = wheels_result['mask_scores'][idx_keep]
    return wheels_result


def del_non_vehicle_wheels(wheels_result, boxes_2d, ratio=0.9):
    if len(wheels_result['labels']) == 0:
        return wheels_result
    idx_keep = []
    for idx in range(len(wheels_result['labels'])):
        u_min, v_min, u_max, v_max = wheels_result['boxes'][idx]
        area = (u_max - u_min) * (v_max - v_min)
        flag_in_box = False
        for box_2d in boxes_2d:
            u_min_inter = max(u_min, box_2d[0])
            u_max_inter = min(u_max, box_2d[2])
            v_min_inter = max(v_min, box_2d[1])
            v_max_inter = min(v_max, box_2d[3])
            if u_max_inter < u_min_inter or v_max_inter < v_min_inter:
                continue
            area_inter = (u_max_inter - u_min_inter) * (v_max_inter - v_min_inter)
            if area_inter > ratio * area:
                flag_in_box = True
                break
        if flag_in_box:
            idx_keep.append(idx)
    if len(wheels_result['labels']) == len(idx_keep):
        return wheels_result
    if not idx_keep:
        wheels_result['scores'] = np.empty(0)
        wheels_result['boxes'] = np.empty((0, 4))
        wheels_result['text_labels'] = []
        wheels_result['labels'] = []
        wheels_result['masks'] = []
        wheels_result['mask_scores'] = []
        return wheels_result
    wheels_result['scores'] = wheels_result['scores'][idx_keep]
    wheels_result['boxes'] = wheels_result['boxes'][idx_keep]
    wheels_result['text_labels'] = [wheels_result['text_labels'][idx] for idx in idx_keep]
    wheels_result['labels'] = [wheels_result['labels'][idx] for idx in idx_keep]
    wheels_result['masks'] = wheels_result['masks'][idx_keep]
    wheels_result['mask_scores'] = wheels_result['mask_scores'][idx_keep]
    return wheels_result


def update_association_info(wheels_result, boxes_2d, boxes_3d):
    return wheels_result


def post_process_result(wheels_result, boxes_3d, camera_intrinsic):
    boxes_3d = filtering_non_vehicles(boxes_3d) # only keep vehicles
    boxes_2d = get_boxes_2d(boxes_3d, camera_intrinsic) # get boxes_2d from boxes_3d
    # wheels_result = del_small_wheels(wheels_result)
    wheels_result = del_non_vehicle_wheels(wheels_result, boxes_2d)
    wheels_result = update_association_info(wheels_result, boxes_2d, boxes_3d)
    return wheels_result, boxes_3d, boxes_2d


def get_boxes_2d(boxes_3d, camera_intrinsic):
    boxes_2d = []
    for box in boxes_3d:
        corners_3d = box.corners()
        corners_img = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
        u_min, u_max = corners_img[0].min(), corners_img[0].max()
        v_min, v_max = corners_img[1].min(), corners_img[1].max()
        boxes_2d.append(np.array([u_min, v_min, u_max, v_max])) # xyxy
    return boxes_2d


def filtering_non_vehicles(boxes_3d):
    boxes_3d_filtering = []
    for box in boxes_3d:
        if box.name.split('.')[0] == 'vehicle':
            boxes_3d_filtering.append(box)
    return boxes_3d_filtering


def main():
    model = LangSAM(sam_type='sam2.1_hiera_large',
                    sam_ckpt_path='../checkpoints/sam2.1_hiera_large.pt',
                    gdino_model_ckpt_path='../grounding-dino-base/',
                    gdino_processor_ckpt_path='../grounding-dino-base/')
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)
    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
            for sensor_name in CAM_SENSORS:
                cam_data = nusc.get('sample_data', sample['data'][sensor_name])
                wheels_result = seg_wheels(model, cam_data)
                _, boxes_3d, camera_intrinsic = nusc.get_sample_data(sample['data'][sensor_name],
                                                                  box_vis_level=BoxVisibility.ANY)
                wheels_result, boxes_3d, boxes_2d = post_process_result(wheels_result, boxes_3d, camera_intrinsic)
                save_result_img(cam_data, wheels_result, boxes_2d, sensor_name)


if __name__ == '__main__':
    main()
