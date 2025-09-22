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
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

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
        cv2.rectangle(img_draw, (box_2d[0].astype(np.int32), box_2d[1].astype(np.int32)),
                      (box_2d[2].astype(np.int32), box_2d[3].astype(np.int32)), (0, 255, 0), 2)
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
    wheels_result['assoc_info'] = []
    for idx_wheel in range(len(wheels_result['labels'])):
        assoc_info = None
        u_min_wheel, v_min_wheel, u_max_wheel, v_max_wheel = wheels_result['boxes'][idx_wheel]
        area_wheel = (u_max_wheel - u_min_wheel) * (v_max_wheel - v_min_wheel)
        for idx_box, box_2d in enumerate(boxes_2d):
            u_min_inter = max(u_min_wheel, box_2d[0])
            u_max_inter = min(u_max_wheel, box_2d[2])
            v_min_inter = max(v_min_wheel, box_2d[1])
            v_max_inter = min(v_max_wheel, box_2d[3])
            if u_max_inter < u_min_inter or v_max_inter < v_min_inter:
                continue
            area_inter = (u_max_inter - u_min_inter) * (v_max_inter - v_min_inter)
            if area_inter > 0.9 * area_wheel:
                dis = np.linalg.norm(boxes_3d[idx_box].center - wheels_result['pos_3d'][idx_wheel])
                if assoc_info is None:
                    assoc_info = [area_inter, dis, boxes_3d[idx_box].token]
                else:
                    if area_inter > assoc_info[0]:
                        assoc_info = [area_inter, dis, boxes_3d[idx_box].token]
                    elif area_inter == assoc_info[0]:
                        if dis < assoc_info[1]:
                            assoc_info = [area_inter, dis, boxes_3d[idx_box].token]
        if assoc_info is not None:
            wheels_result['assoc_info'].append(assoc_info[2])
        else:
            wheels_result['assoc_info'].append(None)
    return wheels_result


def post_process_result(wheels_result, boxes_3d, camera_intrinsic, img_size):
    boxes_3d = filtering_non_vehicles(boxes_3d)  # only keep vehicles
    boxes_2d = get_boxes_2d(boxes_3d, camera_intrinsic, img_size)  # get boxes_2d from boxes_3d (cam coordinate)
    # wheels_result = del_small_wheels(wheels_result)
    wheels_result = del_non_vehicle_wheels(wheels_result, boxes_2d)
    return wheels_result, boxes_3d, boxes_2d


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


def filtering_non_vehicles(boxes_3d):
    boxes_3d_filtering = []
    for box in boxes_3d:
        if box.name.split('.')[0] == 'vehicle':
            boxes_3d_filtering.append(box)
    return boxes_3d_filtering


def map_pointcloud_to_image(lidar_data, cam_data, nusc):
    pcl_path = os.path.join(DATA_ROOT, lidar_data['filename'])
    pcs_3d = LidarPointCloud.from_file(pcl_path)

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pcs_3d.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pcs_3d.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pcs_3d.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
    pcs_3d.translate(np.array(pose_record['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    pcs_3d.translate(-np.array(pose_record['translation']))
    pcs_3d.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pcs_3d.translate(-np.array(cs_record['translation']))
    pcs_3d.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    pcs_2d = view_points(pcs_3d.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    return pcs_3d, pcs_2d


def update_wheels_coordinate(wheels_result, pcs_3d, pcs_2d):
    pcs_num = pcs_3d.points.shape[1]
    wheels_num = len(wheels_result['labels'])
    wheels_result['pos_3d'] = []
    for idx_wheel in range(wheels_num):
        u_min_wheel, v_min_wheel, u_max_wheel, v_max_wheel = wheels_result['boxes'][idx_wheel]
        pos_3d = np.zeros(3)
        pos_3d_num = 0
        for idx_p in range(pcs_num):
            if pcs_3d.points[2, idx_p] < 1:
                continue
            u_p, v_p = pcs_2d[0, idx_p], pcs_2d[1, idx_p]
            if u_min_wheel < u_p < u_max_wheel and v_min_wheel < v_p < v_max_wheel:
                pos_3d += pcs_3d.points[:3, idx_p]
                pos_3d_num += 1
        pos_3d /= pos_3d_num
        wheels_result['pos_3d'].append(pos_3d)
    return wheels_result


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
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            nxt_sample_token = sample['next']
            for sensor_name in CAM_SENSORS:
                cam_data = nusc.get('sample_data', sample['data'][sensor_name])
                img_size = (cam_data['height'], cam_data['width'])
                wheels_result = seg_wheels(model, cam_data)
                _, boxes_3d, camera_intrinsic = nusc.get_sample_data(sample['data'][sensor_name],
                                                                     box_vis_level=BoxVisibility.ANY)
                wheels_result, boxes_3d, boxes_2d = post_process_result(wheels_result, boxes_3d, camera_intrinsic,
                                                                        img_size)
                pcs_3d, pcs_2d = map_pointcloud_to_image(lidar_data, cam_data, nusc)
                wheels_result = update_wheels_coordinate(wheels_result, pcs_3d, pcs_2d)
                wheels_result = update_association_info(wheels_result, boxes_2d, boxes_3d)
                save_result_img(cam_data, wheels_result, boxes_2d, sensor_name)


if __name__ == '__main__':
    main()
