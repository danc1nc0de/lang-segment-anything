import os
import json
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from pyquaternion import Quaternion
from PIL import Image
import cv2

VERSION = 'v1.0-mini'
DATA_ROOT = '/home/danc1nc0de/Datasets/nuScenes/'
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# R G B
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_PINK = (255, 220, 233)


def save_json_wheel(annos_wheel, json_wheel_path):
    with open(json_wheel_path, 'w') as f:
        json.dump(annos_wheel, f, indent=2)


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def get_colors_map(cnt):
    colors_map = []
    for i in range(cnt):
        colors_map.append(np.random.choice(range(256), size=3).tolist())
    return colors_map


def get_img_wheel_path(img_name, cam_sensor):
    img_wheel_dir = os.path.join(DATA_ROOT, 'samples_wheel_assoc_n_filtering', cam_sensor)
    if not os.path.exists(img_wheel_dir):
        os.makedirs(img_wheel_dir)
    img_wheel_name = img_name + '_wheel.jpg'
    img_wheel_path = os.path.join(img_wheel_dir, img_wheel_name)
    return img_wheel_path


def load_orig_img(img_name, cam_sensor):
    img_dir = os.path.join(DATA_ROOT, 'samples', cam_sensor)
    img_path = os.path.join(img_dir, img_name + '.jpg')
    img = load_image(img_path)
    return img


def save_img(img, img_path):
    img.save(img_path)


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


def draw_wheels(img, anno_wheel, boxes_3d, colors_map):
    img_output = np.asarray(img).copy()
    boxes_3d_token_lst = []
    for box_3d in boxes_3d:
        boxes_3d_token_lst.append(box_3d.token)
    mask_tot = []
    for mask_loc in anno_wheel["masks"]:
        mask = np.zeros((img.height, img.width))
        for v, u in mask_loc:
            mask[v][u] = 1
        mask_tot.append(mask)
    if anno_wheel['wheel_num'] > 0:
        colored_mask = np.array(img_output, copy=True, dtype=np.uint8)
        for idx, box_2d in enumerate(anno_wheel['boxes']):
            assoc_box_token = anno_wheel['assoc_box_tokens'][idx]
            idx_box = boxes_3d_token_lst.index(assoc_box_token)
            color = colors_map[idx_box]
            u_min, v_min, u_max, v_max = box_2d
            img_output = cv2.rectangle(img_output, (int(u_min), int(v_min)), (int(u_max), int(v_max)), color, 2)
            colored_mask[mask_tot[idx].astype(bool)] = color
        img_output = cv2.addWeighted(colored_mask, 0.5, img_output, 0.5, 0, dst=img_output)
    return Image.fromarray(np.uint8(img_output)).convert("RGB")


def get_iou(box_0, box_1):
    u_min_0, v_min_0, u_max_0, v_max_0 = box_0
    u_min_1, v_min_1, u_max_1, v_max_1 = box_1
    u_min = max(u_min_0, u_min_1)
    u_max = min(u_max_0, u_max_1)
    v_min = max(v_min_0, v_min_1)
    v_max = min(v_max_0, v_max_1)
    s_0 = (u_max_0 - u_min_0) * (v_max_0 - v_min_0)
    s_1 = (u_max_1 - u_min_1) * (v_max_1 - v_min_1)
    if u_max < u_min or v_max < v_min:
        return 0.0, s_0, s_1, 0.0
    s_i = (u_max - u_min) * (v_max - v_min)
    return s_i / (s_0 + s_1 - s_i), s_0, s_1, s_i


def filtering_false_wheels(anno_wheel):
    anno_wheel_output = {}
    anno_wheel_output['scores'] = []
    anno_wheel_output['boxes'] = []
    anno_wheel_output['masks'] = []
    anno_wheel_output['mask_scores'] = []
    anno_wheel_output['wheel_num'] = 0
    anno_wheel_output['wheel_tokens'] = []

    idx_del_lst = set()
    for idx_0 in range(anno_wheel['wheel_num']):
        for idx_1 in range(anno_wheel['wheel_num']):
            if idx_0 == idx_1:
                continue
            iou, s_0, s_1, _ = get_iou(anno_wheel['boxes'][idx_0], anno_wheel['boxes'][idx_1])
            if iou > 0.1:
                if s_0 > s_1:
                    idx_del_lst.add(idx_1)
                else:
                    idx_del_lst.add(idx_0)
    for idx in range(anno_wheel['wheel_num']):
        if idx in idx_del_lst:
            continue
        anno_wheel_output['scores'].append(anno_wheel['scores'][idx])
        anno_wheel_output['boxes'].append(anno_wheel['boxes'][idx])
        anno_wheel_output['masks'].append(anno_wheel['masks'][idx])
        anno_wheel_output['mask_scores'].append(anno_wheel['mask_scores'][idx])
        anno_wheel_output['wheel_tokens'].append(anno_wheel['wheel_tokens'][idx])
        anno_wheel_output['wheel_num'] += 1
    return anno_wheel_output


def filtering_non_assoc_wheels_n_update_assoc_box(anno_wheel, assoc_box_token_lst):
    anno_wheel_output = {}
    anno_wheel_output['scores'] = []
    anno_wheel_output['boxes'] = []
    anno_wheel_output['masks'] = []
    anno_wheel_output['mask_scores'] = []
    anno_wheel_output['wheel_num'] = 0
    anno_wheel_output['wheel_tokens'] = []
    anno_wheel_output['assoc_box_tokens'] = []
    for idx in range(len(assoc_box_token_lst)):
        if assoc_box_token_lst[idx] is None:
            continue
        anno_wheel_output['scores'].append(anno_wheel['scores'][idx])
        anno_wheel_output['boxes'].append(anno_wheel['boxes'][idx])
        anno_wheel_output['masks'].append(anno_wheel['masks'][idx])
        anno_wheel_output['mask_scores'].append(anno_wheel['mask_scores'][idx])
        anno_wheel_output['wheel_tokens'].append(anno_wheel['wheel_tokens'][idx])
        anno_wheel_output['wheel_num'] += 1
        anno_wheel_output['assoc_box_tokens'].append(assoc_box_token_lst[idx])
    return anno_wheel_output


def get_wheels_3d(annos_wheel, pcs_3d, pcs_2d):
    wheels_3d = []
    pcs_num = pcs_3d.points.shape[1]
    wheels_num = annos_wheel['wheel_num']
    for idx_wheel in range(wheels_num):
        wheel_3d_lst = []
        for idx_p in range(pcs_num):
            if pcs_3d.points[2, idx_p] < 1:
                continue
            u_p, v_p = pcs_2d[0, idx_p], pcs_2d[1, idx_p]
            if [int(v_p), int(u_p)] in annos_wheel['masks'][idx_wheel]:
                wheel_3d_lst.append(pcs_3d.points[:3, idx_p])
        if len(wheel_3d_lst) > 0:
            wheel_3d_lst.sort(key=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            wheels_3d.append(wheel_3d_lst[-1])
        else:
            wheels_3d.append(None)
    return wheels_3d


def mapping_pointcloud_to_image(lidar_data, cam_data, nusc):
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
    pcs_2d = view_points(pcs_3d.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)[:2, :]

    return pcs_3d, pcs_2d


def get_boxes_2d(boxes_3d, cam_data, camera_intrinsic):
    boxes_2d = []
    for box_3d in boxes_3d:
        corners_3d = box_3d.corners()
        corners_img = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
        u_min, u_max = corners_img[0].min(), corners_img[0].max()
        v_min, v_max = corners_img[1].min(), corners_img[1].max()
        u_min, u_max = np.clip(u_min, 0, cam_data['width']), np.clip(u_max, 0, cam_data['width'])
        v_min, v_max = np.clip(v_min, 0, cam_data['height']), np.clip(v_max, 0, cam_data['height'])
        boxes_2d.append(np.array([u_min, v_min, u_max, v_max]))  # xyxy
    return boxes_2d


def do_association(annos_wheel, boxes_3d, cam_data, camera_intrinsic, pcs_3d, pcs_2d):
    assoc_box_token_lst = []
    boxes_2d = get_boxes_2d(boxes_3d, cam_data, camera_intrinsic)
    wheels_3d = get_wheels_3d(annos_wheel, pcs_3d, pcs_2d)
    for idx_wheel in range(annos_wheel['wheel_num']):
        assoc_info = None
        u_min_wheel, v_min_wheel, u_max_wheel, v_max_wheel = annos_wheel['boxes'][idx_wheel]
        area_wheel = (u_max_wheel - u_min_wheel) * (v_max_wheel - v_min_wheel)
        for idx_box in range(len(boxes_2d)):
            u_min_box, v_min_box, u_max_box, v_max_box = boxes_2d[idx_box]
            v_mid_box = v_min_box + (v_max_box - v_min_box) / 4
            u_min_inter = max(u_min_wheel, u_min_box)
            u_max_inter = min(u_max_wheel, u_max_box)
            v_min_inter = max(v_min_wheel, v_mid_box)
            v_max_inter = min(v_max_wheel, v_max_box)
            if u_max_inter < u_min_inter or v_max_inter < v_min_inter:
                continue
            area_inter = (u_max_inter - u_min_inter) * (v_max_inter - v_min_inter)
            if area_inter > 0.9 * area_wheel:
                dis = np.inf
                if wheels_3d[idx_wheel] is not None:
                    dis = np.linalg.norm(boxes_3d[idx_box].center - wheels_3d[idx_wheel])
                if assoc_info is None:
                    assoc_info = [area_inter, dis, boxes_3d[idx_box].token]
                else:
                    if area_inter > assoc_info[0]:
                        assoc_info = [area_inter, dis, boxes_3d[idx_box].token]
                    elif area_inter == assoc_info[0]:
                        if dis < assoc_info[1]:
                            assoc_info = [area_inter, dis, boxes_3d[idx_box].token]
                        else:
                            continue
                    else:
                        continue
        if assoc_info is None:
            assoc_box_token_lst.append(None)
        else:
            assoc_box_token_lst.append(assoc_info[2])
    return assoc_box_token_lst


def filtering_non_vehicles(boxes_3d):
    boxes_3d_output = []
    for box_3d in boxes_3d:
        if box_3d.name.split('.')[0] == 'vehicle':
            boxes_3d_output.append(box_3d)
    return boxes_3d_output


def load_json_wheel(cam_sensor):
    wheel_result = {}
    json_wheel_path = os.path.join(DATA_ROOT, 'json_wheel', cam_sensor, 'sample_wheel_annotation.json')
    with open(json_wheel_path, 'r') as f:
        wheel_result = json.load(f)
    return wheel_result


def get_save_json_path(cam_sensor):
    json_wheel_dir = os.path.join(DATA_ROOT, VERSION, 'json_wheel_assoc_n_filtering', cam_sensor)
    if not os.path.exists(json_wheel_dir):
        os.makedirs(json_wheel_dir)
    json_wheel_path = os.path.join(json_wheel_dir, 'sample_wheel_annotation.json')
    return json_wheel_path, os.path.exists(json_wheel_path)


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    for cam_sensor in tqdm(CAM_SENSORS, desc="cam_sensor"):
        json_wheel_path, flag_path_exists = get_save_json_path(cam_sensor)
        if flag_path_exists:
            continue
        annos_wheel = load_json_wheel(cam_sensor)
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
                lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                cam_data = nusc.get('sample_data', sample['data'][cam_sensor])
                calibrated_sensor_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                img_name = cam_data['filename'].split('/')[-1].split('.')[0]
                # mapping lidar point cloud to image
                pcs_3d, pcs_2d = mapping_pointcloud_to_image(lidar_data, cam_data, nusc)
                # get boxes_3d
                _, boxes_3d_tot, camera_intrinsic = nusc.get_sample_data(sample['data'][cam_sensor],
                                                                         box_vis_level=BoxVisibility.ANY)
                boxes_3d_veh = filtering_non_vehicles(boxes_3d_tot)
                # filtering false wheels
                annos_wheel[img_name] = filtering_false_wheels(annos_wheel[img_name])
                assoc_box_token_lst = do_association(annos_wheel[img_name], boxes_3d_veh, cam_data, camera_intrinsic,
                                                     pcs_3d, pcs_2d)
                annos_wheel[img_name] = filtering_non_assoc_wheels_n_update_assoc_box(annos_wheel[img_name],
                                                                                      assoc_box_token_lst)

                colors_map = get_colors_map(len(boxes_3d_veh))
                img = load_orig_img(img_name, cam_sensor)
                img = draw_boxes_3d(img, boxes_3d_veh, calibrated_sensor_data, colors_map=colors_map)
                img = draw_wheels(img, annos_wheel[img_name], boxes_3d_veh, colors_map=colors_map)
                img_wheel_path = get_img_wheel_path(img_name, cam_sensor)
                save_img(img, img_wheel_path)
        save_json_wheel(annos_wheel, json_wheel_path)


if __name__ == '__main__':
    main()
