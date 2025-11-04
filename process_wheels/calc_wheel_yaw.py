from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
import json
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility, view_points
from lang_sam.utils import draw_image, load_image
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

VERSION = 'v1.0-mini'
DATA_ROOT = os.path.join('/home/danc1nc0de/Datasets/nuScenes')
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# R G B
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_SKY_BLUE = (135, 206, 235)


def draw_wheel_direction(img, boxes_3d_ego, boxes_sensor, color=COLOR_GREEN):
    img_output = np.asarray(img).copy()
    for box_ego, box_sensor in zip(boxes_3d_ego, boxes_sensor):
        yaw_box_ego, _, _ = R.from_matrix(box_ego.rotation_matrix).as_euler('zyx', degrees=False)
        if box_sensor.wheel_direction_valid:
            # draw line for ground point
            u_min = box_sensor.wheel_ground_point[0]
            v_min = box_sensor.wheel_ground_point[1]
            u_max = box_sensor.wheel_ground_point[2]
            v_max = box_sensor.wheel_ground_point[3]
            u_txt, v_txt = (u_min + u_max) / 2, v_max
            cv2.line(img_output, (int(u_min), int(v_min)),
                     (int(u_max), int(v_max)), color, 5)
            cv2.putText(img_output, 'yaw_box {:.2f}'.format(yaw_box_ego),
                        (int(u_txt), int(v_txt) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img_output, 'yaw_wheel {:.2f}'.format(box_sensor.wheel_yaw),
                        (int(u_txt), int(v_txt) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return Image.fromarray(np.uint8(img_output)).convert("RGB")


def save_img(img, img_path):
    img.save(img_path)


def get_img_wheel_path(img_name, cam_sensor):
    img_wheel_dir = os.path.join(DATA_ROOT, 'samples_wheel_calc_yaw', cam_sensor)
    if not os.path.exists(img_wheel_dir):
        os.makedirs(img_wheel_dir)
    img_wheel_name = img_name + '_wheel.jpg'
    img_wheel_path = os.path.join(img_wheel_dir, img_wheel_name)
    return img_wheel_path


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


def convert_boxes_3d_ego_to_sensor(boxes_3d_ego, sample_sensor_data, calibrated_sensor_data):
    cam_intrinsic = np.array(calibrated_sensor_data['camera_intrinsic'])
    imsize = (sample_sensor_data['width'], sample_sensor_data['height'])
    boxes_3d_sensor = []
    for box_3d_ego in boxes_3d_ego:
        box_3d_sensor = box_3d_ego.copy()
        #  Move box to sensor coord system.
        box_3d_sensor.translate(-np.array(calibrated_sensor_data['translation']))
        box_3d_sensor.rotate(Quaternion(calibrated_sensor_data['rotation']).inverse)
        if not box_in_image(box_3d_sensor, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
            continue
        boxes_3d_sensor.append(box_3d_sensor)
    return boxes_3d_sensor


def load_orig_img(img_name, cam_sensor):
    img_dir = os.path.join(DATA_ROOT, 'samples', cam_sensor)
    img_path = os.path.join(img_dir, img_name + '.jpg')
    img = load_image(img_path)
    return img


def filtering_non_vehicles(boxes_3d):
    boxes_3d_output = []
    for box_3d in boxes_3d:
        if box_3d.name.split('.')[0] == 'vehicle':
            boxes_3d_output.append(box_3d)
    return boxes_3d_output


def get_colors_map(cnt):
    colors_map = []
    for i in range(cnt):
        colors_map.append(np.random.choice(range(256), size=3).tolist())
    return colors_map


def finetune_box_using_wheel_yaw(box):
    box_output = box.copy()

    x, y, z = box_output.center

    wheel_yaw = box.wheel_yaw
    wheel_pitch = np.arctan2(box.wheel_direction[2], np.abs(box.wheel_direction[0]))
    box_orientation = Quaternion(R.from_matrix(box_output.orientation.rotation_matrix).as_quat(scalar_first=True))
    box_yaw, box_pitch, box_roll = R.from_matrix(box_output.rotation_matrix).as_euler('zyx', degrees=False)
    rotation_quat = R.from_euler('zyx', [wheel_yaw, wheel_pitch, box_roll], degrees=False).as_quat(scalar_first=True)

    box_output.translate(-np.array([x, y, z]))
    box_output.rotate(box_orientation.inverse)
    box_output.rotate(Quaternion(rotation_quat))
    box_output.translate(np.array([x, y, z]))
    return box_output


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


def vis_wheel_direction_tot(img, boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data):
    # calc calibration
    R_sensor_to_ego = Quaternion(calibrated_sensor_data['rotation']).rotation_matrix
    K = np.array(calibrated_sensor_data['camera_intrinsic'])
    U = R_sensor_to_ego @ np.linalg.inv(K)
    u = U[2, :]
    for box_ego in boxes_ego:
        wheel_anno_lst = wheel_annos[sensor_name][box_ego.token]
        for i in range(len(wheel_anno_lst)):
            wheel_anno_0 = wheel_anno_lst[i]
            wheel_token_0 = wheel_anno_0['token']
            u_grd_0, v_grd_0 = get_wheel_ground_point(wheel_anno_0)
            P_i_0 = np.array([u_grd_0, v_grd_0, 1.0])
            for j in range(i + 1, len(wheel_anno_lst)):
                wheel_anno_1 = wheel_anno_lst[j]
                wheel_token_1 = wheel_anno_1['token']
                u_grd_1, v_grd_1 = get_wheel_ground_point(wheel_anno_1)
                P_i_1 = np.array([u_grd_1, v_grd_1, 1.0])
                wheel_direction_ego = U @ ((u.T @ P_i_0) * P_i_1 - (u.T @ P_i_1) * P_i_0)
                wheel_yaw = np.arctan2(wheel_direction_ego[1], wheel_direction_ego[0])
                wheel_yaw_deg = np.degrees(wheel_yaw)
                img = cv2.line(img, (int(u_grd_0), int(v_grd_0)), (int(u_grd_1), int(v_grd_1)), (0, 255, 0), 2)
                img = cv2.putText(img, '%.2f' % wheel_yaw, (int((u_grd_0 + u_grd_1) / 2), int((v_grd_0 + v_grd_1) / 2)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


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


def vis_wheel_direction(img, boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data, color_box=COLOR_RED,
                        color_direction=COLOR_GREEN):
    cam_intrinsic = np.array(calibrated_sensor_data['camera_intrinsic'])
    wheel_boxes = []
    wheel_masks = []
    for box in boxes_ego:
        if box.wheel_direction_valid:
            box_output = finetune_box_using_wheel_yaw(box)
            # move box to sensor coord system.
            box_output.translate(-np.array(calibrated_sensor_data['translation']))
            box_output.rotate(Quaternion(calibrated_sensor_data['rotation']).inverse)
            # filtering out of image
            if not box_in_image(box_output, cam_intrinsic, (img.shape[1], img.shape[0]), vis_level=BoxVisibility.ANY):
                continue
            box_output.render_cv2(img, view=cam_intrinsic, normalize=True,
                                  colors=(color_box[::-1], color_box[::-1], color_box[::-1]))
            cv2.line(img, (int(box_output.wheel_ground_point[0]), int(box_output.wheel_ground_point[1])),
                     (int(box_output.wheel_ground_point[2]), int(box_output.wheel_ground_point[3])), color_direction, 2)
            token = box_output.token
            if token in wheel_annos[sensor_name]:
                for wheel in wheel_annos[sensor_name][token]:
                    if wheel['token'] in box_output.wheel_token:
                        wheel_boxes.append(wheel['box'])
                        wheel_mask = np.zeros((img.shape[0], img.shape[1]))
                        for x, y in wheel['mask']:
                            wheel_mask[x, y] = 1
                        wheel_masks.append(wheel_mask)
    if len(wheel_boxes):
        img = draw_wheels(img, wheel_masks, wheel_boxes, color=color_box)
    return img


def vis_wheels(img, boxes_ego_tot, wheel_annos_tot, sensor_name, sample_sensor_data, color=COLOR_SKY_BLUE):
    wheel_boxes = []
    wheel_masks = []
    for box in boxes_ego_tot:
        if box.name.split('.')[0] == 'vehicle':
            token = box.token
            if token in wheel_annos_tot[sensor_name]:
                for wheel in wheel_annos_tot[sensor_name][token]:
                    wheel_boxes.append(wheel['box'])
                    wheel_mask = np.zeros((sample_sensor_data['height'], sample_sensor_data['width']))
                    for x, y in wheel['mask']:
                        wheel_mask[x, y] = 1
                    wheel_masks.append(wheel_mask)
    if len(wheel_boxes):
        img = draw_wheels(img, wheel_masks, wheel_boxes, color=color)
    return img


def vis_boxes(img, boxes_ego_tot, calibrated_sensor_data, color=COLOR_BLUE):
    cam_intrinsic = np.array(calibrated_sensor_data['camera_intrinsic'])
    for box in boxes_ego_tot:
        # filtering non-vehicle
        if box.name.split('.')[0] != 'vehicle':
            continue
        # move box to sensor coord system.
        box.translate(-np.array(calibrated_sensor_data['translation']))
        box.rotate(Quaternion(calibrated_sensor_data['rotation']).inverse)
        # filtering out of image
        if not box_in_image(box, cam_intrinsic, (img.shape[1], img.shape[0]), vis_level=BoxVisibility.ANY):
            # recover to ego coordinate
            box.rotate(Quaternion(calibrated_sensor_data['rotation']))
            box.translate(np.array(calibrated_sensor_data['translation']))
            continue
        box.render_cv2(img, view=cam_intrinsic, normalize=True, colors=(color[::-1], color[::-1], color[::-1]))
        corners = view_points(box.corners(), cam_intrinsic, normalize=True)[:2, :]
        center_top = np.mean(corners.T[[0, 1, 5, 4]], axis=0)
        # recover to ego coordinate
        box.rotate(Quaternion(calibrated_sensor_data['rotation']))
        box.translate(np.array(calibrated_sensor_data['translation']))
        yaw, _, _ = R.from_matrix(box.rotation_matrix).as_euler('zyx', degrees=False)
        img = cv2.putText(img, '%.2f' % yaw, (int(center_top[0]), int(center_top[1])),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def vis(sample_sensor_data, sensor_name, boxes_ego_tot, boxes_ego, wheel_annos_tot, wheel_annos,
        calibrated_sensor_data):
    img_path = os.path.join(DATA_ROOT, sample_sensor_data['filename'])
    img = np.asarray(load_image(img_path)).copy()

    output_path = os.path.join(DATA_ROOT, 'wheels', sensor_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_name = sample_sensor_data['filename'].split('/')[-1].split('.')[0]
    output_path = os.path.join(output_path, img_name + '_wheels.jpg')

    img = vis_boxes(img, boxes_ego_tot, calibrated_sensor_data, color=COLOR_SKY_BLUE)
    img = vis_wheels(img, boxes_ego_tot, wheel_annos_tot, sensor_name, sample_sensor_data, color=COLOR_SKY_BLUE)
    img = vis_wheel_direction(img, boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data, color_box=COLOR_RED,
                              color_direction=COLOR_GREEN)
    # img = vis_wheel_direction_tot(img, boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data)

    img = Image.fromarray(np.uint8(img)).convert("RGB")
    img.save(output_path)


def update_box_wheel_direction(wheel_direction_ego, wheel_yaw, box_ego, P_i_0, P_i_1, wheel_token_0, wheel_token_1,
                               thr=np.radians(45.0)):
    wheel_yaw_valid = 0
    box_ego_yaw, _, _ = R.from_matrix(box_ego.rotation_matrix).as_euler('zyx', degrees=False)
    if np.abs(wheel_yaw - box_ego_yaw) < thr:
        wheel_yaw_valid = 1
        # wheel_direction_ego = wheel_direction_ego
        # wheel_yaw = wheel_yaw
        # P_i_0, P_i_1 = P_i_0, P_i_1
        # wheel_token_0, wheel_token_1 = wheel_token_0, wheel_token_1
    elif np.abs(wheel_yaw + np.pi - box_ego_yaw) < thr:
        wheel_yaw_valid = 1
        wheel_direction_ego = -wheel_direction_ego
        wheel_yaw = wheel_yaw + np.pi
        P_i_0, P_i_1 = P_i_1, P_i_0
        wheel_token_0, wheel_token_1 = wheel_token_1, wheel_token_0
    elif np.abs(wheel_yaw - np.pi - box_ego_yaw) < thr:
        wheel_yaw_valid = 1
        wheel_direction_ego = -wheel_direction_ego
        wheel_yaw = wheel_yaw - np.pi
        P_i_0, P_i_1 = P_i_1, P_i_0
        wheel_token_0, wheel_token_1 = wheel_token_1, wheel_token_0
    elif np.abs(wheel_yaw + 2. * np.pi - box_ego_yaw) < thr:
        wheel_yaw_valid = 1
        # wheel_direction_ego = wheel_direction_ego
        wheel_yaw = wheel_yaw + 2. * np.pi
        # P_i_0, P_i_1 = P_i_0, P_i_1
        # wheel_token_0, wheel_token_1 = wheel_token_0, wheel_token_1
    elif np.abs(wheel_yaw - 2. * np.pi - box_ego_yaw) < thr:
        wheel_yaw_valid = 1
        # wheel_direction_ego = wheel_direction_ego
        wheel_yaw = wheel_yaw - 2. * np.pi
        # P_i_0, P_i_1 = P_i_0, P_i_1
        # wheel_token_0, wheel_token_1 = wheel_token_0, wheel_token_1
    else:
        wheel_yaw_valid = 0
    yaw_diff = np.abs(wheel_yaw - box_ego_yaw)

    if wheel_yaw_valid > 0:
        # already have a valid wheel direction
        if box_ego.wheel_direction_valid:
            if yaw_diff < np.abs(box_ego.wheel_yaw - box_ego_yaw):
                box_ego.wheel_direction = wheel_direction_ego
                box_ego.wheel_yaw = wheel_yaw
                box_ego.wheel_ground_point = [P_i_0[0], P_i_0[1], P_i_1[0], P_i_1[1]]
                box_ego.wheel_token = [wheel_token_0, wheel_token_1]
        else:
            box_ego.wheel_direction_valid = True
            box_ego.wheel_direction = wheel_direction_ego
            box_ego.wheel_yaw = wheel_yaw
            box_ego.wheel_ground_point = [P_i_0[0], P_i_0[1], P_i_1[0], P_i_1[1]]
            box_ego.wheel_token = [wheel_token_0, wheel_token_1]


def get_wheel_ground_point(wheel_anno_mask):
    u_grd, v_grd = -1.0, -1.0
    for v, u in wheel_anno_mask:
        if v > v_grd:
            u_grd, v_grd = u, v
    return u_grd, v_grd


def update_wheel_direction(boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data):
    # calc calibration
    R_sensor_to_ego = Quaternion(calibrated_sensor_data['rotation']).rotation_matrix
    K = np.array(calibrated_sensor_data['camera_intrinsic'])
    U = R_sensor_to_ego @ np.linalg.inv(K)
    u = U[2, :]
    for box_ego in boxes_ego:
        idxes_wheel = [idx for idx, token in enumerate(wheel_annos['assoc_box_tokens']) if token == box_ego.token]
        for i in range(len(idxes_wheel)):
            idx_i = idxes_wheel[i]
            wheel_token_0 = wheel_annos['wheel_tokens'][idx_i]
            u_grd_0, v_grd_0 = get_wheel_ground_point(wheel_annos['masks'][idx_i])
            P_i_0 = np.array([u_grd_0, v_grd_0, 1.0])
            for j in range(i + 1, len(idxes_wheel)):
                idx_j = idxes_wheel[j]
                wheel_token_1 = wheel_annos['wheel_tokens'][idx_j]
                u_grd_1, v_grd_1 = get_wheel_ground_point(wheel_annos['masks'][idx_j])
                P_i_1 = np.array([u_grd_1, v_grd_1, 1.0])
                wheel_direction_ego = U @ ((u.T @ P_i_0) * P_i_1 - (u.T @ P_i_1) * P_i_0)
                wheel_yaw = np.arctan2(wheel_direction_ego[1], wheel_direction_ego[0])
                update_box_wheel_direction(wheel_direction_ego, wheel_yaw, box_ego,
                                           P_i_0, P_i_1, wheel_token_0, wheel_token_1)


def filtering_not_valid_wheels_for_calc_yaw(wheel_annos, sample_sensor_data):
    wheel_annos_output = {}
    wheel_annos_output['scores'] = []
    wheel_annos_output['boxes'] = []
    wheel_annos_output['masks'] = []
    wheel_annos_output['mask_scores'] = []
    wheel_annos_output['wheel_num'] = 0
    wheel_annos_output['wheel_tokens'] = []
    wheel_annos_output['assoc_box_tokens'] = []
    wheel_del_idx_st = set()
    for idx in range(wheel_annos['wheel_num']):
        u_min, v_min, u_max, v_max = wheel_annos['boxes'][idx]
        width, height = u_max - u_min, v_max - v_min
        if height < width * 0.7:
            wheel_del_idx_st.add(idx)
            continue
        if width < 15 or height < 15:
            wheel_del_idx_st.add(idx)
            continue
        if width < 20 and height < 20:
            wheel_del_idx_st.add(idx)
            continue
        if u_min < 10 or v_min < 10 or \
                u_max + 10 > sample_sensor_data['width'] or v_max + 10 > sample_sensor_data['height']:
            wheel_del_idx_st.add(idx)
            continue
    for idx in range(wheel_annos['wheel_num']):
        if idx in wheel_del_idx_st:
            continue
        wheel_annos_output['scores'].append(
            wheel_annos['scores'][idx])
        wheel_annos_output['boxes'].append(
            wheel_annos['boxes'][idx])
        wheel_annos_output['masks'].append(
            wheel_annos['masks'][idx])
        wheel_annos_output['mask_scores'].append(
            wheel_annos['mask_scores'][idx])
        wheel_annos_output['wheel_tokens'].append(
            wheel_annos['wheel_tokens'][idx])
        wheel_annos_output['wheel_num'] += 1
        wheel_annos_output['assoc_box_tokens'].append(
            wheel_annos['assoc_box_tokens'][idx])
    return wheel_annos_output


def filtering_not_valid_boxes_for_calc_yaw(boxes_ego, wheel_annos):
    boxes_ego_output = []
    for box_ego in boxes_ego:
        # filtering no wheels
        if box_ego.token not in wheel_annos['assoc_box_tokens']:
            continue
        # filtering less than 2 wheels
        if wheel_annos['assoc_box_tokens'].count(box_ego.token) < 2:
            continue
        boxes_ego_output.append(box_ego)
    return boxes_ego_output


def get_boxes_3d_ego(nusc, sample, sensor_name):
    sample_data_token = sample['data'][sensor_name]
    boxes_3d_world = nusc.get_boxes(sample_data_token)  # boxes in world coordinate
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    boxes_3d_ego = []
    for box_3d_world in boxes_3d_world:
        box_3d_ego = box_3d_world.copy()
        box_3d_ego.translate(-np.array(ego_pose['translation']))
        box_3d_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        boxes_3d_ego.append(box_3d_ego)
    return boxes_3d_ego


def load_json_wheel():
    wheel_result = {}
    for cam_sensor in CAM_SENSORS:
        json_wheel_path = os.path.join(DATA_ROOT, 'json_wheel_assoc_n_filtering', cam_sensor,
                                       'sample_wheel_annotation.json')
        with open(json_wheel_path, 'r') as f:
            wheel_result[cam_sensor] = json.load(f)
    return wheel_result


def main():
    annos_wheel_tot = load_json_wheel()
    annos_wheel = load_json_wheel()
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    for scene in tqdm(nusc.scene):
        sample_token_lst = []
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample_token_lst.append(nxt_sample_token)
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
        for sample_token in tqdm(sample_token_lst):
            sample = nusc.get('sample', sample_token)
            for cam_sensor in CAM_SENSORS:
                sample_sensor_data = nusc.get('sample_data', sample['data'][cam_sensor])
                img_name = sample_sensor_data['filename'].split('/')[-1].split('.')[0]
                # calibration data of sensor to ego
                calibrated_sensor_data = nusc.get('calibrated_sensor', sample_sensor_data['calibrated_sensor_token'])
                # get boxes_3d_ego
                boxes_3d_ego_tot = get_boxes_3d_ego(nusc, sample, cam_sensor)
                boxes_3d_ego_veh = filtering_non_vehicles(boxes_3d_ego_tot)
                boxes_3d_sensor_veh = convert_boxes_3d_ego_to_sensor(boxes_3d_ego_veh, sample_sensor_data,
                                                                     calibrated_sensor_data)
                # filtering not valid wheels for calc yaw
                annos_wheel[cam_sensor][img_name] = filtering_not_valid_wheels_for_calc_yaw(
                    annos_wheel_tot[cam_sensor][img_name], sample_sensor_data)
                # filtering not valid boxes for calc yaw
                boxes_3d_ego_filtering = filtering_not_valid_boxes_for_calc_yaw(boxes_3d_ego_veh,
                                                                                annos_wheel[cam_sensor][img_name])
                update_wheel_direction(boxes_3d_ego_filtering, annos_wheel[cam_sensor][img_name], cam_sensor,
                                       calibrated_sensor_data)
                boxes_3d_sensor_filtering = convert_boxes_3d_ego_to_sensor(boxes_3d_ego_filtering, sample_sensor_data,
                                                                           calibrated_sensor_data)
                colors_map = get_colors_map(len(boxes_3d_ego_veh))
                img = load_orig_img(img_name, cam_sensor)
                img = draw_boxes_3d(img, boxes_3d_sensor_veh, calibrated_sensor_data, colors_map=colors_map)
                img = draw_wheels(img, annos_wheel_tot[cam_sensor][img_name], boxes_3d_sensor_veh,
                                  colors_map=colors_map)
                img = draw_wheel_direction(img, boxes_3d_ego_filtering, boxes_3d_sensor_filtering)
                img_wheel_path = get_img_wheel_path(img_name, cam_sensor)
                save_img(img, img_wheel_path)


if __name__ == '__main__':
    main()
