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
DATA_ROOT = os.path.join('/home/danc1nc0de/Datasets/nuScenes', VERSION)
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# R G B
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_SKY_BLUE = (135, 206, 235)


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


def draw_wheels(img, wheel_masks, wheel_boxes, color=COLOR_RED):
    for wheel_box in wheel_boxes:
        u_min, v_min, u_max, v_max = wheel_box
        img = cv2.rectangle(img, (int(u_min), int(v_min)), (int(u_max), int(v_max)), color, 2)
        img = cv2.putText(img, 'w {:.2f} h {:.2f}'.format(u_max - u_min, v_max - v_min), (int(u_max), int(v_max)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    colored_mask = np.array(img, copy=True, dtype=np.uint8)
    for wheel_mask in wheel_masks:
        colored_mask[wheel_mask.astype(bool)] = color
    img = cv2.addWeighted(colored_mask, 0.5, img, 0.5, 0, dst=img)
    return img


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


def get_wheel_ground_point(wheel_anno):
    u_grd, v_grd = -1.0, -1.0
    for v, u in wheel_anno['mask']:
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
                update_box_wheel_direction(wheel_direction_ego, wheel_yaw, box_ego, P_i_0, P_i_1, wheel_token_0,
                                           wheel_token_1)


def filtering_not_valid_wheels(wheel_annos):
    wheel_annos_output = {}
    for sensor_name in wheel_annos:
        if sensor_name not in wheel_annos_output:
            wheel_annos_output[sensor_name] = {}
        for box_token in wheel_annos[sensor_name]:
            if box_token not in wheel_annos_output[sensor_name]:
                wheel_annos_output[sensor_name][box_token] = []
            wheel_annos_lst = []
            wheel_del_idx_st = set()
            for wheel_anno in wheel_annos[sensor_name][box_token]:
                u_min, v_min, u_max, v_max = wheel_anno['box']
                width, height = u_max - u_min, v_max - v_min
                if height < width * 0.7:
                    continue
                if width < 15 or height < 15:
                    continue
                if width < 20 and height < 20:
                    continue
                wheel_annos_lst.append(wheel_anno)
            for idx_0, wheel_anno_0 in enumerate(wheel_annos_lst):
                for idx_1, wheel_anno_1 in enumerate(wheel_annos_lst):
                    if idx_0 == idx_1:
                        continue
                    iou, s_0, s_1, _ = get_iou(wheel_anno_0['box'], wheel_anno_1['box'])
                    if iou > 0.1:
                        if s_0 > s_1:
                            wheel_del_idx_st.add(idx_1)
                        else:
                            wheel_del_idx_st.add(idx_0)
            for idx, wheel_anno in enumerate(wheel_annos_lst):
                if idx in wheel_del_idx_st:
                    continue
                wheel_annos_output[sensor_name][box_token].append(wheel_anno)
    return wheel_annos_output


def filtering_boxes(boxes_ego, wheel_annos, sensor_name):
    boxes_ego_output = []
    for box_ego in boxes_ego:
        # filtering non-vehicle
        if box_ego.name.split('.')[0] != 'vehicle':
            continue
        # filtering no wheels
        if box_ego.token not in wheel_annos[sensor_name]:
            continue
        # filtering less than 2 wheels
        if len(wheel_annos[sensor_name][box_ego.token]) < 2:
            continue
        boxes_ego_output.append(box_ego)
    return boxes_ego_output


def get_boxes_ego(nusc, sample, sensor_name):
    sample_data_token = sample['data'][sensor_name]
    boxes_world = nusc.get_boxes(sample_data_token)  # boxes in world coordinate
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    boxes_ego = []
    for box_world in boxes_world:
        box_ego = box_world.copy()
        box_ego.translate(-np.array(ego_pose['translation']))
        box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        boxes_ego.append(box_ego)
    return boxes_ego


def load_wheel_annos():
    wheel_annos_json_path = os.path.join(DATA_ROOT, VERSION, 'wheel_annotation.json')
    wheel_annos = {}
    with open(wheel_annos_json_path, 'r') as f:
        wheel_annos_json = json.load(f)
        for wheel_anno in wheel_annos_json:
            sample_annotation_token = wheel_anno['sample_annotation_token']
            sensor_name = wheel_anno['sensor_name']
            if sensor_name not in wheel_annos:
                wheel_annos[sensor_name] = {}
            if sample_annotation_token not in wheel_annos[sensor_name]:
                wheel_annos[sensor_name][sample_annotation_token] = []
            wheel_annos[sensor_name][sample_annotation_token].append(wheel_anno)
    return wheel_annos


def main():
    wheel_annos_tot = load_wheel_annos()
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        nxt_sample_token = first_sample_token
        while nxt_sample_token != '':
            sample = nusc.get('sample', nxt_sample_token)
            nxt_sample_token = sample['next']
            for sensor_name in CAM_SENSORS:
                sample_sensor_data = nusc.get('sample_data', sample['data'][sensor_name])
                img_name = sample_sensor_data['filename'].split('/')[-1].split('.')[0]
                if img_name == 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151614412404':
                    pass
                # calibration data of sensor to ego
                calibrated_sensor_data = nusc.get('calibrated_sensor', sample_sensor_data['calibrated_sensor_token'])
                boxes_ego_tot = get_boxes_ego(nusc, sample, sensor_name)
                # filtering not valid wheels
                wheel_annos = filtering_not_valid_wheels(wheel_annos_tot)
                wheel_annos_tot = wheel_annos
                boxes_ego = filtering_boxes(boxes_ego_tot, wheel_annos, sensor_name)
                update_wheel_direction(boxes_ego, wheel_annos, sensor_name, calibrated_sensor_data)
                vis(sample_sensor_data, sensor_name, boxes_ego_tot, boxes_ego, wheel_annos_tot, wheel_annos,
                    calibrated_sensor_data)


if __name__ == '__main__':
    main()
