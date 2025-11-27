import os
import json
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

VERSION = 'v1.0-mini'
DATA_ROOT = os.path.join('/home/danc1nc0de/Datasets/nuScenes')
CAM_SENSORS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def compare_delta_img_corners_2d(delta_box_img_corners_dict, delta_wheel_img_corners_dict):
    u_box_img_corner_lst, delta_yaw_ego_box_u_lst = [], []
    v_box_img_corner_lst, delta_yaw_ego_box_v_lst = [], []
    for img_name in delta_box_img_corners_dict:
        for box_token in delta_box_img_corners_dict[img_name]:
            for data in delta_box_img_corners_dict[img_name][box_token]:
                delta_yaw_ego = data[0]
                delta_u = np.mean(np.abs(np.array(data[1][0])))
                delta_v = np.mean(np.abs(np.array(data[1][1])))
                if delta_v < 1 and delta_u < 10:
                    u_box_img_corner_lst.append(delta_u)
                    delta_yaw_ego_box_u_lst.append(delta_yaw_ego)
                if delta_u < 1:
                    v_box_img_corner_lst.append(delta_v)
                    delta_yaw_ego_box_v_lst.append(delta_yaw_ego)

    u_wheel_img_corner_lst, delta_yaw_ego_wheel_u_lst = [], []
    v_wheel_img_corner_lst, delta_yaw_ego_wheel_v_lst = [], []
    for img_name in delta_wheel_img_corners_dict:
        for box_token in delta_wheel_img_corners_dict[img_name]:
            for data in delta_wheel_img_corners_dict[img_name][box_token]:
                delta_yaw_ego = data[0]
                delta_u = data[1][0]
                delta_v = data[1][1]
                if delta_u < 0 or delta_v < 0:
                    continue
                while delta_yaw_ego > 180:
                    delta_yaw_ego -= 360
                while delta_yaw_ego < -180:
                    delta_yaw_ego += 360
                if delta_v < 1:
                    u_wheel_img_corner_lst.append(delta_u)
                    delta_yaw_ego_wheel_u_lst.append(delta_yaw_ego)
                if delta_u < 1:
                    v_wheel_img_corner_lst.append(delta_v)
                    delta_yaw_ego_wheel_v_lst.append(delta_yaw_ego)

    fig = plt.figure()
    ax_0 = fig.add_subplot(121)
    ax_1 = fig.add_subplot(122)

    # 绘制散点图
    # scatter_0 = ax_0.scatter(u_box_img_corner_lst, delta_yaw_ego_box_u_lst)
    scatter_0 = ax_0.scatter(v_box_img_corner_lst, delta_yaw_ego_box_v_lst)
    # scatter_1 = ax_1.scatter(u_wheel_img_corner_lst, delta_yaw_ego_wheel_u_lst)
    scatter_1 = ax_1.scatter(v_wheel_img_corner_lst, delta_yaw_ego_wheel_v_lst)

    # 添加颜色条
    plt.colorbar(scatter_0)
    plt.colorbar(scatter_1)

    # 设置坐标轴标签
    ax_0.set_xlabel('delta_u')
    ax_0.set_ylabel('delta_yaw')

    ax_1.set_xlabel('delta_u')
    ax_1.set_ylabel('delta_yaw')

    # 显示图形
    plt.show()


def compare_delta_img_corners(delta_box_img_corners_dict, delta_wheel_img_corners_dict):
    u_box_img_corner_lst, v_box_img_corner_lst, delta_yaw_ego_box_lst = [], [], []
    for img_name in delta_box_img_corners_dict:
        for box_token in delta_box_img_corners_dict[img_name]:
            for data in delta_box_img_corners_dict[img_name][box_token]:
                delta_yaw_ego = data[0]
                delta_u = np.mean(np.abs(np.array(data[1][0])))
                delta_v = np.mean(np.abs(np.array(data[1][1])))
                if delta_u > 10:
                    continue
                delta_yaw_ego_box_lst.append(delta_yaw_ego)
                u_box_img_corner_lst.append(delta_u)
                v_box_img_corner_lst.append(delta_v)
        if len(delta_yaw_ego_box_lst) > 10000:
            break

    u_wheel_img_corner_lst, v_wheel_img_corner_lst, delta_yaw_ego_wheel_lst = [], [], []
    for img_name in delta_wheel_img_corners_dict:
        for box_token in delta_wheel_img_corners_dict[img_name]:
            for data in delta_wheel_img_corners_dict[img_name][box_token]:
                delta_yaw_ego = data[0]
                delta_u = data[1][0]
                delta_v = data[1][1]
                if delta_u < 0 or delta_v < 0:
                    continue
                if delta_v > 3:
                    continue
                while delta_yaw_ego > 180:
                    delta_yaw_ego -= 360
                while delta_yaw_ego < -180:
                    delta_yaw_ego += 360
                u_wheel_img_corner_lst.append(delta_u)
                v_wheel_img_corner_lst.append(delta_v)
                delta_yaw_ego_wheel_lst.append(delta_yaw_ego)
        if len(delta_yaw_ego_wheel_lst) > 10000:
            break

    fig = plt.figure()
    ax_0 = fig.add_subplot(121, projection='3d')
    ax_1 = fig.add_subplot(122, projection='3d')

    # 绘制散点图
    scatter_0 = ax_0.scatter(u_box_img_corner_lst, v_box_img_corner_lst, delta_yaw_ego_box_lst, c=delta_yaw_ego_box_lst,
                             cmap='viridis')
    scatter_1 = ax_1.scatter(u_wheel_img_corner_lst, v_wheel_img_corner_lst, delta_yaw_ego_wheel_lst,
                             c=delta_yaw_ego_wheel_lst,
                             cmap='viridis')

    # 添加颜色条
    plt.colorbar(scatter_0)
    plt.colorbar(scatter_1)

    # 设置坐标轴标签
    ax_0.set_xlabel('delta_u')
    ax_0.set_ylabel('delta_v')
    ax_0.set_zlabel('delta_yaw')

    ax_1.set_xlabel('delta_u')
    ax_1.set_ylabel('delta_v')
    ax_1.set_zlabel('delta_yaw')

    # 显示图形
    plt.show()


def load_json(cam_sensor, img_name):
    json_delta_img_corners_dir = os.path.join(DATA_ROOT, VERSION, 'json_delta_img_corners', cam_sensor)
    json_delta_box_img_corners_path = os.path.join(json_delta_img_corners_dir, 'sample_delta_box_img_corners.json')
    json_delta_wheel_img_corners_path = os.path.join(json_delta_img_corners_dir, 'sample_delta_wheel_img_corners.json')
    with open(json_delta_box_img_corners_path, 'r') as f:
        delta_box_img_corners_dict = json.load(f)
    with open(json_delta_wheel_img_corners_path, 'r') as f:
        delta_wheel_img_corners_dict = json.load(f)
    return delta_box_img_corners_dict[img_name], delta_wheel_img_corners_dict[img_name]


def process_cmp(delta_box_img_corners_dict, delta_wheel_img_corners_dict, nusc):
    u_box_img_corner_lst, v_box_img_corner_lst, delta_yaw_ego_box_lst = [], [], []
    u_wheel_img_corner_lst, v_wheel_img_corner_lst, delta_yaw_ego_wheel_lst = [], [], []
    for box_token in delta_box_img_corners_dict:
        sample_anno = nusc.get('sample_annotation', box_token)
        for data in delta_box_img_corners_dict[box_token]:
            delta_yaw_ego = data[0]
            delta_u = np.mean(np.abs(np.array(data[1][0])))
            delta_v = np.mean(np.abs(np.array(data[1][1])))
            if delta_u > 10 or delta_v > 10:
                continue
            if delta_yaw_ego < 0:
                delta_yaw_ego_box_lst.append(delta_yaw_ego)
                u_box_img_corner_lst.append(-delta_u)
                v_box_img_corner_lst.append(-delta_v)
            else:
                delta_yaw_ego_box_lst.append(delta_yaw_ego)
                u_box_img_corner_lst.append(delta_u)
                v_box_img_corner_lst.append(delta_v)
        for data in delta_wheel_img_corners_dict[box_token]:
            delta_yaw_ego = data[0]
            delta_u = data[1][0]
            delta_v = data[1][1]
            if abs(delta_u) > 10 or abs(delta_v) > 10:
                continue
            while delta_yaw_ego > 180:
                delta_yaw_ego -= 360
            while delta_yaw_ego < -180:
                delta_yaw_ego += 360
            u_wheel_img_corner_lst.append(delta_u)
            v_wheel_img_corner_lst.append(delta_v)
            delta_yaw_ego_wheel_lst.append(delta_yaw_ego)

        fig_box = plt.figure(0)
        fig_wheel = plt.figure(1)

        fig_box_ax_u = fig_box.add_subplot(211)
        fig_box_ax_v = fig_box.add_subplot(212)
        fig_wheel_ax = fig_wheel.add_subplot(111, projection='3d')

        fig_box_ax_u.plot(delta_yaw_ego_box_lst, u_box_img_corner_lst)
        fig_box_ax_v.plot(delta_yaw_ego_box_lst, v_box_img_corner_lst)
        fig_wheel_ax.scatter(u_wheel_img_corner_lst, v_wheel_img_corner_lst,
                             delta_yaw_ego_wheel_lst,
                             c=delta_yaw_ego_wheel_lst,
                             cmap='viridis')

        xticks = fig_box_ax_u.get_xticks()
        fig_box_ax_u.set_xticklabels([f"{float(d)}°" for d in xticks])

        xticks = fig_box_ax_v.get_xticks()
        fig_box_ax_v.set_xticklabels([f"{float(d)}°" for d in xticks])

        zticks = fig_wheel_ax.get_zticks()
        fig_wheel_ax.set_zticklabels([f"{float(d)}°" for d in zticks])

        # 设置坐标轴标签
        # fig_box_ax_u.set_xlabel(r'$\Delta \theta$')
        fig_box_ax_u.set_ylabel(r'$\Delta x_i$')

        fig_box_ax_v.set_xlabel(r'$\Delta \theta$')
        fig_box_ax_v.set_ylabel(r'$\Delta y_i$')

        fig_box_ax_u.grid(True, linestyle='--', alpha=0.5)
        fig_box_ax_v.grid(True, linestyle='--', alpha=0.5)

        fig_wheel_ax.set_xlabel(r'$\Delta x_i$')
        fig_wheel_ax.set_ylabel(r'$\Delta y_i$')
        fig_wheel_ax.set_zlabel(r'$\Delta \theta$')
        fig_wheel_ax.view_init(elev=30, azim=150)

        # 显示图形
        plt.show()
        plt.cla()


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    cam_sensor = 'CAM_FRONT'
    # img_name = 'n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448764012460'
    # img_name = 'n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657128112404'
    img_name = 'n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151612362404'
    delta_box_img_corners_dict, delta_wheel_img_corners_dict = load_json(cam_sensor, img_name)
    process_cmp(delta_box_img_corners_dict, delta_wheel_img_corners_dict, nusc)


if __name__ == '__main__':
    main()
