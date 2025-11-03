import os
import numpy as np
from lang_sam import LangSAM
from tqdm import tqdm
from lang_sam.utils import load_image, draw_image
import uuid
import json
from PIL import Image

DATA_ROOT_PATH = '/home/danc1nc0de/Datasets/nuScenes'
CAM_SENSOR_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
TXT_PROMPT = 'wheel.'


def save_wheel_img(image_pil, wheels_result, img_wheel_path):
    img_output = image_pil.copy()
    if len(wheels_result['labels']) > 0:
        img_output = draw_image(
            np.asarray(img_output),
            wheels_result["masks"],
            wheels_result["boxes"],
            wheels_result["scores"],
            wheels_result["labels"],
        )
    img_output = Image.fromarray(np.uint8(img_output)).convert("RGB")
    img_output.save(img_wheel_path)


def update_infos(result, img_name):
    result['wheel_num'] = len(result['scores'])
    result.pop('text_labels', None)
    result.pop('labels', None)
    result['wheel_tokens'] = []
    for i in range(result['wheel_num']):
        result['wheel_tokens'].append(uuid.uuid4().hex)


def numpy_to_list(result):
    result_output = result.copy()
    if isinstance(result_output['scores'], np.ndarray):
        result_output['scores'] = result_output['scores'].tolist()
    if isinstance(result_output['boxes'], np.ndarray):
        result_output['boxes'] = result_output['boxes'].tolist()
    if isinstance(result_output['mask_scores'], np.ndarray):
        result_output['mask_scores'] = result_output['mask_scores'].tolist()
    if isinstance(result_output['masks'], np.ndarray):
        masks_lst = result_output['masks'].tolist()
        result_output['masks'] = []
        for i in range(len(masks_lst)):
            mask = []
            for j in range(len(masks_lst[i])):
                for k in range(len(masks_lst[i][j])):
                    if masks_lst[i][j][k] > 0.5:
                        mask.append((j, k))  # h, w
            result_output['masks'].append(mask)
    return result_output


def main():
    model = LangSAM(sam_type='sam2.1_hiera_large',
                    sam_ckpt_path='../checkpoints/sam2.1_hiera_large.pt',
                    gdino_model_ckpt_path='../grounding-dino-base/',
                    gdino_processor_ckpt_path='../grounding-dino-base/')

    for cam_sensor in tqdm(CAM_SENSOR_LIST, desc="cam_sensor", leave=True):
        wheels_result_dict = {}
        json_dir = os.path.join(DATA_ROOT_PATH, 'json_wheel', cam_sensor)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        json_path = os.path.join(json_dir, 'sample_wheel_annotation.json')
        img_dir = os.path.join(DATA_ROOT_PATH, 'samples', cam_sensor)
        img_wheel_dir = os.path.join(DATA_ROOT_PATH, 'samples_wheel', cam_sensor)
        if not os.path.exists(img_wheel_dir):
            os.makedirs(img_wheel_dir)
        for img_name in tqdm(os.listdir(img_dir), desc='img_name', leave=True):
            img_path = os.path.join(img_dir, img_name)
            image_pil = load_image(img_path)
            wheels_result = model.predict([image_pil], [TXT_PROMPT])[0]
            wheels_result_lst = numpy_to_list(wheels_result)
            update_infos(wheels_result_lst, img_name)
            wheels_result_dict[img_name.split('.')[0]] = wheels_result_lst

            img_wheel_name = img_name.split('.')[0] + '_wheel.jpg'
            img_wheel_path = os.path.join(img_wheel_dir, img_wheel_name)
            save_wheel_img(image_pil, wheels_result, img_wheel_path)

        with open(json_path, 'w') as f:
            json.dump(wheels_result_dict, f, indent=2)


if __name__ == '__main__':
    main()
