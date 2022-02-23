import cv2
import numpy as np
import pathlib
import os

def read_data_from_file(path, shape):
    f = open(path, "r") #opens the file in read mode
    data = f.read() #puts the file into an array

    data_list = [float(i) for i in data.split()]
    data_array = np.reshape(data_list, newshape=shape)

    f.close()
    return data_array

object_list = ["ape", "can", "cat", "driller", "duck", "glue", "eggbox", "glue", "holepuncher"]

path_to_test_images = "datasets/BOP_DATASETS/lmo/test/000002/rgb"
path_to_results = "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_gdrn_lmo_real_pbr/lmo_test"

image_ids = ["0000", "0100", "0200", "0300", "0400", "0500", "0600", "0700", "0800", "0900", "1118"]

for image_id in image_ids:
    input_path = path_to_test_images + "/00" + image_id + ".png"
    image = cv2.imread(input_path)
    image_pr = image.copy()
    image_gt = image.copy()

    count = 0
    for object_name in object_list:
        pr_path = path_to_results+"/"+object_name+"/pr"
        gt_path = path_to_results+"/" + object_name + "/gt"

        corners_pr_path = pr_path + "/corners_00" + image_id + ".txt"
        rotation_pr_path = pr_path + "/R_00" + image_id + ".txt"
        translation_pr_path = pr_path + "/t_00" + image_id + ".txt"
        corners_gt_path = gt_path + "/corners_00" + image_id + ".txt"
        rotation_gt_path = gt_path + "/R_00" + image_id + ".txt"
        translation_gt_path = gt_path + "/t_00" + image_id + ".txt"

        if not os.path.isfile(corners_pr_path):
            continue

        output_path = path_to_results+"/img"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        # predictions
        corners_pr = read_data_from_file(corners_pr_path, (9, 2)).round().astype(dtype="int32")
        rotation_pr = read_data_from_file(rotation_pr_path, (3, 3))
        translation_pr = read_data_from_file(translation_pr_path, (3, 1))

        # ground truth
        corners_gt = read_data_from_file(corners_gt_path, (9, 2)).round().astype(dtype="int32")
        rotation_gt = read_data_from_file(rotation_gt_path, (3, 3))
        translation_gt = read_data_from_file(translation_gt_path, (3, 1))

        corner_idx_1 = [1, 2, 4, 3, 5, 6, 8, 7, 1, 2, 3, 4]
        corner_idx_2 = [2, 4, 3, 1, 6, 8, 7, 5, 5, 6, 7, 8]


        #BGR color convention in opencv
        for i in range(0, len(corner_idx_1)):
            image_pr = cv2.line(image_pr, corners_pr[corner_idx_1[i]], corners_pr[corner_idx_2[i]], (0, 0, 255), thickness=2)
            image_gt = cv2.line(image_gt, corners_gt[corner_idx_1[i]], corners_gt[corner_idx_2[i]], (255, 0, 0), thickness=2)

        cv2.putText(image_pr, object_name, (corners_pr[0, 0], corners_pr[0, 1]), cv2.FONT_HERSHEY_PLAIN, 1.6,
                    color=(0, 255, 0), thickness=2)
        cv2.putText(image_gt, object_name, (corners_gt[0, 0], corners_gt[0, 1]), cv2.FONT_HERSHEY_PLAIN, 1.6,
                    color=(0, 255, 0), thickness=2)

    count+=1

    cv2.imwrite(output_path+"/"+image_id+".png", image)
    cv2.imwrite(output_path+"/"+image_id+"_pr.png", image_pr)
    cv2.imwrite(output_path+"/"+image_id+"_gt.png", image_gt)

print("Images saved")
