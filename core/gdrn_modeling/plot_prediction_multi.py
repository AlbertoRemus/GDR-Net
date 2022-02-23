import cv2
import numpy as np
import pathlib
import os

def read_data_from_file(path, shape=None):
    f = open(path, "r") #opens the file in read mode
    data = f.read() #puts the file into an array
    f.close()

    data_list = [float(i) for i in data.split()]

    if shape is not None:
        data_array = np.reshape(data_list, newshape=shape)
    else:
        data_array = np.reshape(data_list, newshape=(len(data_list) // 2, 2))

    return data_array

dataset_id = "lmo"

object_lists = {"lmo": ["ape", "can", "cat", "driller", "duck", "glue", "eggbox", "glue", "holepuncher"],
                "ycbv": ["002_master_chef_can", "003_cracker_box", "004_sugar_box",
                         "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
                         "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
                         "011_banana", "019_pitcher_base", "021_bleach_cleanser",
                         "024_bowl", "025_mug", "035_power_drill",
                         "036_wood_block", "037_scissors", "040_large_marker",
                         "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"
                         ]}
path_to_datasets = {"lmo": "datasets/BOP_DATASETS/lmo",
                    "ycbv": "datasets/BOP_DATASETS/ycbv"}

path_to_results = {"lmo": "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_gdrn_lmo_real_pbr/lmo_test",
                   "ycbv": "output/gdrn/ycbv/a6_cPnP_AugAAETrunc_BG0.5_Rsym_ycbv_real_pbr_visib20_10e/inference_gdrn_ycbv/ycbv_test"}

image_tuples = {"lmo":  [("000002", "000000"),
                         ("000002", "001118")],
                "ycbv": [("000048", "000001"), ("000048", "000036"), ("000048", "000047"),
                         ("000048", "000112"), ("000048", "000135"), ("000048", "000168"),
                         ("000048", "000181"), ("000048", "002004"), ("000048", "002040"),
                         ("000048", "002090"), ("000048", "002160"), ("000048", "002217"),
                         ("000051", "000001"), ("000051", "000100"), ("000051", "000208"),
                         ("000051", "000304"), ("000051", "000402"), ("000051", "000501"),
                         ("000051", "000601"), ("000051", "000713"), ("000051", "000803"),
                         ("000051", "000912"), ("000051", "001005"), ("000051", "001110"),
                         ("000051", "001220"), ("000051", "001316"), ("000051", "001424"),
                         ("000051", "001518"), ("000051", "001603"), ("000051", "001700"),
                         ("000051", "001803"), ("000051", "001914"), ("000051", "001996"),
                         ]}

for image_tuple in image_tuples[dataset_id]:

    dir_id = image_tuple[0]
    img_id = image_tuple[1]

    input_path = path_to_datasets[dataset_id] + "/test/" + dir_id + "/rgb/" + img_id + ".png"
    image = cv2.imread(input_path)
    image_pr = image.copy()
    image_gt = image.copy()
    image_pr_seg = image.copy()#cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    image_gt_seg = image.copy()#cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    count = 0
    for object_name in object_lists[dataset_id]:
        pr_path = path_to_results[dataset_id] + "/" + object_name + "/pr"
        gt_path = path_to_results[dataset_id] + "/" + object_name + "/gt"

        corners_pr_path = pr_path + "/corners_" + dir_id + "_" + img_id + ".txt"
        corners_pr_all_path = pr_path + "/corners_2d_all_" + dir_id + "_" + img_id + ".txt"
        rotation_pr_path = pr_path + "/R_" + dir_id + "_" + img_id + ".txt"
        translation_pr_path = pr_path + "/t_" + dir_id + "_" + img_id + ".txt"

        corners_gt_path = gt_path + "/corners_" + dir_id + "_" + img_id + ".txt"
        corners_gt_all_path = gt_path + "/corners_2d_all_" + dir_id + "_" + img_id + ".txt"
        rotation_gt_path = gt_path + "/R_" + dir_id + "_" + img_id + ".txt"
        translation_gt_path = gt_path + "/t_" + dir_id + "_" + img_id + ".txt"

        if not os.path.isfile(corners_pr_path):
            continue

        output_path = path_to_results[dataset_id]+"/img"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        # predictions
        corners_pr = read_data_from_file(corners_pr_path, (9, 2)).round().astype(dtype="int32")
        corners_all_pr = read_data_from_file(corners_pr_all_path).round().astype(dtype="int32")
        rotation_pr = read_data_from_file(rotation_pr_path, (3, 3))
        translation_pr = read_data_from_file(translation_pr_path, (3, 1))

        # ground truth
        corners_gt = read_data_from_file(corners_gt_path, (9, 2)).round().astype(dtype="int32")
        corners_all_gt = read_data_from_file(corners_gt_all_path).round().astype(dtype="int32")
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

        height, width, _ = image.shape
        image_pr_seg_mask = np.zeros((height, width), np.uint8)
        image_gt_seg_mask = np.zeros((height, width), np.uint8)

        for p in corners_all_gt:
            image_gt_seg_mask = cv2.circle(image_gt_seg_mask, (int(p[0]), int(p[1])), 1, 255, -1)

        kernel = np.ones((5,5), np.uint8)
        image_gt_seg_mask = cv2.morphologyEx(image_gt_seg_mask, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours_gt, _ = cv2.findContours(image_gt_seg_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        image_gt_seg = cv2.drawContours(image_gt_seg, contours_gt, -1, (255, 0, 0), 4, cv2.LINE_AA) # border

        for p in corners_all_pr:
            image_pr_seg_mask = cv2.circle(image_pr_seg_mask, (int(p[0]), int(p[1])), 1, 255, -1)

        kernel = np.ones((5,5), np.uint8)
        image_pr_seg_mask = cv2.morphologyEx(image_pr_seg_mask, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours_pr, _ = cv2.findContours(image_pr_seg_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        image_pr_seg = cv2.drawContours(image_pr_seg, contours_pr, -1, (0, 0, 255), 4, cv2.LINE_AA) # border

        cv2.putText(image_pr_seg, object_name, (corners_pr[0, 0], corners_pr[0, 1]), cv2.FONT_HERSHEY_PLAIN, 1.6,
                    color=(0, 255, 0), thickness=2)
        cv2.putText(image_gt_seg, object_name, (corners_gt[0, 0], corners_gt[0, 1]), cv2.FONT_HERSHEY_PLAIN, 1.6,
                    color=(0, 255, 0), thickness=2)

    count+=1

    cv2.imwrite(output_path+"/"+dir_id+"_"+img_id+".png", image)
    cv2.imwrite(output_path+"/"+dir_id+"_"+img_id+"_pr.png", image_pr)
    cv2.imwrite(output_path+"/"+dir_id+"_"+img_id+"_gt.png", image_gt)
    cv2.imwrite(output_path + "/" + dir_id + "_" + img_id + "_gt_seg.png", image_gt_seg)
    cv2.imwrite(output_path + "/" + dir_id + "_" + img_id + "_pr_seg.png", image_pr_seg)
    cv2.imwrite(output_path + "/" + dir_id + "_" + img_id + "_gt_seg_mask.png", image_gt_seg_mask)
    cv2.imwrite(output_path + "/" + dir_id + "_" + img_id + "_pr_seg_mask.png", image_pr_seg_mask)

print("Images saved")
