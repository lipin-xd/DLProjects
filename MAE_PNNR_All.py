# _*_ coding: utf-8 _*_
"""
Time:     2022/12/11 19:42
Author:   YYLin
File:     MAE_PNNR_All.py
e-mail: 854280599@qq.com
"""
import os, cv2
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim as ssim
import warnings
warnings.filterwarnings('ignore')


def mae(img1, img2):
    mae = np.mean(abs(img1 - img2))
    return mae


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


source_dir = 'Last_Best_Result'
target_img_path = os.path.join(source_dir, 'art_img_test')
target_liver_path = os.path.join(source_dir, 'nc_liver_test')
source_tumor_path = os.path.join(source_dir, 'nc_liver_test')

generated_dirs = []
for tmp_dir in os.listdir(source_dir):
    if 'art' in tmp_dir:
    # if 'pv' in tmp_dir:
        generated_dirs.append(os.path.join(source_dir, tmp_dir))

# img, liver, tumor
type_mae_psnr = 'img'
sum_mae = 0
sum_psnr = 0
sum_ssim = 0
sum_my_psnr = 0
num_img = 0

img_size = 256
num_real_img = 0
sum_mae_list = []
name_list = []

tmp_patinet = '210442'
tmp_patinet_num = 0
Test_Dir = 'Test_Img'
for generated_dir in generated_dirs:
    # print(generated_dir)

    if len(os.listdir(target_img_path))<5:
        continue

    for img_name in os.listdir(target_img_path):
        # print(img_name)
        # print(generated_dir, img_name, os.path.join(target_img_path, img_name))
        try:
            generated_value = cv2.resize(cv2.imread(os.path.join(generated_dir, img_name), cv2.IMREAD_GRAYSCALE),
                                     (img_size, img_size))
        except:
            break


        '''
        real_value = cv2.resize(cv2.imread(os.path.join(target_img_path, img_name.replace('png', 'jpg')), cv2.IMREAD_GRAYSCALE),
                                (img_size, img_size))
        '''

        real_value = cv2.resize(
            cv2.imread(os.path.join(target_img_path, img_name), cv2.IMREAD_GRAYSCALE),
            (img_size, img_size))

        if type_mae_psnr == 'liver':
            # print(os.path.join(source_tumor_path, img_name))
            source_art = cv2.resize(cv2.imread(os.path.join(source_tumor_path, img_name), cv2.IMREAD_GRAYSCALE),
                                    (img_size, img_size)) / 255.0

            target_art = cv2.resize(cv2.imread(os.path.join(target_liver_path, img_name), cv2.IMREAD_GRAYSCALE),
                                    (img_size, img_size)) / 255.0

            real_value = real_value * source_art
            generated_value = generated_value * source_art

            # cv2.imwrite(os.path.join(Test_Dir, img_name), generated_value)

        # ssim psnr
        err = ssim(real_value, generated_value)
        sum_ssim = sum_ssim + err

        err_psnr = psnr(real_value, generated_value)
        sum_psnr = sum_psnr + err_psnr

        err_mae = mean_squared_error(real_value, generated_value)
        sum_mae = sum_mae + err_mae

        num_img += 1

        '''
        from math import log10
        my_psnr = 10 * log10(256*256/err_mae)
        sum_my_psnr = sum_my_psnr + my_psnr
        '''
    if num_img == 0:
        continue
    print(generated_dir, tmp_patinet_num, len(os.listdir(generated_dir)), 'sum_ssim, sum_mae, sum_psnr', sum_ssim/num_img, sum_mae/num_img, sum_psnr/num_img)

    sum_mae = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_my_psnr = 0
    num_img = 0
    tmp_patinet_num = 0



