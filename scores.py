import argparse
import numpy as np
import imageio
from PIL import Image  # 用于图像大小调整
import os
import sys
import time

sys.path.append('./utils/')
from metrics import fast_hist
from rgb_ind_convertor import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='R3D',
                    help='define the benchmark')

parser.add_argument('--result_dir', type=str, default='./out',
                    help='define the storage folder of network prediction')


def evaluate_semantic(benchmark_path, result_dir, num_of_classes=11, need_merge_result=False,
                      im_downsample=False, gt_downsample=False):
    gt_paths = open(benchmark_path, 'r').read().splitlines()
    d_paths = [p.split('\t')[2] for p in gt_paths]  # 1 denote wall, 2 denote door, 3 denote room
    r_paths = [p.split('\t')[3] for p in gt_paths]  # 1 denote wall, 2 denote door, 3 denote room
    cw_paths = [p.split('\t')[-1] for p in gt_paths]  # last one denote close wall

    im_paths = [os.path.join(result_dir, os.path.basename(p)) for p in r_paths]

    if need_merge_result:
        im_paths = [os.path.join(result_dir + '/room', os.path.basename(p)) for p in r_paths]
        im_d_paths = [os.path.join(result_dir + '/door', os.path.basename(p)) for p in d_paths]
        im_cw_paths = [os.path.join(result_dir + '/close_wall', os.path.basename(p)) for p in cw_paths]

    n = len(im_paths)
    hist = np.zeros((num_of_classes, num_of_classes))

    for i in range(n):  # 使用range替代xrange
        # 使用imageio读取图像
        im = imageio.v2.imread(im_paths[i])
        if need_merge_result:
            im_d = imageio.v2.imread(im_d_paths[i], mode='L')
            im_cw = imageio.v2.imread(im_cw_paths[i], mode='L')

        # 读取ground truth
        cw = imageio.v2.imread(cw_paths[i], mode='L')
        dd = imageio.v2.imread(d_paths[i], mode='L')
        rr = imageio.v2.imread(r_paths[i])

        # 图像大小调整
        def resize_img(img, size, is_gray=False):
            if is_gray and len(img.shape) == 2:
                img = Image.fromarray(img).resize(size, Image.BILINEAR)
            else:
                img = Image.fromarray(img).resize(size, Image.BILINEAR)
            return np.array(img)

        size = (512, 512)
        if im_downsample:
            im = resize_img(im, size)
            if need_merge_result:
                im_d = resize_img(im_d, size, is_gray=True)
                im_cw = resize_img(im_cw, size, is_gray=True)
                im_d = im_d / 255.0
                im_cw = im_cw / 255.0

        if gt_downsample:
            cw = resize_img(cw, size, is_gray=True)
            dd = resize_img(dd, size, is_gray=True)
            rr = resize_img(rr, size)

        # 归一化
        cw = cw / 255.0
        dd = dd / 255.0

        # 转换索引图像
        im_ind = rgb2ind(im, color_map=floorplan_fuse_map)
        if im_ind.sum() == 0:
            im_ind = rgb2ind(im + 1)

        rr_ind = rgb2ind(rr, color_map=floorplan_fuse_map)
        if rr_ind.sum() == 0:
            rr_ind = rgb2ind(rr + 1)

        if need_merge_result:
            im_d = (im_d > 0.5).astype(np.uint8)
            im_cw = (im_cw > 0.5).astype(np.uint8)
            im_ind[im_cw == 1] = 10
            im_ind[im_d == 1] = 9

        # 合并标签
        cw = (cw > 0.5).astype(np.uint8)
        dd = (dd > 0.5).astype(np.uint8)
        rr_ind[cw == 1] = 10
        rr_ind[dd == 1] = 9

        name = os.path.basename(im_paths[i])
        r_name = os.path.basename(r_paths[i])

        print(f'Evaluating {name}(im) <=> {r_name}(gt)...')

        hist += fast_hist(im_ind.flatten(), rr_ind.flatten(), num_of_classes)

    print('*' * 60)
    # 总体准确率
    acc = np.diag(hist).sum() / hist.sum()
    print(f'overall accuracy {acc:.4f}')

    # 每类准确率
    acc = np.diag(hist) / (hist.sum(1) + 1e-6)
    print(f'room-type: mean accuracy {np.nanmean(acc[:7]):.4f}, '
          f'room-type+bd: mean accuracy {(np.nansum(acc[:7]) + np.nansum(acc[-2:])) / 9.:.4f}')

    for t in range(0, acc.shape[0]):
        if t not in [7, 8]:
            print(f'room type {t}th, accuracy = {acc[t]:.4f}')

    print('*' * 60)
    # 每类IoU
    iu = np.diag(hist) / (hist.sum(1) + 1e-6 + hist.sum(0) - np.diag(hist))
    print(f'room-type: mean IoU {np.nanmean(iu[:7]):.4f}, '
          f'room-type+bd: mean IoU {(np.nansum(iu[:7]) + np.nansum(iu[-2:])) / 9.:.4f}')

    for t in range(iu.shape[0]):
        if t not in [7, 8]:  # 忽略类别7和8
            print(f'room type {t}th, IoU = {iu[t]:.4f}')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.dataset.lower() == 'r2v':
        benchmark_path = './dataset/r2v_test.txt'
    else:
        benchmark_path = './dataset/r3d_test.txt'

    result_dir = FLAGS.result_dir

    tic = time.time()
    evaluate_semantic(
        benchmark_path,
        result_dir,
        need_merge_result=False,
        im_downsample=False,
        gt_downsample=True
    )

    print("*" * 60)
    print(f"Evaluate time: {time.time() - tic:.2f} sec")