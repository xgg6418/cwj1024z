import argparse
import os
import sys
import glob
import numpy as np
import imageio

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import *

parser = argparse.ArgumentParser()

parser.add_argument('--result_dir', type=str, default='./out',
                    help='The folder that save network predictions.')


def post_process(input_dir, save_dir, merge=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # 使用makedirs更安全

    input_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    names = [os.path.basename(i) for i in input_paths]  # 更安全的路径处理
    out_paths = [os.path.join(save_dir, i) for i in names]

    n = len(input_paths)
    for i in range(n):  # 使用range代替xrange(Python3)
        # 使用imageio读取图像
        im = imageio.v2.imread(input_paths[i])
        if im.ndim == 2:  # 如果是灰度图，转换为RGB
            im = np.stack([im] * 3, axis=-1)
        elif im.shape[2] == 4:  # 如果是RGBA，转换为RGB
            im = im[:, :, :3]

        im_ind = rgb2ind(im, color_map=floorplan_fuse_map)
        # seperate image into room-seg & boundary-seg
        rm_ind = im_ind.copy()
        rm_ind[im_ind == 9] = 0
        rm_ind[im_ind == 10] = 0

        bd_ind = np.zeros(im_ind.shape, dtype=np.uint8)
        bd_ind[im_ind == 9] = 9
        bd_ind[im_ind == 10] = 10

        hard_c = (bd_ind > 0).astype(np.uint8)

        # region from room prediction it self
        rm_mask = np.zeros(rm_ind.shape)
        rm_mask[rm_ind > 0] = 1
        # region from close_wall line
        cw_mask = hard_c
        # refine close wall mask by filling the grap between bright line
        cw_mask = fill_break_line(cw_mask)

        fuse_mask = cw_mask + rm_mask
        fuse_mask[fuse_mask >= 1] = 255

        # refine fuse mask by filling the hole
        fuse_mask = flood_fill(fuse_mask)
        fuse_mask = fuse_mask // 255

        # one room one label
        new_rm_ind = refine_room_region(cw_mask, rm_ind)

        # ignore the background mislabeling
        new_rm_ind = fuse_mask * new_rm_ind

        print(f'Saving {i}th refined room prediciton to {out_paths[i]}')  # f-string
        if merge:
            new_rm_ind[bd_ind == 9] = 9
            new_rm_ind[bd_ind == 10] = 10
            rgb = ind2rgb(new_rm_ind, color_map=floorplan_fuse_map)
        else:
            rgb = ind2rgb(new_rm_ind)

        # 使用imageio保存图像
        imageio.v2.imwrite(out_paths[i], rgb)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    input_dir = FLAGS.result_dir
    save_dir = os.path.join(input_dir, 'post')

    post_process(input_dir, save_dir)