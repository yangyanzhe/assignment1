import os
import argparse
import numpy as np

import sys
sys.path.append('common')
from BVH import load, save
from Visualize import visualize_anim


def main(src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    anim, joint_names, frame_time, order = load(src_path)

    # TODO: edit anim

    # Save anim to bvh file
    save(os.path.join(dst_path, 'output.bvh'), anim, joint_names, frame_time, order)
    
    # Visualize anim and save as mp4
    name = os.path.basename(src_path).split('.')[0]
    visualize_anim(anim, title=None, img_dir=os.path.join(dst_path, name), multi_view=False,
                   video_path=os.path.join(dst_path, name + ".mp4"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='data/Samba_Dancing.bvh')
    parser.add_argument('--dst_path', type=str, default='output/')
    args = parser.parse_args()
    main(**vars(args))
