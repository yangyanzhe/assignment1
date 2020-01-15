import os
import argparse
import numpy as np

import sys

sys.path.append('common')
from ImportFBX import transform_format


def main(src_path, dst_path, root_name):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for root, dirs, files in os.walk(src_path):
        files.sort()
        for file in files:
            if file.endswith('.fbx'):
                transform_format(os.path.join(root, file), os.path.join(dst_path, file[:-4]+'.bvh'), root_name)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='data/fbx')
    parser.add_argument('--dst_path', type=str, default='data/')
    parser.add_argument('--root_name', type=str, default='pelvis')

    args = parser.parse_args()
    main(**vars(args))
