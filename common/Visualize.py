"""
Visualize.py
Author: Yanzhe Yang
Time: 1/29/2019
Function: Visualize motions and save to mp4 files
"""

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os
import math
import subprocess
from PIL import Image, ImageDraw, ImageFont
from Animation import positions_global


def add_line(p0, p1, X, Y, Z):
    X.append(p0[0])
    X.append(p1[0])
    Y.append(p0[1])
    Y.append(p1[1])
    Z.append(p0[2])
    Z.append(p1[2])


def add_arrows(p0, p1, X, Y, Z):
    add_line(p0, p1, X, Y, Z)

    vec = p1 - p0
    if vec[0] > vec[2] and vec[0] > vec[1]:
        norm = np.array([vec[1], -vec[0], vec[2]])
    else:
        norm = np.array([vec[0], -vec[2], vec[1]])

    p2 = 0.8 * vec + p0
    p3 = p2 + 0.1 * norm
    p4 = p2 - 0.1 * norm

    add_line(p1, p3, X, Y, Z)
    add_line(p1, p4, X, Y, Z)


def plot_line(ax, start, end, color):
    X = list()
    Y = list()
    Z = list()
    add_line(start, end, X, Y, Z)
    ax.plot(X, -np.array(Z), Y, color=color)


def plot_arrows(ax, start, end, color):
    X = list()
    Y = list()
    Z = list()
    add_arrows(start, end, X, Y, Z)
    ax.plot(X, -np.array(Z), Y, color=color)


def get_bone_lines(data):
    poses = data['pos']
    parents = data['structure']

    num_frame = len(poses)
    num_bone = len(parents)

    lines_frames = list()
    for frame_id in range(num_frame):
        lines_frame = list()
        for bone_id in range(num_bone):
            if parents[bone_id] == -1:
                continue

            loc = np.array([poses[frame_id, parents[bone_id]], poses[frame_id, bone_id]])
            lines_frame.append(np.transpose(loc))

        lines_frames.append(lines_frame)
    return lines_frames


def get_bone_lines_all(data):
    num_display = len(data)
    lines = list()  # (num_display, num_frame, num_bones)

    for display_id in range(num_display):
        lines_frames = get_bone_lines(data[display_id])
        lines.append(lines_frames)

    return np.array(lines)


def plot_axes(ax, min_X, max_X, min_Y, max_Y, min_Z, max_Z):
    """
    :param ax: matplot axis
    :param min_X, max_X, min_Y, max_Y, min_Z, max_Z: value ranges in plot
    :param x, y, z here are in matplot coordinate system, Z-up
    :return:
    """
    # x axis
    # setting x values in mplot
    arrow_length = abs(max_X - min_X) / 20
    X = list()
    Y = list()
    Z = list()
    add_line((min_X, max_Y, min_Z), (max_X, max_Y, min_Z), X, Y, Z)
    add_line((max_X, max_Y, min_Z), (max_X - arrow_length, max_Y - arrow_length, min_Z), X, Y, Z)
    add_line((max_X, max_Y, min_Z), (max_X - arrow_length, max_Y + arrow_length, min_Z), X, Y, Z)
    ax.plot(X, Y, Z, color='r')

    # plot y axis in Y-up world
    # i.e. setting z values in mplot
    X = list()
    Y = list()
    Z = list()
    arrow_length = abs(max_Z - min_Z) / 20
    add_line((min_X, max_Y, min_Z), (min_X, max_Y, max_Z), X, Y, Z)
    add_line((min_X, max_Y, max_Z), (min_X + arrow_length, max_Y, max_Z - arrow_length), X, Y, Z)
    add_line((min_X, max_Y, max_Z), (min_X - arrow_length, max_Y, max_Z - arrow_length), X, Y, Z)
    ax.plot(X, Y, Z, color='g')

    # z axis in Y-up world
    # i.e. setting -y values in mplot
    X = list()
    Y = list()
    Z = list()
    arrow_length = abs(max_Y - min_Y) / 20
    add_line((min_X, max_Y, min_Z), (min_X, min_Y, min_Z), X, Y, Z)
    add_line((min_X, min_Y, min_Z), (min_X + arrow_length, min_Y + arrow_length, min_Z), X, Y, Z)
    add_line((min_X, min_Y, min_Z), (min_X - arrow_length, min_Y + arrow_length, min_Z), X, Y, Z)
    ax.plot(X, Y, Z, color='b')


def plot_multi(data, img_dir, video_path, audio_path=None, interactive=False, multi_view=False,
               savefig=True, height=5, width=6):
    """
    :param data: an array of plotting data
           each item in the array is an dictionary with 'name', 'pos', 'structure'
           'name': string
           'pos': (F, J, 3)
           'structure': (J, )

    :param img_dir: the directory for saving save plotted image
    :param video_path: the file path for saving synthesized videos, videos will not be synthesized if len(video_path)==0
    :param audio_path: the file path for audio to be synthesized, default: None
    :param interactive: whether show images interactively, default: False
    :param multi_view: whether generatig results in different views
    :param savefig: whether save figures, default: True
    :param height: figure height
    :param width: figure width
    :return:
    """
    if len(data) <= 0:
        return

    num_display = len(data)
    num_rows = int(math.ceil(num_display / 2.0))
    num_cols = min(num_display, 2)
    num_frame = len(data[0]['pos'])

    for i in range(num_display):
        if num_frame > len(data[i]['pos']):
            num_frame = len(data[i]['pos'])

    lines = get_bone_lines_all(data)
    from matplotlib import rcParams
    rcParams['figure.figsize'] = width, height

    view_names = ['3d', 'side', 'front', 'top']
    view_params = [(10, -19), (0, 0), (0, -90), (90, 0)]
    if multi_view is False:
        view_names = view_names[:1]
        view_params = view_params[:1]

    for view_id in range(len(view_names)):
        view_name = view_names[view_id]

        for frame_id in range(num_frame):
            fig = plt.figure()
            colors = ['g', 'r', 'b']
            for index in range(num_display):
                ax = fig.add_subplot(num_rows, num_cols, index + 1, projection='3d')
                ax.view_init(elev=view_params[view_id][0], azim=view_params[view_id][1])
                if index >= len(lines):
                    print('index > len(lines): %d > %d' % (index, len(lines)))
                    continue

                if frame_id >= len(lines[index]):
                    print('frame_id > len(lines[index): %d > %d, index=%d' % (frame_id, len(lines[index]), index))
                    continue
                bone_id = 0

                for [X, Y, Z] in lines[index][frame_id]:
                    ax.plot(X, -Z, Y, color=colors[index % num_cols])
                    bone_id += 1

                min_Z = 0
                max_Z = 160
                min_X = -80
                max_X = 80
                min_Y = -240
                max_Y = 80

                # Scaling axes
                x_scale = (max_X - min_X) / (max_Z - min_Z)
                y_scale = (max_Y - min_Y) / (max_Z - min_Z)
                z_scale = 1

                scale = np.diag([x_scale, y_scale, z_scale, 1.0])
                scale = scale * (1.0 / scale.max())
                scale[3, 3] = 1.0

                def short_proj():
                    return np.dot(Axes3D.get_proj(ax), scale)

                ax.get_proj = short_proj
                ax.set_xlim3d(min_X, max_X)
                ax.set_ylim3d(min_Y, max_Y)
                ax.set_zlim3d(min_Z, max_Z)
                if data[index]['name'] is not None:
                    ax.set_title(data[index]['name'])
                plot_axes(ax, min_X, max_X, min_Y, max_Y, min_Z, max_Z)

            plt.suptitle('')
            os.makedirs(img_dir + '_' + view_name, exist_ok=True)
            filename = os.path.join(img_dir + '_' + view_name, '%d.png' % frame_id)
            if savefig:
                plt.savefig(filename)
                fig.clear()
                plt.close()
            elif interactive:
                plt.show()
                fig.clear()
                plt.close()

        if len(img_dir) > 0 and len(video_path) > 0:
            if not os.path.exists(os.path.dirname(video_path)):
                os.makedirs(os.path.dirname(video_path))
            image_path = img_dir + '_' + view_name + '/' + '%d.png'
            if audio_path is not None:
                subprocess.call(['ffmpeg', '-r', str(30), '-i', image_path, '-i', audio_path, '-vcodec', 'libx264',
                                 '-pix_fmt', 'yuv420p', video_path[:-4] + '_' + view_name + '.mp4',
                                 '-loglevel', 'panic'])
            else:
                subprocess.call(['ffmpeg', '-r', str(30), '-i', image_path, '-vcodec', 'libx264',
                                 '-pix_fmt', 'yuv420p', video_path[:-4] + '_' + view_name + '.mp4',
                                 '-loglevel', 'panic'])


def visualize_anim(anim, title, img_dir, video_path, multi_view=False, interactive=False, savefig=True, height=5, width=6):
    data = [{
        'pos': positions_global(anim),
        'structure': anim.parents,
        'name': title
    }]
    plot_multi(data, img_dir, video_path, multi_view=multi_view,
               interactive=interactive, savefig=savefig, height=height, width=width)
