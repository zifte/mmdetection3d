import glob
from os import path as osp

import numpy as np
import pandas as pd

# 把 S3DIS 一个房间（room）的所有实例 txt 文件读出来 → 合并成一个房间级别的点云 → 保存为 point.npy / sem_label.npy / ins_label.npy。

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# __file__：当前这个 indoor3d_util.py 文件的路径
# osp.abspath(__file__)：拿到绝对路径
# osp.dirname(...)：取上一级目录，即当前脚本所在的目录

BASE_DIR = osp.dirname(osp.abspath(__file__)) # path of s3dis

class_names = [
    x.rstrip() for x in open(osp.join(BASE_DIR, 'meta_data/class_names.txt')) # 将class_name.txt中的所有class形成列表
]

# {
#     'ceiling': 0,
#     'floor': 1,
#     'wall': 2,
#     ...
#     'clutter': 12
# }
class2label = {one_class: i for i, one_class in enumerate(class_names)}

# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO POINTS, SEM_LABEL AND INS_LABEL FILES
# -----------------------------------------------------------------------------


# 输入：
# anno_path：某一个房间的 Annotations 目录
# 例如：Area_1/office_2/Annotations

# 输出整个房间的点云
# out_filename：输出文件的前缀
# 例如：s3dis_data/Area_1_office_2

# 功能：
# 把这个房间里面所有实例（每个实例一个 txt 文件）全部读出来合并，生成该房间的：
#  - 点云数据：*_point.npy
#  - 语义标签：*_sem_label.npy
#  - 实例标签：*_ins_label.npy
# 注意：会把 XYZ 整体平移，使得房间中最小的坐标对齐到原点 (0,0,0)。

def export(anno_path, out_filename): # anno_path = 
    """Convert original dataset files to points, instance mask and semantic
    mask files. We aggregated all the points from each instance in the room.

    Args:
        anno_path (str): path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename (str): path to save collected points and labels
        file_format (str): txt or numpy, determines what file format to save.

    Note:
        the points are shifted before save, the most negative point is now
            at origin.
    """
    points_list = []
    ins_idx = 1  # instance ids should be indexed from 1, so 0 is unannotated

    # glob 是 文件路径模式匹配工具，核心作用是：根据你指定的「通配符模式」，自动查找符合条件的所有文件路径，避免手动遍历目录。
    for f in glob.glob(osp.join(anno_path, '*.txt')):
        one_class = osp.basename(f).split('_')[0] # osp.basename(f)：拿到文件名，例如 "ceiling_1.txt"; .split('_')[0]：取下划线前面的部分，例如 "ceiling"
        if one_class not in class_names:  # some rooms have 'staris' class # 如果这个名字不在 class_names 里（比如原始数据里有拼错的 "staris"），就统一归到 'clutter' 类。
            one_class = 'clutter' # 这一步就是从文件名中提取语义类别（ceiling, wall, floor 等）
        points = pd.read_csv(f, header=None, sep=' ').to_numpy() # 读取这个实例 txt 文件里所有的点,转成 numpy 数组 → points shape ~ (Mi, 9)（Mi 是这个实例的点数）
        labels = np.ones((points.shape[0], 1)) * class2label[one_class] # points.shape[0]：该实例有多少个点; 生成 (N, 1) 的数组; 每一行都是同一个语义类别 ID
        ins_labels = np.ones((points.shape[0], 1)) * ins_idx # 生成 (N, 1) 的数组，每一行的值都是当前实例的 ID（比如 1、2、3…）
        ins_idx += 1
        # points 是 (N, 9)（XYZRGB + Normal）
        # labels 是 (N, 1)（语义类别）
        # ins_labels 是 (N, 1)（实例 ID）
        # np.concatenate([...], 1) 沿列的方向拼接 → (N, 11) 或 (N, 8)，
        # 每处理完一个实例，房间的所有点被逐步累积到 points_list 当中。
        points_list.append(np.concatenate([points, labels, ins_labels], 1))
        
    # 目前已经获取了整个房间的点云
    # points_list = [
    # (12000, 8),   # ceiling instance
    # (20000, 8),   # wall instance
    # (3000, 8),    # chair instance
    # (4500, 8)     # table instance
    # ]
    
    # 把不同实例的点在“行方向（axis=0）”拼接在一起
    # 形成整个房间的点云集合：
    # data_label.shape = (N_total, 8)
    data_label = np.concatenate(points_list, 0)  # [N, 8], (pts, rgb, sem, ins)
    xyz_min = np.amin(data_label, axis=0)[0:3] # 坐标最小值
    data_label[:, 0:3] -= xyz_min # 把点云整体平移，使最小点在原点

    # 输出整个房间的点云
    np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int64))
    np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int64))
