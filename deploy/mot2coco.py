# Copyright (c) OpenMMLab. All rights reserved.
# This script converts MOT labels into COCO style.
# Official website of the MOT dataset: https://motchallenge.net/
#
# Label format of MOT dataset:
#   GTs:
#       <frame_id> # starts from 1 but COCO style starts from 0,
#       <instance_id>, <x1>, <y1>, <w>, <h>,
#       <conf> # conf is annotated as 0 if the object is ignored,
#       <class_id>, <visibility>
#
#   DETs and Results:
#       <frame_id>, <instance_id>, <x1>, <y1>, <w>, <h>, <conf>,
#       <x>, <y>, <z> # for 3D objects

import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import numpy as np
from tqdm import tqdm

# Classes in MOT:
CLASSES = [
    dict(id=1, name='ship')
]


def parse_gts(gts, is_mot15):
    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        if is_mot15:
            conf = 1.
            category_id = 1
            visibility = 1.
        else:
            conf = float(gt[6])
            category_id = int(gt[7])
            visibility = float(gt[8])
        anns = dict(
            category_id=category_id,
            bbox=bbox,
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf)
        outputs[frame_id].append(anns)
    return outputs



def get_annotation(input_path, output, data_list, isIR=False):
    if not osp.isdir(output):
        os.makedirs(output)

    vid_id, img_id, ann_id = 1, 1, 1
    ins_id = 0
    print(f'Converting test set to COCO format')
    in_folder = osp.join(input_path, 'test')
    out_file = osp.join(output, 'test_cocoformat.json')
    outputs = defaultdict(list)
    outputs['categories'] = CLASSES
    video_names = os.listdir(in_folder)
    for video_name in tqdm(video_names):
        if video_name not in data_list:
            continue
        parse_gt = True
        ins_maps = dict()
        # load video infos
        video_folder = osp.join(in_folder, video_name)
        infos = mmengine.list_from_file(f'{video_folder}/seqinfo.ini')
        # video-level infos
        assert video_name == infos[1].strip().split('=')[1]
        if isIR:
            img_folder = 'img3'
        else:
            img_folder = infos[2].strip().split('=')[1]
        img_names = os.listdir(f'{video_folder}/{img_folder}')
        img_names = sorted(img_names)
        fps = int(infos[3].strip().split('=')[1])
        num_imgs = int(infos[4].strip().split('=')[1])
        assert num_imgs == len(img_names)
        width = int(infos[5].strip().split('=')[1])
        height = int(infos[6].strip().split('=')[1])
        video = dict(
            id=vid_id,
            name=video_name,
            fps=fps,
            width=width,
            height=height)
        # parse annotations
        if parse_gt:
            gts = mmengine.list_from_file(f'{video_folder}/gt/gt.txt')
            if 'MOT15' in video_folder:
                img2gts = parse_gts(gts, True)
            else:
                img2gts = parse_gts(gts, False)
        # image and box level infos
        for frame_id, name in enumerate(img_names):
            img_name = osp.join(video_name, img_folder, name)
            mot_frame_id = int(name.split('.')[0])
            image = dict(
                id=img_id,
                video_id=vid_id,
                file_name=img_name,
                height=height,
                width=width,
                frame_id=frame_id,
                mot_frame_id=mot_frame_id)
            if parse_gt:
                gts = img2gts[mot_frame_id]
                for gt in gts:
                    gt.update(id=ann_id, image_id=img_id)
                    mot_ins_id = gt['mot_instance_id']
                    if mot_ins_id in ins_maps:
                        gt['instance_id'] = ins_maps[mot_ins_id]
                    else:
                        gt['instance_id'] = ins_id
                        ins_maps[mot_ins_id] = ins_id
                        ins_id += 1
                    outputs['annotations'].append(gt)
                    ann_id += 1
            outputs['images'].append(image)
            img_id += 1
        outputs['videos'].append(video)
        vid_id += 1
        outputs['num_instances'] = ins_id
    print(f'test has {ins_id} instances.')
    mmengine.dump(outputs, out_file)
    print(f'Done! Saved as {out_file}')
