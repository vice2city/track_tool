import os
import os.path as osp
import shutil

import mmengine
from tqdm import tqdm

in_folder = '/data1/zhuhongchun/datasets/SAT-MTB_ship/test/'
out_folder = '/data1/zhuhongchun/datasets/SAT-MTB_split/test/'


def get_gts(gt_list, length):
    result = [[]] * length
    for gt in gt_list:
        index = int(gt.strip().split(',')[0])
        result[index].append(gt)
    return result


def copy_gts(gt1, gt2):
    for gt in gt1:
        gt2.append(gt)


def create_dir(n):
    path = osp.join(out_folder, n)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'gt'), exist_ok=True)
    os.makedirs(osp.join(path, 'img'), exist_ok=True)


def split_video(index):
    sets = ['a', 'b']
    video_folder = osp.join(in_folder, index)
    seq_info = mmengine.list_from_file(f'{video_folder}/seqinfo.ini')
    gts = mmengine.list_from_file(f'{video_folder}/gt/gt.txt')
    seq_name = seq_info[1].strip().split('=')[1]
    num_img = int(seq_info[4].strip().split('=')[1])
    num_transmit = 30
    num_part = num_img / 2

    split_img = {}
    split_gt = {}
    for s in sets:
        split_img[s] = []
        split_gt[s] = []
        create_dir(seq_name + s)

    img_path = osp.join(video_folder, 'img')
    img_names = os.listdir(img_path)
    img_names.sort()
    classified_gt = get_gts(gts, len(img_names)+1)
    i = 0
    for img in img_names:
        img_index = int(img.split('.')[0])
        if i < num_part:
            split_img[sets[0]].append(img)
            copy_gts(classified_gt[img_index], split_gt[sets[0]])
        elif i < num_part + num_transmit:
            split_img[sets[0]].append(img)
            split_img[sets[1]].append(img)
            copy_gts(classified_gt[img_index], split_gt[sets[0]])
            copy_gts(classified_gt[img_index], split_gt[sets[1]])
        else:
            split_img[sets[1]].append(img)
            copy_gts(classified_gt[img_index], split_gt[sets[1]])
        i += 1

    for s in sets:
        name = seq_name + s
        out_path = osp.join(out_folder, name)
        out_img_path = osp.join(out_path, 'img')
        for img in split_img[s]:
            shutil.copy(osp.join(img_path, img), out_img_path)
        with open(osp.join(out_path, 'gt/gt.txt'), 'w') as file:
            file.write('\n'.join(split_gt[s]))
        with open(osp.join(out_path, 'seqinfo.ini'), 'w') as file:
            infos = seq_info.copy()
            infos[1] = f'name={name}'
            infos[4] = f'seqLength={len(split_img[s])}'
            file.write('\n'.join(infos))


if __name__ == '__main__':
    video_names = os.listdir(in_folder)
    for video in tqdm(video_names):
        split_video(video)
