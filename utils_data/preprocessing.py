import argparse
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import pandas as pd
import os
import sys
import json
import pybboxes as pbx
from collections import defaultdict
import shutil
from random import shuffle
from pathlib import Path, PurePath

import logging
logging.basicConfig()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class RtsdDataset:

    def __init__(self, input_path, output_path):
        self.input_path = ROOT / input_path
        self.output_path = ROOT / output_path
        self.train_annotations_path = Path.joinpath(self.input_path, 'train_anno.json')
        self.val_annotations_path = Path.joinpath(self.input_path, 'val_anno.json')
        self.images_path = Path.joinpath(self.input_path, 'rtsd-frames/rtsd-frames')
        self.classes = None
        self.n_classes = None
        self.train_dir, self.val_dir = self.make_dirs_train_val_data()

    def make_dirs_train_val_data(self):
        train_dir = Path.joinpath(self.output_path, 'data', 'train')
        val_dir = Path.joinpath(self.output_path, 'data', 'val')

        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)

        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)

        os.makedirs(train_dir)
        os.makedirs(val_dir)

        Path.joinpath(train_dir, 'images').mkdir()
        Path.joinpath(train_dir, 'labels').mkdir()
        Path.joinpath(val_dir, 'images').mkdir()
        Path.joinpath(val_dir, 'labels').mkdir()

        return train_dir, val_dir

    def data_preprocessing(self):
        train_labels_df = self.parse_input_annotations(self.train_annotations_path)
        val_labels_df = self.parse_input_annotations(self.val_annotations_path)

        train_labels_df = self.filter_anno_by_area(train_labels_df)
        val_labels_df = self.filter_anno_by_area(val_labels_df)

        top_cats_counts = train_labels_df['label'].value_counts()[0:60]
        top_cats = list(top_cats_counts.index)

        train_labels_df = self.filter_anno_by_label(train_labels_df, top_cats)
        val_labels_df = self.filter_anno_by_label(val_labels_df, top_cats)

        train_labels_df = self.remove_single_sign_annotation(train_labels_df)
        val_labels_df = self.remove_single_sign_annotation(val_labels_df)

        top_cats_counts = train_labels_df['label'].value_counts()
        classes = list(top_cats_counts.index[0:50])
        classes = list(set(val_labels_df['label'].unique()).intersection(classes))

        train_labels_df = self.filter_anno_by_label(train_labels_df, classes)
        val_labels_df = self.filter_anno_by_label(val_labels_df, classes)

        train_labels_df['class'] = train_labels_df['label'].apply(lambda x: classes.index(x))
        val_labels_df['class'] = val_labels_df['label'].apply(lambda x: classes.index(x))

        train_labels_df.reset_index(inplace=True)
        val_labels_df.reset_index(inplace=True)

        self.classes = classes
        self.n_classes = len(classes)

        train_img_labels = self.make_annotations(train_labels_df)
        val_img_labels = self.make_annotations(val_labels_df)

        self.write_annotations(train_img_labels, Path.joinpath(self.train_dir, 'labels'))
        self.write_annotations(val_img_labels, Path.joinpath(self.val_dir, 'labels'))

        train_files, val_files = list(train_img_labels.keys()), list(val_img_labels.keys())

        dst_train_path = Path.joinpath(self.train_dir, 'images')
        dst_val_path = Path.joinpath(self.val_dir, 'images')

        self.copy_imgs(self.images_path, dst_train_path, train_files)
        self.copy_imgs(self.images_path, dst_val_path, val_files)

        self.yaml_save(dst_train_path, dst_val_path)

    def parse_input_annotations(self, annotations_path):
        with open(annotations_path, "r") as read_content:
            content = json.load(read_content)
        labels_df = pd.DataFrame(content['annotations'])
        images_list = content['images']
        categories_list = content['categories']

        img_id2file_name = {}

        img_id2size = {}
        for img in images_list:
            img_id2file_name[img['id']] = img['file_name']
            img_id2size[img['id']] = (img['width'], img['height'])

        cat2name = {}
        for cat in categories_list:
            cat2name[cat['id']] = cat['name']

        labels_df['label'] = labels_df['category_id'].apply(lambda x: cat2name[x])
        labels_df['img_name_path'] = labels_df['image_id'].apply(lambda x: img_id2file_name[x])
        labels_df['img_name'] = labels_df['img_name_path'].apply(lambda x: x.split('/')[1])
        labels_df['width'] = labels_df['image_id'].apply(lambda x: img_id2size.get(x)[0])
        labels_df['height'] = labels_df['image_id'].apply(lambda x: img_id2size.get(x)[1])

        return labels_df

    def filter_anno_by_area(self, df, low_lim=850):
        except_img_list = list(set(df[df['area'] <= low_lim]['img_name']))
        return df[~df['img_name'].isin(except_img_list)]

    def filter_anno_by_label(self, df, interest_labels):
        return df[df['label'].isin(interest_labels)]

    def remove_single_sign_annotation(self, df):
        img_name_count = df['img_name'].value_counts()
        imgs_with_one_sign = list(img_name_count[img_name_count == 1].index)
        imgs_5_19_1 = list(df[df['label'] == '5_19_1']['img_name'].unique())
        imgs_2_1 = list(df[df['label'] == '2_1']['img_name'].unique())
        imgs_5_16 = list(df[df['label'] == '5_16']['img_name'].unique())
        imgs_5_15_2 = list(df[df['label'] == '5_15_2']['img_name'].unique())

        imgs_for_delete = []
        imgs_for_delete.extend(list(set(imgs_with_one_sign) & set(imgs_5_19_1)))
        imgs_for_delete.extend(list(set(imgs_with_one_sign) & set(imgs_2_1)))
        imgs_for_delete.extend(list(set(imgs_with_one_sign) & set(imgs_5_16)))
        imgs_for_delete.extend(list(set(imgs_with_one_sign) & set(imgs_5_15_2)))

        return df[~df['img_name'].isin(imgs_for_delete)]

    def make_annotations(self, labels_df):
        img_dict = defaultdict(list)

        for idx in tqdm(range(len(labels_df))):
            sample_label_list = []
            img_name = labels_df.loc[idx, 'img_name']
            x = labels_df.loc[idx, 'bbox'][0]
            y = labels_df.loc[idx, 'bbox'][1]
            w = labels_df.loc[idx, 'bbox'][2]
            h = labels_df.loc[idx, 'bbox'][3]
            class_num = labels_df.loc[idx, 'class']
            W, H = int(labels_df.loc[idx, 'width']), int(labels_df.loc[idx, 'height'])

            coco_bbox = (int(x), int(y), int(w), int(h))

            x_center, y_center, w, h = pbx.convert_bbox(coco_bbox, from_type="coco", to_type="yolo", image_size=(W, H))

            sample_label_list.append(str(class_num))
            sample_label_list.append(str(x_center))
            sample_label_list.append(str(y_center))
            sample_label_list.append(str(w))
            sample_label_list.append(str(h))
            line = ' '.join(sample_label_list)

            img_dict[img_name].append(line)
        return img_dict

    def write_annotations(self, img_dict, dst):

        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)

        for img_name, lines in img_dict.items():
            img_name = img_name.split('.')[0]
            with open(f'{dst}/{img_name}.txt', 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

    def copy_imgs(self, images_path, destination_path, files):
        for file_name in tqdm(files, desc=f'Copy images to {destination_path}'):
            src = Path.joinpath(images_path, file_name)
            shutil.copy(src, destination_path)

    def yaml_save(self, train_images_path, val_images_path):
        yaml_path = Path.joinpath(self.output_path, 'data', 'traffic_signs.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f'train: {train_images_path}\n')
            f.write(f'val: {val_images_path}\n')
            f.write(f'nc: {self.n_classes}\n')
            f.write(f"names: {self.classes}")


def main(opt):
    ds = RtsdDataset(opt.input_path, opt.output_path)
    ds.data_preprocessing()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=ROOT / './../input/rtsd', help='input path for data')
    parser.add_argument('--output_path', type=str, default=ROOT / './../datasets/rtsd', help='dataset path')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
