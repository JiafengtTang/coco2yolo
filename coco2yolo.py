import json
import os
import random
import argparse
from shutil import copyfile, rmtree
from tqdm import tqdm
from coco_dataset_split import main

def parse_args():
    parser = argparse.ArgumentParser(description='coco2yolo and split')
    parser.add_argument('--images_path', type=str, default='../data/MyData/images3/', help='original images path')
    parser.add_argument('--json_path', type=str, default='../data/MyData/Annotations/train_coco.json')
    parser.add_argument('--data_path', type=str, default='../data/Data3/', help='split dataset path')
    parser.add_argument('--split_data', type=bool, default=True, help='split dataset')
    parser.add_argument('--split_ratio', type=float, default=[0.8, 0.2, 0], help='train:val:test')
    _args = parser.parse_args()
    return _args

def coco2yolo(args):
    fr = open(args.json_path, 'r', encoding='utf-8')
    f = json.loads(fr.read())
    img_path = os.listdir(args.images_path)
    coco_data_path = args.data_path + 'coco/'
    coco_data_images = coco_data_path + 'images/'
    coco_data_labels = coco_data_path + 'labels/'
    os.makedirs(coco_data_images, exist_ok=True)
    os.makedirs(coco_data_labels, exist_ok=True)
    bar = tqdm(range(len(img_path)), colour='GREEN')
    for i in bar:
        bar.set_description(desc='coco2yolo')
        bar.set_postfix(image=img_path[i])
        name, suffix = img_path[i].rsplit('.jpg')
        name = name.replace('.', '')
        old_name = args.images_path + img_path[i]
        new_name = coco_data_images + name + '.jpg'
        copyfile(old_name, new_name)
        name = f['images'][i]['file_name'][:-4]
        w1 = float(f['images'][i]['width'])
        h1 = float(f['images'][i]['height'])
        dw = 1. / w1
        dh = 1. / h1
        ft = open(coco_data_labels + f'{name}.txt', 'w')
        for j in range(len(f['annotations'])):
            data = []
            data.append(str(f['annotations'][j]['category_id'] - 1))  # maybe need category_id-1
            data.append(' ')
            bboxs = f['annotations'][j]['bbox']
            x = bboxs[0] + bboxs[2] / 2.0
            y = bboxs[1] + bboxs[3] / 2.0
            w = bboxs[2]
            h = bboxs[3]
            x = x * dw
            y = y * dh
            w = w * dw
            h = h * dh
            data.append(' '.join([str(x), str(y), str(w), str(h), '\n']))
            if f['annotations'][j]['image_id'] == f['images'][i]['id']:
                ft.writelines(data)
    return coco_data_path, coco_data_images, coco_data_labels

def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def split_yolo_data(args, image_dir, annotation_dir):
    list_imgs = os.listdir(image_dir)
    yolo_train_dir = args.data_path + 'yolo/train/'
    yolo_val_dir = args.data_path + 'yolo/val/'
    yolo_test_dir = args.data_path + 'yolo/test/'
    os.makedirs(yolo_train_dir, exist_ok=True)
    os.makedirs(yolo_val_dir, exist_ok=True)
    if args.split_ratio[2] != 0:
        os.makedirs(yolo_test_dir, exist_ok=True)

    train_images_dir = os.path.join(yolo_train_dir, "images/")
    os.makedirs(train_images_dir, exist_ok=True)
    clear_hidden_files(train_images_dir)
    train_labels_dir = os.path.join(yolo_train_dir, "labels/")
    os.makedirs(train_labels_dir, exist_ok=True)
    clear_hidden_files(train_labels_dir)

    val_images_dir = os.path.join(yolo_val_dir, "images/")
    os.makedirs(val_images_dir, exist_ok=True)
    clear_hidden_files(val_images_dir)
    val_labels_dir = os.path.join(yolo_val_dir, "labels/")
    os.makedirs(val_labels_dir, exist_ok=True)
    clear_hidden_files(val_labels_dir)

    if args.split_ratio[2] != 0:
        test_images_dir = os.path.join(yolo_test_dir, "images/")
        os.makedirs(test_images_dir, exist_ok=True)
        clear_hidden_files(test_images_dir)
        test_labels_dir = os.path.join(yolo_test_dir, "labels/")
        os.makedirs(test_labels_dir, exist_ok=True)
        clear_hidden_files(test_labels_dir)

    assert sum(args.split_ratio), 'Split ratio over the range!!!'
    loop = tqdm(range(0, len(list_imgs)), colour='GREEN')
    for i in loop:
        loop.set_description(desc='split yolo dataset ')
        loop.set_postfix(image=list_imgs[i])
        image_path = image_dir + list_imgs[i]
        label_name, _ = os.path.splitext(os.path.basename(image_path))
        label_path = annotation_dir + label_name + '.txt'
        prob = random.randint(1, 100)
        if prob < (args.split_ratio[0])*100:  # train dataset
            copyfile(image_path, train_images_dir + list_imgs[i])
            copyfile(label_path, train_labels_dir + label_name + '.txt')
        elif (args.split_ratio[0]) * 100 <= prob < (args.split_ratio[0] + args.split_ratio[1]) * 100:  # val dataset
            copyfile(image_path, val_images_dir + list_imgs[i])
            copyfile(label_path, val_labels_dir + label_name + '.txt')
        else:  # test dataset
            if args.split_ratio[2] != 0:
                copyfile(image_path, test_images_dir + list_imgs[i])
                copyfile(label_path, test_labels_dir + label_name + '.txt')

def coco_dataset_split(args):
    list_image = os.listdir(args.images_path)
    with open(args.json_path, 'r', encoding='utf-8') as fj:
        label = json.load(fj)

    ann_path = args.data_path + 'coco/' + 'annotation/'
    train_image_path = args.data_path + 'coco/' + 'train2017/'
    val_image_path = args.data_path + 'coco/' + 'val2017/'
    test_image_path = args.data_path + 'coco/' + 'test2017/'

    os.makedirs(ann_path, exist_ok=True)
    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(val_image_path, exist_ok=True)
    if args.split_ratio[2] != 0:
        os.makedirs(test_image_path, exist_ok=True)

    train_images = []
    train_ann = []
    val_images = []
    val_ann = []
    test_images = []
    test_ann = []
    loop = tqdm(range(0, len(list_image)), colour='GREEN', desc='split coco dataset')
    for i in loop:
        loop.set_postfix(image=list_image[i])
        img_path = args.images_path + list_image[i]
        prob = random.randint(1, 100)
        if prob < (args.split_ratio[0])*100:  # train dataset
            copyfile(img_path, train_image_path + list_image[i])
            train_images.append(label['images'][i])
            for j in range(len(label['annotations'])):
                if label['annotations'][j]['image_id'] == label['images'][i]['id']:
                    train_ann.append(label['annotations'][j])
        elif (args.split_ratio[0])*100 <= prob < (args.split_ratio[0]+args.split_ratio[1])*100:  # val dataset
            copyfile(img_path, val_image_path + list_image[i])
            val_images.append(label['images'][i])
            for j in range(len(label['annotations'])):
                if label['annotations'][j]['image_id'] == label['images'][i]['id']:
                    val_ann.append(label['annotations'][j])
        else:  # test dataset
            if args.split_ratio[2] != 0:
                copyfile(img_path, test_image_path + list_image[i])
                test_images.append(label['images'][i])
                for j in range(len(label['annotations'])):
                    if label['annotations'][j]['image_id'] == label['images'][i]['id']:
                        test_ann.append(label['annotations'][j])

    train = {'images': train_images,
             'annotations': train_ann,
             'licenses': label['licenses'],
             'categories': label['categories']}
    val = {'images': val_images,
           'annotations': val_ann,
           'licenses': label['licenses'],
           'categories': label['categories']}

    ann_train = ann_path + 'instance_train2017.json'
    ann_val = ann_path + 'instance_val2017.json'
    with open(ann_train, 'w') as t:
        json.dump(train, t, indent=2, ensure_ascii=False)
    with open(ann_val, 'w') as v:
        json.dump(val, v, indent=2, ensure_ascii=False)

    if args.split_ratio[2] != 0:
        ann_test = ann_path + 'instance_test2017.json'
        test = {'images': val_images,
                'annotations': test_ann,
                'licenses': label['licenses'],
                'categories': label['categories']}
        with open(ann_test, 'w') as te:
            json.dump(test, te, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    arg = parse_args()
    coco_data_path, image_dir, annotation_dir = coco2yolo(arg)
    if arg.split_data:
        split_yolo_data(arg, image_dir, annotation_dir)
        rmtree(coco_data_path)
        coco_dataset_split(arg)
    # rmtree(coco_data_path)
    print('Finished!!!')