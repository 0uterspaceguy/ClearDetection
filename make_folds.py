import os
import argparse
from utils import parse_config, mkdir, rmdir
from random import shuffle
from sklearn.model_selection import KFold
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into folds')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args

def main(config_path):
    config = parse_config(config_path)

    num_folds = config['Dataset']['num_folds']

    dataset_path = config['Dataset']['dataset_path']
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    images_names = os.listdir(images_path)
    shuffle(images_names)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(images_names)):
        fold_dir = os.path.join("/workspace/folds", f"fold_{fold_idx}")

        train_fold_dir = os.path.join(fold_dir, 'train')
        test_fold_dir = os.path.join(fold_dir, 'test')

        train_fold_images = os.path.join(train_fold_dir, 'images')
        test_fold_images = os.path.join(test_fold_dir, 'images')

        train_fold_labels = os.path.join(train_fold_dir, 'labels')
        test_fold_labels = os.path.join(test_fold_dir, 'labels')

        rmdir(fold_dir)
        mkdir(fold_dir)

        mkdir(train_fold_dir)
        mkdir(test_fold_dir)

        mkdir(train_fold_images)
        mkdir(test_fold_images)

        mkdir(train_fold_labels)
        mkdir(test_fold_labels)


        for train_id in train_index:
            image_name = images_names[train_id]
            label_name = os.path.splitext(image_name)[0]+".txt"

            src_image_path = os.path.join(images_path, image_name)
            sym_image_path = os.path.join(train_fold_images, f"sym_{image_name}")

            src_label_path = os.path.join(labels_path, label_name)
            sym_label_path = os.path.join(train_fold_labels, f"sym_{label_name}")

            if os.path.exists(src_image_path) and (not os.path.lexists(sym_image_path)):
                os.symlink(src_image_path, sym_image_path)

            if os.path.exists(src_label_path) and (not os.path.lexists(sym_label_path)):
                os.symlink(src_label_path, sym_label_path)

        for test_id in test_index:
            image_name = images_names[test_id]
            label_name = os.path.splitext(image_name)[0]+".txt"

            src_image_path = os.path.join(images_path, image_name)
            sym_image_path = os.path.join(test_fold_images, f"sym_{image_name}")

            src_label_path = os.path.join(labels_path, label_name)
            sym_label_path = os.path.join(test_fold_labels, f"sym_{label_name}")

            if os.path.exists(src_image_path) and (not os.path.lexists(sym_image_path)):
                os.symlink(src_image_path, sym_image_path)

            if os.path.exists(src_label_path) and (not os.path.lexists(sym_label_path)):
                os.symlink(src_label_path, sym_label_path)

        
        yolo_data = {
            'path': fold_dir,
            'train': "train/images", 
            'val': "test/images",  
            'test':  "test/images",
            'names': config['Dataset']['names']
        }

        with open(os.path.join("/workspace/data", f"data_fold_{fold_idx}.yaml"), 'w') as file:
            yaml.dump(yolo_data, file)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)