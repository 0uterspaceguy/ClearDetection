import os 
import pickle
import argparse
from utils import parse_config, mkdir, rmdir
import matplotlib
import json
from shutil import copyfile as cp

from cleanlab.object_detection.filter import find_label_issues
from cleanlab.object_detection.rank import (
    get_label_quality_scores,
    issues_from_scores,
)
from cleanlab.object_detection.summary import visualize 

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args


def main(config_path):
    config = parse_config(config_path)

    class_names = config['Dataset']['names']
    dataset_path = config['Dataset']['dataset_path']

    images_path = os.path.join(dataset_path, 'images')

    with open(os.path.join('/workspace/results', 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(os.path.join('/workspace/results', 'predictions.pickle'), 'rb') as handle:
        predictions = pickle.load(handle)

    label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)

    issue_images_result = []
    issue_images_set = set([labels[issue_to_visualize]['image_name'].replace('sym_', '') for issue_to_visualize in label_issue_idx])

    if config["Correction"]["correct_issue_labels"]:
        corrected_path = os.path.join(config['Dataset']['dataset_path'], 'corrected_labels')
        src_images_path = os.path.join(config['Dataset']['dataset_path'], 'images')
        src_labels_path = os.path.join(config['Dataset']['dataset_path'], 'labels')

        rmdir(corrected_path)
        mkdir(corrected_path)

        for image_name in os.listdir(src_images_path):
            label_name = os.path.splitext(image_name)[0] + ".txt"
            dst_label_path = os.path.join(corrected_path, label_name)

            if image_name in issue_images_set:
                src_label_path = os.path.join("./yolo_labels", label_name)
            else:
                src_label_path = os.path.join(src_labels_path, label_name)

            cp(src_label_path, dst_label_path)
    
    for issue_to_visualize in label_issue_idx:
        label = labels[issue_to_visualize]
        image_name = label['image_name'].replace('sym_', '')

        issue_images_result.append({"image_name": image_name})

    with open('/workspace/results/result.json', 'w', encoding='utf-8') as result_file:
        json.dump(issue_images_result, result_file)


    if config['do_visualize']:
        for issue_to_visualize in label_issue_idx[:config['num_images_to_vis']]:

            label = labels[issue_to_visualize]
            prediction = predictions[issue_to_visualize]

            image_path = os.path.join(images_path, label['image_name'].replace('sym_', ''))

            visualize(image_path, label=label, prediction=prediction, class_names=class_names, overlay=False, save_path=f"./drawed/{label['image_name'].replace('sym_', '')}")
            matplotlib.pyplot.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)




        