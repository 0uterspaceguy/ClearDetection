import os 
import pickle
import argparse
from utils import parse_config
import matplotlib
import json
from tqdm import tqdm

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

    with open(os.path.join(dataset_path, 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(os.path.join(dataset_path, 'predictions.pickle'), 'rb') as handle:
        predictions = pickle.load(handle)

    label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)
    print("Num issue samples:", len(label_issue_idx))

    # scores = get_label_quality_scores(labels, predictions)
    # print(scores)

    # issue_idx = issues_from_scores(scores, threshold=0.1)  # lower threshold will return fewer (but more confident) label issues
    # print(issue_idx)

    issue_images = []

    for issue_to_visualize in label_issue_idx:
        label = labels[issue_to_visualize]
        image_name = label['image_name'].replace('sym_', '')

        issue_images.append({"image_name": image_name})

    with open('./results/result.json', 'w', encoding='utf-8') as result_file:
        json.dump(issue_images, result_file)


    if config['do_visualize']:
        for issue_to_visualize in tqdm(label_issue_idx[:config['num_images_to_vis']], desc="Visualizing"):

            label = labels[issue_to_visualize]
            prediction = predictions[issue_to_visualize]

            image_path = os.path.join(images_path, label['image_name'].replace('sym_', ''))

            visualize(image_path, label=label, prediction=prediction, class_names=class_names, overlay=False, save_path=f"./drawed/{label['image_name'].replace('sym_', '')}")
            matplotlib.pyplot.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)




        