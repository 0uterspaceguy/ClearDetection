import os
import cv2
from copy import deepcopy
from detector import Detector
import numpy as np
import pickle
import argparse
from utils import parse_config

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args

def main(config_path):
    config = parse_config(config_path)

    labels = []
    predictions = []

    for fold_idx in range(config['Dataset']['num_folds']):
        model_path = os.path.join("runs", "detect", f"fold_{fold_idx}", "weights", "best.onnx")

        model = Detector(model_path,
                        conf_thres=config['Inference']['conf_thres'], 
                        iou_thres=config['Inference']['iou_thres'])
            
        dataset_path = os.path.join(config['Dataset']['dataset_path'], f"fold_{fold_idx}")

        images_path = os.path.join(dataset_path, 'test', 'images')
        labels_path = os.path.join(dataset_path, 'test', 'labels')

        num_classes = len(config["Dataset"]["names"])

        for image_name in os.listdir(images_path):
            image_path = os.path.join(images_path, image_name)

            orig_image = cv2.imread(image_path)
            orig_h, orig_w = orig_image.shape[:-1]

            label_name = os.path.splitext(image_name)[0]+'.txt'
            label_path = os.path.join(labels_path, label_name)

            label_template = {
                "bboxes":[],
                "labels":[],
                "image_name": image_name,
            }

            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as file:
                    lines = [line[:-1] for line in file.readlines()]

                for line in lines:
                    if len(line.split()) == 5:
                        label_id, xn, yn, wn, hn = [float(var) for var in line.split()]
                        w = wn * orig_w
                        h = hn * orig_h
                        x1 = xn * orig_w - w / 2
                        y1 = yn * orig_h - h / 2

                        x2 = x1 + w
                        y2 = y1 + h
                        
                    else:
                        line = [float(var) for var in line.split()]
                        label_id = line[0]

                        xs = [var*orig_w for var in line[1::2]]
                        ys = [var*orig_h for var in line[2::2]] 

                        x1 = min(xs)
                        y1 = min(ys)
                        x2 = max(xs)
                        y2 = max(ys)

                    label_template['bboxes'].append([x1,y1,x2,y2])
                    label_template['labels'].append(label_id)
            
            label_template['bboxes'] = np.array(label_template['bboxes']) if len(label_template['bboxes']) else np.empty((0,4))
            label_template['labels'] = np.array(label_template['labels']) if len(label_template['labels']) else np.empty((0,4))

            labels.append(deepcopy(label_template))

            prediction_template = [[] for _ in range(num_classes)]

            boxes, scores, class_ids = model(orig_image)

            for box, score, class_id in zip(boxes, scores, class_ids):
                prediction_template[class_id].append(list(box) + [score])
            
            for class_id in range(num_classes):
                prediction_template[class_id] = np.array(prediction_template[class_id], dtype=np.float32)if len(prediction_template[class_id]) else np.empty((0, 5), dtype=np.float32)
                
            prediction_template = np.array(prediction_template, dtype=object)

            prediction_template = prediction_template if len(prediction_template) else np.empty((num_classes,0,5))
            predictions.append(prediction_template)

    with open(os.path.join('/workspace/results', 'labels.pickle'), 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join('/workspace/results', 'predictions.pickle'), 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)











            


