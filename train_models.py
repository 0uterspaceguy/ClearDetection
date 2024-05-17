from ultralytics import YOLO
import argparse
from utils import parse_config
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Training loop of yolo models')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args

def main(config_path):
    config = parse_config(config_path)
    data_path = config['Training']['data']

    for fold_idx in range(config['Dataset']['num_folds']):
        config = parse_config(config_path)

        model = YOLO(config['Training']['model']+'.yaml')

        model = YOLO(config['Training']['model']+'.pt')

        config['Training']['data'] = os.path.join(data_path, f"data_fold_{fold_idx}.yaml")
        results = model.train(name=f"fold_{fold_idx}", **config['Training'])

        # results = model.val()
        
        success = model.export(format='onnx')

if __name__ == "__main__":
    args = parse_args()
    main(args.config)

