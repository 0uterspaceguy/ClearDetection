import yaml 
import os
from shutil import rmtree

def parse_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path: str) -> None:
    if os.path.exists(path):
        rmtree(path)
