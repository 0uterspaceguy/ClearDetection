echo Split dataset into folds...
python make_folds.py -c configs/config.yaml

echo Start training loops...
python train_models.py -c configs/config.yaml

echo Inference...
python predict.py -c configs/config.yaml

echo Analyze...
python analyze.py -c configs/config.yaml

rm -R runs data/* datasets/*/fold*  datasets/*/*.pickle

