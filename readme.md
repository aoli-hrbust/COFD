# The official repository for COFD-Net: Complete Object Feature Diffusion Network for Occluded Person Re-Identification.

# Installation
	pip install -r requirements.txt


# Prepare Datasets
	the datasets can be download from : https://pan.quark.cn/s/fffabba6df8e
	mkdir data
	Download the person datasets Market-1501. Then unzip it and rename it under the directory like
		data
		└── market1501
		    └── images ..

# Prepare ViT Pre-trained Models
	jx_vit_base_p16_224-80ecf9dd.pth
	the model can be download from : https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

# Training
	python train.py --config_file configs/Market/COFD_194.yml DATASETS.ROOT_DIR "('your dataset path')" MODEL.DEVICE_ID "('your device id')" OUTPUT_DIR "('your path of output')"
	or
	Bash baseline_market.sh	#Please configure the data path and output path.

# Evaluation
	python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"


	


