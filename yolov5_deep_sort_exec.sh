#!/bin/bash

cd ../Yolov5_DeepSort_Pytorch
source venv/bin/activate
for var in `find ../signate_537/train_videos/*.mp4`
do
	echo $var
	python3 track.py --source $var --yolo_model ../yolov5/runs/train/exp6/weights/best.pt --img 640 --deep_sort_model resnet50 --save-vid --save-txt --project ../signate_537/inference_result --exist-ok --classes 0 1
done
deactivate
cd -
