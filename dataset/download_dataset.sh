#!/bin/bash

CURL="curl -s -L -O"

if [ ! -d "train2017" ] || [ ! -d "val2017" ] || [ ! -d "annotations" ]; then
    echo "Downloading COCO 2017 dataset..."
    $CURL http://images.cocodataset.org/zips/val2017.zip &
    $CURL http://images.cocodataset.org/zips/train2017.zip &
    $CURL http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
    $CURL https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip &
    wait
    unzip val2017.zip -d . > /dev/null && rm val2017.zip &
    unzip train2017.zip -d . > /dev/null && rm train2017.zip &
    wait
    unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip 
    unzip lvis_v1_val.json.zip -d ./annotations/ && rm lvis_v1_val.json.zip 
    echo "COCO 2017 dataset downloaded and extracted."
else
    echo "COCO 2017 dataset already exists. Skipping download."
fi
