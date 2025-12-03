#!/bin/bash

echo "Downloading COCO 2017 dataset ..."

mkdir -p dataset && cd dataset

curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip && rm val2017.zip
unzip train2017.zip && rm train2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip


echo "Downloading SAM2.1 checkpoints ..."

cd ../checkpoints && ./download_ckpts.sh && cd ..


echo "Setup python environment ..."

# check if uv is available
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing uv ..."
    pip install uv
else
    echo "uv is already installed."
fi

uv sync



