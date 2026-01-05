import torch

from torch.utils.data import DataLoader
from pathlib import Path

from minimal_sam.models import MicroSAM, PicoSAM
from minimal_sam.utils.helper import benchmark
from minimal_sam import MinimalSamDataset


IMG_SIZE = 96

BASE_PATH = Path(__file__).parents[2]

DATASET_DIR    = BASE_PATH.joinpath("dataset")
CHECKPOINT_DIR = BASE_PATH.joinpath("checkpoints")

COCO_ANN_FILE  = DATASET_DIR.joinpath("annotations/instances_val2017.json")
LVIS_ANN_FILE  = DATASET_DIR.joinpath("annotations/lvis_v1_val.json")

COCO_FILTERED  = BASE_PATH.joinpath(f"configs/annotations/filtered_anns_{IMG_SIZE}x{IMG_SIZE}_val2017_coco.json")
LVIS_FILTERED  = BASE_PATH.joinpath(f"configs/annotations/filtered_anns_{IMG_SIZE}x{IMG_SIZE}_val2017_lvis.json")


def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define models
    micro_sam = MicroSAM()
    pico_sam  = PicoSAM()

    # load weights
    micro_sam.load_state_dict(torch.load(CHECKPOINT_DIR.joinpath("MicroSAM.pt"), map_location=device))
    pico_sam.load_state_dict(torch.load(CHECKPOINT_DIR.joinpath("PicoSAM_KD.pt"), map_location=device))

    # load validation dataset
    coco_dataset = MinimalSamDataset(IMG_SIZE, DATASET_DIR, COCO_ANN_FILE, COCO_FILTERED)
    coco_loader  = DataLoader(coco_dataset, batch_size=1, shuffle=False)

    # benchmark models
    print("Evaluating MicroSAM...")
    benchmark(micro_sam, input_size=(1, 3, IMG_SIZE, IMG_SIZE), dataloader=coco_loader, device=device)

    print("Evaluating PicoSAM-distilled...")
    benchmark(pico_sam, input_size=(1, 3, IMG_SIZE, IMG_SIZE), dataloader=coco_loader, device=device)

if __name__ == "__main__":
    compare_models()