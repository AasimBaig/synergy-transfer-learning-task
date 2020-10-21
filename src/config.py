import os

DATA_DIR = "/home/aasim/synergy-ai-task/data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

BATCH_SIZE = 64
SEED = 42

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# no. of epochs
EPOCHS = 50

# using GPU
DEVICE = "cuda"

# number of worker set according to CPU
NUM_WORKERS = 12

# learning rate
LR = 1e-4
