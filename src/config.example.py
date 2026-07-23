import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("CWRU_DATA_ROOT", "/absolute/path/to/Source_Datasets")

WINDOW_LENGTH = 1024
SAMPLE_RATE = 12000
OVERLAP_RANDOM = 0.5
# Keep segmentation fixed across split protocols so the comparison isolates
# only the split unit. Recording-level leakage is prevented by file-disjoint
# partitions, not by changing the window stride.
OVERLAP_RECORDING = 0.5
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

RANDOM_SEEDS = [42, 123, 456, 789, 1024]

BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = os.environ.get("CWRU_DEVICE", "cpu")

RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
