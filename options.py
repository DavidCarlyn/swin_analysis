CONFIG_FILE = "swin_transformer/configs/hierarchical-vision-project/groovy-grape-192.yaml"
NON_HIER_CONFIG_FILE = "swin_transformer/configs/hierarchical-vision-project/fuzzy-fig-192.yaml"


NON_HIER_MODEL_PATH = "/local/scratch/hierarchical-vision-checkpoints/fuzzy-fig-192-epoch89.pth"
MODEL_PATH = "/local/scratch/hierarchical-vision-checkpoints/groovy-grape-192-epoch89.pth"

ROOT_DATA_DIR = "/local/scratch/cv_datasets/inat21/resize-192"

NUM_LEVELS = 7 # Number of level on the classification hierchy
BATCH_SIZE = 8
WORKERS = 4

RESULTS_DIR = "/local/scratch/carlyn.1/swin_inat_results"

LABEL_MAP_PATH = "label_map.json"