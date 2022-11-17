import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np

from swin_transformer.models import build_model
from swin_transformer.config import get_config
from swin_transformer.models.swin_transformer_v2 import SwinTransformerV2
from swin_transformer.logger import create_logger
from swin_transformer.data import build_loader
from swin_transformer.data.build import build_dataset

from options import CONFIG_FILE, NON_HIER_CONFIG_FILE, NON_HIER_MODEL_PATH, MODEL_PATH, ROOT_DATA_DIR, NUM_LEVELS, BATCH_SIZE, WORKERS, RESULTS_DIR

def parse_option(model_type):
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=False,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    # for acceleration
    parser.add_argument(
        "--fused_window_process",
        action="store_true",
        help="Fused window shift & window partition, similar for reversed part.",
    )
    parser.add_argument(
        "--fused_layernorm", action="store_true", help="Use fused layernorm."
    )
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument(
        "--optim",
        type=str,
        help="overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.",
    )

    args, unparsed = parser.parse_known_args()
    args.eval = True
    if model_type == "hierarchy":
        args.resume = MODEL_PATH
        args.cfg = CONFIG_FILE
    else:
        args.cfg = NON_HIER_CONFIG_FILE
        args.resume = NON_HIER_MODEL_PATH
    args.data_path = ROOT_DATA_DIR

    config = get_config(args)

    return args, config

def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")

    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    return model

def load_data(config, model_type):
    train_dset, num_classes = build_dataset(True, config, train_transform=False)
    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    #if model_type == "hierarchy":
    config.freeze()
    val_dset, _ = build_dataset(False, config, train_transform=False)
    return train_dset, val_dset

def load_model(configs):
    model = build_model(configs)
    logger = create_logger(
        output_dir="output",
        dist_rank=0,
        name=f"{configs.EXPERIMENT.NAME}",
    )
    model = load_checkpoint(configs, model, None, None, None, logger)
    return model

def save_data(features, logits, labels, dset_type='val', model_type='hierarchy'):
    np.savez(f'{RESULTS_DIR}/{dset_type}_{model_type}_features.npz', features=features)
    np.savez(f'{RESULTS_DIR}/{dset_type}_{model_type}_labels.npz', labels=labels)
    if model_type == "hierarchy":
        for lvl in range(len(logits)):
            all_logits = []
            for lg in logits[lvl]:
                all_logits.append(lg)
            all_logits = np.array(all_logits)
            np.savez(f'{RESULTS_DIR}/{dset_type}_{model_type}_logits_lvl_{lvl}.npz', logits=all_logits)
    else:
        np.savez(f'{RESULTS_DIR}/{dset_type}_{model_type}_logits.npz', logits=np.array(logits))

if __name__ == "__main__":
    MODEL_TYPE = "non_hierarchy"
    configs = parse_option(MODEL_TYPE)[1]
    train_dset, val_dset = load_data(configs, MODEL_TYPE)
    model = load_model(configs).cuda()
    model.eval()
    if MODEL_TYPE != "hierarchy":
        configs.defrost()
        configs.HIERARCHICAL = True
        configs.freeze()
        train_dset, val_dset = load_data(configs, MODEL_TYPE)

    DSET_TYPE = "val"
    if DSET_TYPE == "train":
        dset = train_dset
    elif DSET_TYPE == "val":
        dset = val_dset

    dloader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    all_labels = []
    all_features = []
    all_logits = []
    if MODEL_TYPE == "hierarchy":
        for lvl in range(NUM_LEVELS):
            all_logits.append([])
    #i = 0
    for img, labels in tqdm(dset):
        #i += 1
        #if i > 10: break
        features = model.forward_features(img.unsqueeze(0).cuda())
        logits = model.head(features)
        all_features.append(features[0].detach().cpu().numpy())
        if MODEL_TYPE == "hierarchy":
            for lvl, logit in enumerate(logits):
                all_logits[lvl].append(logit[0].detach().cpu().numpy())
        else:
            all_logits.append(logits[0].detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    save_data(all_features, all_logits, all_labels, dset_type=DSET_TYPE, model_type=MODEL_TYPE)

    """
    1. Run data through model (train & val/test)
    2. save 
        id
        features
        logits

    3. Look at a subset of categories for TSNE (in hierchical manner)
    4. Look at accuracy of higher order categories in hierchy
    5. Try out images with only partial info (museum images, dset with spurious correlations)

    """

    print("SUCCESS")