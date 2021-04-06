#! /usr/bin/env python

""" Top-level script for evaluating models """
from pathlib import Path

import numpy as np
import logging
import sys
import torch
from torch.utils.data import DataLoader
import configargparse
import copy
import tempfile
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append("../")

from datasets import load_simulator
from utils import create_filename, create_modelname, sum_except_batch, array_to_image_folder
from architectures import create_model
from architectures.create_model import ALGORITHMS

logger = logging.getLogger(__name__)

try:
    from fid_score import calculate_fid_given_paths
except:
    logger.warning("Could not import fid_score, make sure that pytorch-fid is in the Python path")
    calculate_fid_given_paths = None


def parse_args():
    """ Parses command line arguments for the evaluation """

    parser = configargparse.ArgumentParser(ignore_unknown_config_file_keys=True)

    # What what what
    parser.add_argument("--truth", action="store_true", help="Evaluate ground truth rather than learned model")
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument("--algorithm", type=str, default="flow", choices=ALGORITHMS, help="Model: flow (AF), mf (FOM, M-flow), emf (Me-flow), pie (PIE), gamf (M-flow-OT)...")
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others")
    parser.add_argument("-i", type=int, default=0, help="Run number")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ood_dataset", action="append", default=["celeb-a", "svhn", "lsun", "cifar100", "uniform_noise", "textures", "gaussian_noise"])
    parser.add_argument("--dataset_dir", type=str, default="./downloaded_datasets")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=3, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--innertransform", type=str, default="rq-coupling", help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--lineartransform", type=str, default="permutation", help="Scalar linear trf: linear | permutation")
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--pieclip", type=float, default=None, help="Clip v in p(v), in multiples of epsilon")
    parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in Me-flow / PAE encoder")
    parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in Me-flow / PAE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation f)")
    parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
    parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")
    parser.add_argument("--linlayers", type=int, default=2, help="Number of linear layers before the projection for M-flow and PIE on image data")
    parser.add_argument("--linchannelfactor", type=int, default=2, help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
    parser.add_argument("--intermediatensf", action="store_true", help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
    parser.add_argument("--decoderblocks", type=int, default=5, help="Number of blocks in PAE encoder")
    parser.add_argument("--decoderhidden", type=int, default=100, help="Number of hidden units in PAE encoder")

    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--debug", default=True)
    parser.add_argument("--num_classes", default=0, type=int)

    return parser.parse_args()

def evaluate_test_samples(args, simulator, model=None, eval_classifier=False):
    """ Likelihood evaluation """
    # Prepare
    dataset = simulator.load_dataset(train=False, dataset_dir=Path(args.dataset_dir))
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=False,
        num_workers=0,
        shuffle=False,
    )

    # Evaluate
    log_prob_ = []
    reco_error_ = []
    logits = []
    ys = []
    for batch in dataloader:
        x_, y = batch
        ys.append(y.cpu())
        x_ = x_.cuda()
        if args.algorithm == "flow":
            out = model(x_, context=params_)
        elif args.algorithm in ["pie", "slice"]:
            out = model(x_, context=params_, mode=args.algorithm if not args.skiplikelihood else "projection", )
        else:
            out = model(x_, context=None, mode="mf-fixed-manifold", return_classification=True)

        x_reco, log_prob, u, hidden, clf_out = (
            out["x_reco"], out["log_prob"], out["u"], out["hidden"], out["clf_out"])
        
        logits.append(clf_out.detach().cpu())

        log_prob_.append(log_prob.detach().cpu().numpy())
        reco_error_.append((sum_except_batch((x_ - x_reco) ** 2) ** 0.5).detach().cpu().numpy())
    
    if eval_classifier:
        ys = torch.cat(ys)
        logits = torch.cat(logits)
        acc = (ys == logits.argmax(-1)).float().mean()
        print(f"Accuracy: {acc.item() * 100:.02f}")

    log_prob = np.concatenate(log_prob_, axis=0)
    reco_error = np.concatenate(reco_error_, axis=0)
    return {"p(x)": log_prob, "reco_error": -reco_error}


if __name__ == "__main__":
    # Parse args
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    # Silence PIL
    for key in logging.Logger.manager.loggerDict:
        if "PIL" in key:
            logging.getLogger(key).setLevel(logging.WARNING)

    logger.info("Hi!")
    logger.debug("Starting evaluate.py with arguments %s", args)

    create_modelname(args)
    logger.info("Evaluating model %s", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    simulator = load_simulator(args)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = create_model(args, simulator=simulator)
    if args.model_path is None:
        args.model_path = create_filename("model", None, args)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cuda")))
    model.cuda()
    model.eval()

    # Compute ID OOD scores
    id_scores_dict = evaluate_test_samples(args, simulator, model=model, eval_classifier=True)

    # Compute OOD detection metrics
    rows = []
    for ood_ds in args.ood_dataset:
        logger.info(f"\n\n{ood_ds}")

        args.dataset = ood_ds
        simulator = load_simulator(args)
        ood_scores_dict = evaluate_test_samples(args, simulator, model=model)

        for score_name, id_scores in id_scores_dict.items():
            ood_scores = ood_scores_dict[score_name]

            labels = np.concatenate(
                [np.zeros_like(ood_scores), np.ones_like(id_scores)]
            )
            preds = np.concatenate([ood_scores, id_scores])

            auroc = roc_auc_score(labels, preds)
            aupr = average_precision_score(labels, preds)

            logger.info(score_name)
            logger.info(f"AUROC: {auroc * 100:.02f}")
            logger.info(f"AUPR: {aupr * 100:.02f}")
            rows.append((score_name, ood_ds, auroc * 100, aupr * 100))

    model_name = Path(args.model_path).stem
    df = pd.DataFrame(rows, columns=["Score", "OOD Dataset", "AUROC", "AUPR"])
    df.to_csv(f"{model_name}.csv", index=False)
    logger.info("All done! Have a nice day!")
