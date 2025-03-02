import logging
import os
import simplejson

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy # temporary: to resolve metrics issues
from utils import evaluate_metrics

import tent
import norm
import cotta
import cotta_selftrain
import cotta_poly
import cotta_kl
import cotta_cosine

from metrics import *

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    if cfg.MODEL.ADAPTATION == "cotta_selftrain":
        logger.info("test-time adaptation: CoTTA Self-Train")
        model = setup_cotta_selftrain(base_model)
    if cfg.MODEL.ADAPTATION == "cotta_kl":
        logger.info("test-time adaptation: CoTTA w/ KL")
        model = setup_cotta_kl(base_model)
    if cfg.MODEL.ADAPTATION == "cotta_poly":
        logger.info("test-time adaptation: CoTTA w/ POLYLOSS")
        model = setup_cotta_poly(base_model)
    if cfg.MODEL.ADAPTATION == "cotta_cosine":
        logger.info("test-time adaptation: CoTTA w/ COSINE SIM.")
        model = setup_cotta_cosine(base_model)


    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # continual adaptation for all corruption 
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            # metrics = evaluate_metrics(True, True, True, True, model< x_test, y_test, cfg.TEST.BATCH_SIZE)
            arrs = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)

            # err = 1. - acc
            logger.info(f"ACTUAL {corruption_type}{severity}: {arrs[0]}\n\n")
            logger.info(f"PREDICTED {corruption_type}{severity}: {arrs[1]}\n\n")

def apply_metrics(filepath):
    '''Should run immediately after evaluate() in a script. Reads the log from evaluate() and creates a new log summarizing information into a txt.'''
    assert(filepath[-4:] == '.txt')
    filepath_start, filepath_end = '/'.join(filepath.split('/')[:-1]), filepath.split('/')[-1]
    out_filepath = filepath_start + '/../output_metrics/' + filepath_end[:-4] + '_metrics.txt'
    corr_values = extract_corruption_values(filepath) 
    recieved_metrics = calculate_metrics(corr_values)
    with open(out_filepath, "w") as f:
        line = f'METRICS FOR {filepath_end.upper()}'
        f.write(('\n\n' + '#' * len(line)) + '\n' + line + '\n' + ('#' * len(line)) + '\n\n')
        f.write(f'{simplejson.dumps(recieved_metrics, indent = 4)}')


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_cotta(model):
    """Set up CoTTA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_cotta_selftrain(model):
    """Set up CoTTA adaptation w/ self training cross entropy.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta_selftrain.configure_model(model)
    params, param_names = cotta_selftrain.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta_selftrain.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_cotta_poly(model):
    """Set up CoTTA adaptation w/ Polyloss TODO which e?.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta_poly.configure_model(model)
    params, param_names = cotta_poly.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta_poly.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_cotta_kl(model):
    """Set up CoTTA adaptation w/ self training cross entropy.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta_kl.configure_model(model)
    params, param_names = cotta_kl.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta_kl.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_cotta_cosine(model):
    """Set up CoTTA adaptation w/ self training cross entropy.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta_cosine.configure_model(model)
    params, param_names = cotta_cosine.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta_cosine.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
    fp = f'{os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)}'
    apply_metrics(fp)

