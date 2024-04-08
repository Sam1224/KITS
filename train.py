import warnings
import copy
import datetime
import os
import sys
import pathlib
import pandas as pd
from argparse import ArgumentParser

import random
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool


def has_graph_support(model_cls):
    return model_cls in [models.KITS]


def get_model_classes(model_str):
    if model_str == 'kits':
        model, filler = models.KITS, fillers.GCNCycVirtualFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name, miss_rate=0.5, mode="road", test_entries=""):
    if dataset_name[:3] == 'aqi':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36', p=miss_rate)
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=miss_rate, mode=mode, test_entries=test_entries)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'pems07_point':
        dataset = datasets.MissingValuesPems07(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'sea_loop_point':
        dataset = datasets.MissingValuesSeaLoop(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'nrel_al_point':
        dataset = datasets.MissingValuesNrelAl(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'nrel_md_point':
        dataset = datasets.MissingValuesNrelMd(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'ushcn':
        dataset = datasets.MissingValuesUshcn(p_fault=0., p_noise=miss_rate, mode=mode)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--model-name", type=str, default='kits')
    parser.add_argument("--dataset-name", type=str, default='la_point')
    parser.add_argument("--miss-rate", default=0.5, type=float)
    parser.add_argument("--mode", default="road", choices=["road"], type=str)
    parser.add_argument("--test-entries", default="", choices=["", "metr_la_coarse_to_fine.txt", "metr_la_coarse_to_fine_hard.txt", "metr_la_region.txt", "metr_la_region_hard.txt"], type=str)
    parser.add_argument("--config", type=str, default="config/kits/la_point.yaml")
    parser.add_argument("--use-adj-drop", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--use-init", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--pretrained-model", type=str, default="")
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=1.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)

    pl.seed_everything(args.seed)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name, args.miss_rate, args.mode, args.test_entries)

    ########################################
    # create logdir and save configuration #
    ########################################
    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################
    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    if args.dataset_name in ["pems07_point"]:
        data_conf["scaling_type"] = "minmax"
        dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                      **data_conf)
        min_val = 0
        max_val = 1500
        print("Min Max Scaler - max: {}".format(max_val))
        dm.setup(min=min_val, max=max_val)
    elif args.dataset_name in ["nrel_al_point", "nrel_md_point"]:
        print("Use capacities as Min Max Scaler")
        data_conf["scaling_type"] = "minmax"
        dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                      **data_conf)
        files_info = pd.read_pickle('datasets/{}/nrel_file_infos.pkl'.format(args.dataset_name.replace("_point", "")))
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
        capacities = np.expand_dims(capacities, axis=(0, -1))
        min_val = np.zeros_like(capacities)
        dm.setup(min=min_val, max=capacities)
    else:
        dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                      **data_conf)
        dm.setup()

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    # data = dataset.numpy()
    # train_mask = dataset.training_mask
    # test_mask = dataset.eval_mask
    #
    # train_y = data[dm.train_slice, ...]
    # train_m = train_mask[dm.train_slice, ...]
    # val_y = data[dm.val_slice, ...]
    # val_m = test_mask[dm.val_slice, ...]
    # test_y = data[dm.test_slice, ...]
    # test_m = test_mask[dm.test_slice, ...]
    #
    # if not os.path.exists("data"):
    #     os.mkdir("data")
    # data_name = ""
    # if "ushcn" in args.dataset_name:
    #     data_name = "USHCN"
    # base_path = "data/{}/".format(data_name)
    # if not os.path.exists(base_path):
    #     os.mkdir(base_path)
    # base_path = "{}{}/".format(base_path, args.mode)
    # if not os.path.exists(base_path):
    #     os.mkdir(base_path)
    # if not os.path.exists("{}train_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else "")):
    #     np.save("{}train_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), train_y)
    #     np.save("{}train_m_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), train_m)
    #     np.save("{}val_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), val_y)
    #     np.save("{}val_m_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), val_m)
    #     np.save("{}test_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), test_y)
    #     np.save("{}test_m_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), test_m)
    #
    # if not os.path.exists("{}adj.npy".format(base_path, args.miss_rate)):
    #     np.save("{}adj.npy".format(base_path, args.miss_rate), adj)
    # sys.exit(0)

    # # ========================================
    # # for scalability testing purpose
    # print("Original adj shape:", adj.shape)
    # adj = np.repeat(adj, 100, axis=0)
    # adj = np.repeat(adj, 100, axis=1)
    # print("Scaled adj shape:", adj.shape)
    # # ========================================

    ########################################
    # predictor                            #
    ########################################
    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes, args=args)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {
        'mae': MaskedMAE(compute_on_step=False),
        'mape': MaskedMAPE(compute_on_step=False),
        'mse': MaskedMSE(compute_on_step=False),
        'mre': MaskedMRE(compute_on_step=False)
    }

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     }
                                     )
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    if args.pretrained_model == "" or args.pretrained_model == None:
        ########################################
        # training                             #
        ########################################
        # callbacks
        early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min', verbose=True)
        checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

        logger = TensorBoardLogger(logdir, name="model")

        trainer = pl.Trainer(max_epochs=args.epochs,
                             logger=logger,
                             default_root_dir=logdir,
                             gpus=1 if torch.cuda.is_available() else None,
                             gradient_clip_val=args.grad_clip_val,
                             gradient_clip_algorithm=args.grad_clip_algorithm,
                             callbacks=[early_stop_callback, checkpoint_callback])

        trainer.fit(filler, datamodule=dm)

        trainer.test()

        filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                          lambda storage, loc: storage)['state_dict'])
    else:
        state_dict = torch.load(args.pretrained_model, lambda storage, loc: storage)['state_dict']
        adj = torch.from_numpy(adj)
        state_dict["model.adj"] = adj  # in case of using pretrained model of other datasets to infer current dataset
        filler.load_state_dict(state_dict)

    ########################################
    # testing                              #
    ########################################
    filler.freeze()
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)
    y_hat = y_hat.detach().squeeze(-1).cpu().numpy()  # reshape to (eventually) squeeze node channels

    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]

    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mape': numpy_metrics.masked_mape,
        'mre': numpy_metrics.masked_mre,
        'mse': numpy_metrics.masked_mse,
        'r2': numpy_metrics.masked_r2
    }
    # Aggregate predictions in dataframes
    index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']

    sys.exit(0)

    # tar
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
    for aggr_by, df_hat in df_hats.items():
        # Compute error
        print(f'- AGGREGATE BY {aggr_by.upper()}')
        for metric_name, metric_fn in metrics.items():
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            print(f' {metric_name}: {error:.4f}')
            if metric_name == "mse":
                print(f'rmse: {np.sqrt(error):.4f}')

    return y_true, y_hat, mask


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    run_experiment(args)
