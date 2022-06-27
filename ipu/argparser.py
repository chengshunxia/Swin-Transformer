# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import yaml
import multiprocessing
# import import_helper
import models


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


class YAMLNamespace(argparse.Namespace):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def get_available_configs(config_file):
    with open(config_file) as file:
        configs = yaml.full_load(file)
    return configs


def parse_with_config(parser, config_file):
    configurations = get_available_configs(config_file)
    parser.add_argument('--config', choices=configurations.keys(), help="Select from avalible configurations")
    args = parser.parse_args()
    if args.config is not None:
        # Load the configurations from the YAML file and update command line arguments
        loaded_config = YAMLNamespace(configurations[args.config])
        # Check the config file keys
        for k in vars(loaded_config).keys():
            assert k in vars(args).keys(), f"Couldn't recognise argument {k}."

        args = parser.parse_args(namespace=loaded_config)
    if args.dataloader_worker is None:
        # determine dataloader-worker
        args.dataloader_worker = min(32, multiprocessing.cpu_count())
    return args


def get_common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    ipu_arg_group = parser.add_argument_group('IPU Configuration')
    ipu_arg_group.add_argument('--dataloader-rebatch-size', type=int, help='Dataloader rebatching size. (Helps to optimise the host memory footprint)')
    ipu_arg_group.add_argument('--iterations', type=int, help='number of program iterations')
    ipu_arg_group.add_argument('--exchange-memory-target', choices=['cycles', 'balanced', 'memory'], help='Exchange memory optimisation target: balanced/cycles/memory. In case of '
                        'cycles it uses more memory, but runs faster.')
    ipu_arg_group.add_argument('--eight-bit-io', action='store_true', help="Image transfer from host to IPU in 8-bit format, requires normalisation on the IPU")
    ipu_arg_group.add_argument('--dataloader-worker', type=int, help="Number of worker for each dataloader")
    ipu_arg_group.add_argument('--profile', action='store_true', help='Create PopVision Graph Analyzer report')
    ipu_arg_group.add_argument('--num-io-tiles', type=int, help='Number of IO tiles. Minimum 32. Default 0 (no overlap)')
    
    ipu_arg_group.add_argument("--device-iterations", type=int, help="Number of batches per training step")
    ipu_arg_group.add_argument("--replication-factor", type=int, help="Number of replicas")
    ipu_arg_group.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable half partials for matmuls and convolutions globally")
    ipu_arg_group.add_argument("--optimizer-state-offchip", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Set the tensor storage location for optimizer state to be offchip.")
    ipu_arg_group.add_argument("--replicated-tensor-sharding", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable replicated tensor sharding of optimizer state")
    ipu_arg_group.add_argument("--ipus-per-replica", type=int, help="Number of IPUs required by each replica")
    ipu_arg_group.add_argument("--matmul-proportion", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul")
    ipu_arg_group.add_argument("--async-dataloader", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable asynchronous mode in the DataLoader")
    ipu_arg_group.add_argument("--auto-loss-scaling", type=str_to_bool, nargs="?", const=True, default=False, help="Enable automatic loss scaling\
                             for half precision training. Note that this is an experimental feature.")
    ipu_arg_group.add_argument("--executable-cache-dir", type=str, default="",
                        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
                        "Required for both saving and loading executables.")
    ipu_arg_group.add_argument("--compile-only", action="store_true", help="Create an offline IPU target that can only be used for offline compilation.")
    ipu_arg_group.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    ipu_arg_group.add_argument('--normalization-location', choices=['host', 'ipu', 'none'], default='host', help='Location of the data normalization')
    ipu_arg_group.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False, help="Random data created on IPU")
    ipu_arg_group.add_argument("--profile-dir", type=str, help="Directory for profiling results")
    ipu_arg_group.add_argument('--data', choices=['real', 'synthetic', 'generated', 'imagenet'], help="Choose data")
    #TODO
    ipu_arg_group.add_argument('--use-popdist', action='store_true', help='Whether to use poprun to run the training')
    
    return parser
