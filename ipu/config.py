
import os
import yaml
import popdist
import multiprocessing
from yacs.config import CfgNode as CN


def handle_distributed_settings(_config):
    # Initialise popdist
    if popdist.isPopdistEnvSet():
        init_popdist(args)
    else:
        _config.IPU.use_popdist = False

def init_popdist(_config):
    hvd.init()
    config.IPU.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replicas:
        logging.warn(f"The number of replicas is overridden by poprun. The new value is {popdist.getNumTotalReplicas()}.")
    _config.IPU.replicas = int(popdist.getNumLocalReplicas())
    _config.IPU.popdist_rank = popdist.getInstanceIndex()
    _config.IPU.popdist_size = popdist.getNumInstances()
    _config.IPU.popdist_local_rank = hvd.local_rank()


def set_ipu_default_config(config):
    config.IPU.dataloader_rebatch_size = 128
    # config.IPU.iterations = None
    config.IPU.exchange_memory_target = 'memory'
    config.IPU.eight_bit_io = True
    config.IPU.dataloader_worker = min(32, multiprocessing.cpu_count())
    config.IPU.profile = False
    config.IPU.profile_dir = ""
    config.IPU.num_io_tiles = 0
    config.IPU.device_iterations = 1
    config.IPU.replication_factor = 1
    config.IPU.enable_half_partials = True
    config.IPU.optimizer_state_offchip = False
    config.IPU.replicated_tensor_sharding = True
    config.IPU.ipus_per_replica = 1
    config.IPU.matmul_proportion = 0.6
    config.IPU.async_dataloader = False
    config.IPU.auto_loss_scaling = False
    config.IPU.executable_cache_dir = ''
    config.IPU.compile_only = False
    config.IPU.precision = "16.16"
    config.IPU.normalization_location = "host"
    config.IPU.synthetic_data = False
    config.IPU.data = "real"
    config.IPU.accumulation_steps = 1

def update_ipu_config(config, args):
    
    config.defrost()
    config.IPU = CN()

    set_ipu_default_config(config)

    if args.dataloader_rebatch_size != None:
        config.IPU.dataloader_rebatch_size = args.dataloader_rebatch_size
    
    if args.iterations != None:
        config.IPU.iterations = args.iterations
    
    if args.exchange_memory_target != None:
        config.IPU.exchange_memory_target = args.exchange_memory_target

    if args.eight_bit_io != None:
        config.IPU.eight_bit_io = args.eight_bit_io
    
    if args.dataloader_worker != None:
        config.IPU.dataloader_worker = args.dataloader_worker

    if args.profile != None:
        config.IPU.profile = args.profile

    if args.profile_dir != None:
        config.IPU.profile_dir = args.profile_dir

    if args.num_io_tiles != None:
        config.IPU.num_io_tiles = args.num_io_tiles
    
    if args.device_iterations != None:
        config.IPU.device_iterations = args.device_iterations

    if args.replication_factor != None:
        config.IPU.replication_factor = args.replication_factor

    if args.enable_half_partials != None:
        config.IPU.enable_half_partials = args.enable_half_partials

    if args.optimizer_state_offchip != None:
        config.IPU.optimizer_state_offchip = args.optimizer_state_offchip

    if args.replicated_tensor_sharding != None:
        config.IPU.replicated_tensor_sharding = args.replicated_tensor_sharding

    if args.ipus_per_replica != None:
        config.IPU.ipus_per_replica = int(args.ipus_per_replica)

    if args.matmul_proportion!= None:
        config.IPU.matmul_proportion = args.matmul_proportion

    if args.async_dataloader != None:
        config.IPU.async_dataloader = args.async_dataloader

    if args.auto_loss_scaling != None:
        config.IPU.auto_loss_scaling = args.auto_loss_scaling

    if args.executable_cache_dir != None:
        config.IPU.executable_cache_dir = args.executable_cache_dir

    if args.compile_only != None:
        config.IPU.compile_only = args.compile_only

    if args.precision != None:
        config.IPU.precision = args.precision

    if args.normalization_location != None:
        config.IPU.normalization_location = args.normalization_location

    if args.accumulation_steps != None:
        config.IPU.accumulation_steps = args.accumulation_steps

    if args.data != None:
        config.IPU.data = args.data

    handle_distributed_settings (config)
    config.freeze()


   
