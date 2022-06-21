# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import poptorch
import popart
import popdist
import random
import popdist.poptorch
import numpy as np
import ctypes
import os


def generate_random_seed(distributed=False):
    seed = random.randint(0, 2**32-1)
    if distributed:
        seed_tensor = torch.Tensor([seed])
        seed_tensor = hvd.broadcast(seed_tensor, root_rank=0)
        seed = int(seed_tensor.item())
    return seed


def create_training_options(config):
    """
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    """

    if not config.IPU.compile_only and poptorch.ipuHardwareVersion() != 2:
        raise RuntimeError("This version of BERT requires an IPU Mk2 system to run.")

    # Load custom ops
    # if config.IPU.custom_ops is True:
    #     file_dir = os.path.dirname(os.path.realpath(__file__))
    #     CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    #     if os.path.exists(CUSTOM_OP_PATH):
    #         ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
    #     else:
    #         logger("Could not find custom_ops.so. Execute `make` before running this script.")
    #         exit()

    # Poptorch options
    if config.IPU.use_popdist:
        # Use popdist.poptorch options if running in distributed mode
        opts = popdist.poptorch.Options(ipus_per_replica=config.IPU.ipus_per_replica)
    else:
        opts = poptorch.Options()
        # Set the replication factor
        opts.replicationFactor(config.IPU.replication_factor)

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.IPU.device_iterations)

    # Set gradient accumulation factor
    opts.Training.gradientAccumulation(config.IPU.accumulation_steps)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    # Enable automatic loss scaling
    # Note that this is an experimental feature. Note also that it expects
    # accumulationAndReplicationReductionType to be set to Mean as above,
    # and for accumulation by the optimizer to be done in half precision
    # using accum_type=torch.float16 during optimizer instatiation.
    if config.IPU.auto_loss_scaling is True:
        opts.Training.setAutomaticLossScaling(True)
    
    # For efficiency return the sum of the outputs from IPU to host
    opts.outputMode(poptorch.OutputMode.Sum)

    # Fix the random seeds
    seed = generate_random_seed(config.IPU.use_popdist)

    config.defrost()
    config.IPU.seed = seed
    config.freeze()

    opts.randomSeed(seed)

    # Enable Replicated Tensor Sharding (RTS) of optimizer state
    #  with optimizer state residing either on-chip or in DRAM
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        # Optimizer state lives on- or off-chip
        .useOnChipStorage(not config.IPU.optimizer_state_offchip)
        # Shard optimizer state between replicas with zero-redundancy
        .useReplicatedTensorSharding(config.IPU.replicated_tensor_sharding))

    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    # Compile offline (no IPUs required)
    if config.IPU.compile_only:
        opts.useOfflineIpuTarget()

    # Set available Transient Memory For matmuls and convolutions operations
    ## TODO fix here
    # mem_prop = {
    #     f'IPU{i}': config.IPU.matmul_proportion[i]
    #     for i in range( config.IPU.ipus_per_replica )
    # }
    # opts.setAvailableMemoryProportion(mem_prop)

    # Enable caching the compiled executable to disk
    if config.IPU.executable_cache_dir:
        opts.enableExecutableCaching(config.IPU.executable_cache_dir)

    # Enable stochastic rounding (recommended for training with FP16)
    opts.Precision.enableStochasticRounding(True)

    # Half precision partials for matmuls and convolutions
    if config.IPU.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)

    # Enable synthetic random data generated on device (so with no I/O)
    if config.IPU.synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

    # PopART performance options #
    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    # Parallelize optimizer step update across IPUs
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns({"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    # Options for profiling with Popvision
    engine_options = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
        "debug.nanOverflowMode":"true",
        "debug.floatPointOpException":"true"
    }
    if config.IPU.profile_dir:
        engine_options = {
            **engine_options,
            **{
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": config.IPU.profile_dir,
                "profiler.format": "v3",
                "autoReport.all": "true",
            }
        }
    opts._Popart.set("engineOptions", engine_options)
    opts._Popart.set("autoRecomputation", int(popart.RecomputationType.RecomputeAll))

    return opts


def create_validation_options(config, use_popdist = False):
    """
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    """

    if not config.IPU.compile_only and poptorch.ipuHardwareVersion() != 2:
        raise RuntimeError("This version of BERT requires an IPU Mk2 system to run.")

    # Load custom ops
    # if config.IPU.custom_ops is True:
    #     file_dir = os.path.dirname(os.path.realpath(__file__))
    #     CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    #     if os.path.exists(CUSTOM_OP_PATH):
    #         ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
    #     else:
    #         logger("Could not find custom_ops.so. Execute `make` before running this script.")
    #         exit()


    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.IPU.device_iterations)
    # For efficiency return the sum of the outputs from IPU to host
    opts.outputMode(poptorch.OutputMode.Sum)

    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))


    # Set available Transient Memory For matmuls and convolutions operations
    ##TODO fix here
    # mem_prop = {
    #     f'IPU{i}': config.IPU.matmul_proportion[i]
    #     for i in range(config.IPU.ipus_per_replica)
    # }
    # opts.setAvailableMemoryProportion(mem_prop)

    # Enable caching the compiled executable to disk
    if config.IPU.executable_cache_dir:
        opts.enableExecutableCaching(config.IPU.executable_cache_dir)

    # Enable stochastic rounding (recommended for training with FP16)
    opts.Precision.enableStochasticRounding(False)

    # Half precision partials for matmuls and convolutions
    if config.IPU.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)

    # PopART performance options #
    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    # Parallelize optimizer step update across IPUs
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns({"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    # Options for profiling with Popvision
    engine_options = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
    }
    if config.IPU.profile_dir:
        engine_options = {
            **engine_options,
            **{
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": config.IPU.profile_dir,
                "profiler.format": "v3",
                "autoReport.all": "true",
            }
        }
    opts._Popart.set("engineOptions", engine_options)

    return opts

