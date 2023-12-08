# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json

import torch
from brevitas.nn import QuantConv1d, QuantLinear
from torch.nn import BatchNorm1d, Module

# TODO: criar um dicionario para anotar as novas configurações de cada layer
# facilmente salvo como json para verificação dps
# fazer todos os calculos relevantes antes de modificar o modelo

# the given network model changes after function
# creates a new pruned network
# folding_cfg: csv file containing PEs and SIMD of each conv and fc layer
# map_size: the size of the feature maps before the flatten layer
# TODO: to make this truly generic, change how batchnorm is handled
@torch.no_grad()
def filter_pruning_conv1d(
    network_model: Module,
    pruning_rate: float,
    folding_cfg_file: str,
    map_size: int,
    device: str = "cpu",
    independent_prune_flag: bool = True,
    log=None,
) -> Module:
    network_model = network_model.to(device)
    # read folding_cfg json file from FINN folding configuration
    # Remember that #channels (MH) has to be divisible by layers PE in FINN...
    # and #input channels must be divisible by (MW) SIMD
    with open(folding_cfg_file, "r") as cfg_f:
        cfg = json.load(cfg_f)
        PEs = [
            int(layer["PE"])
            for key, layer in cfg.items()
            if key.startswith("StreamingFCLayer_Batch")
        ]
        SIMDs = [
            int(layer["SIMD"])
            for key, layer in cfg.items()
            if key.startswith("StreamingFCLayer_Batch")
        ]

    if not 1.0 > pruning_rate > 0.0:
        raise ValueError(f"Pruning rate of {pruning_rate} outside of range (0, 1).")

    original_output_channels = [
        layer.out_channels for layer in network_model if isinstance(layer, QuantConv1d)
    ]
    # does not work for 2d conv
    kernel_sizes = [
        layer.kernel_size[0] for layer in network_model if isinstance(layer, QuantConv1d)
    ]
    # considers the size of the feature map input in the first FC layer
    kernel_sizes.append(map_size)

    if len(original_output_channels) > len(PEs) or len(PEs) != len(SIMDs):
        raise ValueError(
            f"Number of convolutional layers ({len(original_output_channels)}"
            f" does not match with the number of given PEs ({len(PEs)}) or SIMDs ({len(SIMDs)})"
            f" from configuration file: {folding_cfg_file}"
        )

    # calculates the number of channels to prune to be able to fit into FINN
    # select which layers to prune and the amount of channels to prune
    prune_layers = []
    prune_channels = []
    input_channels = []
    for i, (current_PE, next_SIMD, kernel_size, ch) in enumerate(
        zip(PEs, SIMDs[1:], kernel_sizes, original_output_channels)
    ):
        # calculate the number of channels to maintain after pruning
        prune_ch = round(ch * pruning_rate)
        ch_remnaining = ch - prune_ch
        # pruning must respect the constraints of the hardawre folding
        if test_folding(ch_remnaining, current_PE, next_SIMD, kernel_size):
            prune_layers.append("conv" + str(i))
            prune_channels.append(prune_ch)
            input_channels.append(ch_remnaining)
        else:
            # try every iteration of possible values until requirements are met
            for prune_ch in range(prune_ch, -1, -1):
                ch_remaining = ch - prune_ch
                if test_folding(ch_remaining, current_PE, next_SIMD, kernel_size) and prune_ch > 0:
                    prune_layers.append("conv" + str(i))
                    prune_channels.append(prune_ch)
                    input_channels.append(ch_remaining)
                    break
            if prune_ch == 0:
                input_channels.append(ch_remaining)

    if len(prune_layers) == 0:
        raise RuntimeWarning(
            f"Could not find a folding configuration for this model with pruning rate of {pruning_rate}"
        )

    pruned_model = step_filter_prune_conv1d(
        network_model,
        prune_layers,
        prune_channels,
        independent_prune_flag,
        map_size,
    )
    # change device of the new model
    pruned_model = pruned_model.to(device)
    # log the changes
    if log is not None:
        with open(log, "w") as f:
            f.write(f"{prune_channels}, {prune_layers}, {input_channels}")

    # TODO: salvar o modelo prunado como ONNX
    return pruned_model


# checks if the new number of output channels ch will fit into the given folding configuration of FINN
# next_layer_feature_map_size is the size of the kernel of the next conv layer or the feature map input into FC layer
# In case of FC, it is not being necessarily the output of the current conv layer, as it might be further reduced by a MaxPool layer
def test_folding(ch, current_layer_pe, next_layer_simd, next_layer_feature_map_size=1):
    # checks if the number of inputs to the next channel will fit the number of simds
    t_simd = (ch * next_layer_feature_map_size) % next_layer_simd == 0
    # checks if the number of output will fit the number of PEs
    t_pe = ch % current_layer_pe == 0
    return t_simd and t_pe


# map_size referes to the last feature map size before flatten layer
def step_filter_prune_conv1d(
    network, prune_layers, prune_channels, independent_prune_flag, map_size
):
    # find the index of the first fc layer
    first_fc_index = next((x for x, layer in enumerate(network) if isinstance(layer, QuantLinear)), None)
    if first_fc_index is None:
        first_fc_index = -1

    count = 0  # count for indexing 'prune_channels'
    conv_count = 0  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None  # residue is need to prune by 'independent strategy'
    for i, layer in enumerate(network[:first_fc_index]):
        if isinstance(layer, QuantConv1d):
            # if the previous layer was pruned, change the number of input channels
            if dim == 1:
                # adjust input channels
                residue = prune_conv_layer(
                    network[i],
                    dim,
                    channel_index,
                    independent_prune_flag,
                )
                dim ^= 1  # toggle dim from 1 to 0

            # prune output channels
            if f"conv{conv_count}" in prune_layers:
                channel_index = calculate_indexes_to_prune(
                    layer.weight.data, prune_channels[count], residue
                )
                _ = prune_conv_layer(
                    network[i],
                    dim,
                    channel_index,
                    independent_prune_flag,
                )
                dim ^= 1  # switch dim from 0 to 1 or 1 to 0
                count += 1
            else:
                residue = None
            conv_count += 1

        elif dim == 1 and isinstance(layer, BatchNorm1d):
            prune_batchnorm1d_layer(network[i], channel_index)

    # find the index of the first fc layer
    if first_fc_index != -1:
        prune_first_linear_layer(network[first_fc_index], channel_index, map_size)

    # just for debugging purposes
    layers_weight_and_bias= [
        (layer.weight.data.shape, layer.bias.data.shape)
        for layer in network
        if isinstance(layer, (QuantConv1d, QuantLinear))
    ]
    print(layers_weight_and_bias)
    return network


# given a set of kernels, return the index of kenels to prune
# residue is used in independent strategy
# pruning based on < https://arxiv.org/pdf/1608.08710.pdf >
# bias is not considered for pruning
def calculate_indexes_to_prune(kernel, num_elimination, residue=None):
    # channels are selected based on their floating point representation
    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    _, idx = torch.sort(sum_of_kernel)

    # select the filters with the lowest sum to be pruned
    return idx[:num_elimination].tolist()


def calculate_fc_indexes(filter_indexes, feature_map_input):
    # selects which indexes correspond to the pruned feature map input
    fc_indexes = []
    for f_id in filter_indexes:
        # add all indexes corresponding to the pruned feature map
        fc_indexes += list(range(f_id * feature_map_input, f_id * feature_map_input + feature_map_input))
    return sorted(fc_indexes)


# indexes = list of indexes of the pruned filter to be discarded
# returns the tensor without given indexes and a tensor with the pruned values
# pruned values are residue for the independent pruning strategy
def remove_indexes(tensor, dim, indexes):
    # select only the indexes that were not pruned
    select_index = list(set(range(tensor.size(dim))) - set(indexes))
    new_tensor = torch.index_select(
        tensor, dim, torch.tensor(select_index, device=tensor.device)
    )
    pruned_values = torch.index_select(
        tensor, dim, torch.tensor(indexes, device=tensor.device)
    )

    return new_tensor, pruned_values


# change convolution layer weights and bias and return residue if changing output layer
# works for both 1d and 2d convolution
# dim = 0 -> changes the number of output channels
# dim = 1 -> changes the number of input channels
def prune_conv_layer(conv, dim, channel_index, independent_prune_flag=False):
    conv.weight.data, res = remove_indexes(conv.weight.data, dim, channel_index)
    # only change the bias if changing the number of output channels
    if conv.bias is not None and dim == 0:
        conv.bias.data, _ = remove_indexes(conv.bias.data, 0, channel_index)

    return res if independent_prune_flag else None


# create a new batchnorm layer to ensure correct initialization of values
def prune_batchnorm1d_layer(norm, channel_index):
    norm.num_features = int(norm.num_features) - len(channel_index)
    norm.weight.data, _ = remove_indexes(norm.weight.data, 0, channel_index)
    norm.bias.data, _ = remove_indexes(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        norm.running_mean.data, _ = remove_indexes(norm.running_mean.data, 0, channel_index)
        norm.running_var.data, _ = remove_indexes(norm.running_var.data, 0, channel_index)


# Removes the pruned weights from the first linear layer of the network
# bias is not changed, as the number of neurons remains the same
# map size refers to the size of feature maps from the last Conv or MaxPool layer
def prune_first_linear_layer(linear, channel_index, map_size):
    channel_index = calculate_fc_indexes(channel_index, map_size)
    linear.weight.data, _ = remove_indexes(linear.weight.data, 1, channel_index)
