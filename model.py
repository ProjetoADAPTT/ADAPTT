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

# CNV network model based on the paper:
# 'End-to-end Encrypted Traffic Classification with One-dimensional Convolution Neural Networks',
# available at <www.doi.org/10.1109/ISI.2017.8004872>

import brevitas.nn as qnn
import numpy as np
import torch
from brevitas.export import FINNManager
from brevitas.quant_tensor import QuantTensor
from torch import nn
from copy import deepcopy

import prune
import quantizers as quant


# Model based on the paper from
# "End-to-end encrypted traffic classification with one-dimensional convolution neural networks", Wang et al (2017)
# Available at <www.doi.org/10.1109/ISI.2017.8004872>
# Using brevitas built-in quantizers, changing bit_width
class VpnNonVpnModel(nn.Module):
    def __init__(
        self, num_classes, act_bit_width=4, weight_bit_width=4, bias_bit_width=4, input_bit_width=2
    ):
        super().__init__()

        if num_classes not in (2, 6, 10, 12, 20):
            raise ValueError(
                f"{num_classes} number of classes is invalid.\n"
                "The only implemented number of classes for this model is 2, 6 or 12"
            )

        self.num_classes = num_classes
        self.act_bit_width = act_bit_width
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
        self.input_bit_width = input_bit_width

        input_quant = quant.CommonActQuant
        act_quant = quant.CommonUnsignedActQuant
        weight_quant = quant.CommonWeightQuant
        bias_quant = quant.CommonBiasQuant

        kernel_size = 25
        padding = 12  # same padding

        # Added BatchNorm layers
        # Removed padding from MaxPool1d Layers
        # changed MaxPool1d kernel size from 3 to 4
        self.net = nn.Sequential(
            qnn.QuantIdentity(
                act_quant=input_quant, bit_width=input_bit_width, return_quant_tensor=True
            ),
            qnn.QuantConv1d(
                1,
                32,
                kernel_size,
                padding=padding,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                bias=True,
                bias_bit_width=bias_bit_width,
                bias_quant=bias_quant,
            ),
            nn.BatchNorm1d(32),
            qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True),
            nn.MaxPool1d(4),
            qnn.QuantConv1d(
                32,
                64,
                kernel_size,
                padding=padding,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                bias=True,
                bias_bit_width=bias_bit_width,
                bias_quant=bias_quant,
            ),
            nn.BatchNorm1d(64),
            qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True),
            nn.MaxPool1d(4),
            nn.Flatten(),
            qnn.QuantLinear(
                64 * 49,
                1024,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                bias=True,
                bias_quant=bias_quant,
                bias_bit_width=bias_bit_width,
            ),
            nn.BatchNorm1d(1024),
            qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True),
            qnn.QuantLinear(
                1024,
                self.num_classes,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                bias=True,
                bias_quant=bias_quant,
                bias_bit_width=bias_bit_width,
            ),
            nn.Softmax(1),
        )

        self.init_weights()

    def clip_weights(self, min_val, max_val):
        for mod in self.net:
            if isinstance(mod, (qnn.QuantConv1d, qnn.QuantLinear)):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        return self.net(x)

    # export model to onnx format
    def export_model(self, onnx_name):
        input_size = (1, 1, 784)
        # net containts the model to be synthesize by FINN
        # create a QuantTensor instance to mark input as bipolar during export
        input_a = np.random.randint(-1, 1, size=input_size).astype(np.float32)
        scale = 1.0
        input_t = torch.from_numpy(input_a * scale)
        input_qt = QuantTensor(
            input_t, scale=torch.tensor(scale), bit_width=torch.tensor(1.0), signed=True
        )
        FINNManager.export(self.net.eval().to("cpu"), input_t=input_qt, export_path=onnx_name)

    def init_weights(self):
        for mod in self.net:
            if isinstance(mod, (qnn.QuantLinear, qnn.QuantConv1d)):
                nn.init.uniform_(mod.weight.data, -1, 1)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias.data)

    # Returns a deep copy of the model
    # with pruned channels on convolutional layers
    # folding_cfg is required so that the pruning is FINN-aware
    def get_pruned_model(
        self, pruning_rate: float, folding_cfg_file: str, independent_prune_flag: bool, device:str="cpu"
    ):
        pruned_net = deepcopy(self.net)
        pruned_model = deepcopy(self)
        new_net = prune.filter_pruning_conv1d(
            pruned_net,
            pruning_rate,
            folding_cfg_file,
            49,
            independent_prune_flag=independent_prune_flag,
            device=device
        )
        pruned_model.net = new_net
        return pruned_model

        
