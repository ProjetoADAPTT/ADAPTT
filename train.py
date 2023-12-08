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

import argparse
from pathlib import Path

import torch

from model import VpnNonVpnModel
from trainer import Trainer

from brevitas.nn import QuantConv1d, QuantLinear

# Pytorch precision
torch.set_printoptions(precision=10)

# Util method to add mutually exclusive boolean
def add_bool_arg(parser, name, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# Util method to pass None as a string and be recognized as None value
def none_or_str(value):
    if value == "None":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


# commented out some options that will not be used right now
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    # I/O
    parser.add_argument(
        "--outdir", default=Path("experiments"), type=Path, help="Path to output folder"
    )
    parser.add_argument("--dry_run", action="store_true", help="Disable output files generation")
    parser.add_argument("--log_freq", type=int, default=10)
    # Execution modes
    parser.add_argument(
        "--evaluate",
        type=Path,
        default=None,
        help="Evaluate model on validation set loading params from path",
    )
    parser.add_argument(
        "--export", type=Path, default=None, help="Export ONNX model to the given path"
    )
    parser.add_argument(
        "--save-onnx",
        default=False,
        action="store_true",
        help="Only save onnx file. Does not evaluate or train",
    )
    add_bool_arg(parser, "detect_nan", default=False)
    # Compute resources
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--gpus", type=none_or_str, default="0", help="Comma separated GPUs")
    add_bool_arg(parser, "cpu", default=False)
    # Model Architecture and hyperparameters
    parser.add_argument(
        "--num_classes",
        choices=[2, 6, 10, 12, 20],
        default=2,
        type=int,
        help="Number of classes. Must be 2, 6 or 12 for vpn and 2, 10 or 20 for malware.",
    )
    add_bool_arg(parser, "feature", None)
    parser.add_argument(
        "--dataset",
        choices=["vpn", "malware"],
        default="vpn",
        type=str,
        help="Dataset to train the network",
    )
    parser.add_argument("--bitwidth", default=4, type=int, help="Bit width of quantization")
    parser.add_argument("--input_bitwidth", default=2, type=int, help="Bit width of quantization")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--random_seed", default=1, type=int, help="Random seed")
    parser.add_argument(
        "--pretrained", default=None, type=Path, help="Load pretrained model from given filename"
    )

    # pruning related args
    add_bool_arg(parser, "prune", default=False)
    parser.add_argument(
        "--pruning_rate",
        default=0.0,
        type=float,
        help="Pruning Rate (how much of original size). Between 0 and 1.",
    )
    parser.add_argument(
        "--folding_cfg",
        default=None,
        type=Path,
        help="Folding configuration for FINN-aware pruning.",
    )
    parser.add_argument(
        "--independent_prune",
        action="store_true",
        help='prune multiple layers by "independent strategy"',
        default=False,
    )
    add_bool_arg(parser, "retrain", default=False)
    return parser.parse_args()


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def main():
    args = parse_args()

    # Access config as an object
    args = objdict(args.__dict__)

    if args.evaluate:
        args.dry_run = True

    model = VpnNonVpnModel(
        num_classes=args.num_classes,
        weight_bit_width=args.bitwidth,
        bias_bit_width=args.bitwidth,
        act_bit_width=args.bitwidth,
        input_bit_width=args.input_bitwidth,
    )
    # Init trainer
    trainer = Trainer(model, args)

    if args.evaluate is not None:
        print(f"Loading model from {args.evaluate}")
        saved_dict = torch.load(args.evaluate)
        model.load_state_dict(saved_dict["state_dict"])
        print(f"Loaded model with precision of {saved_dict['best_val_acc']}")
        trainer.eval_model()
    elif args.prune:
        print("\n\nPruning of " + str(args.pruning_rate) + "\n\n")
        pruned_model = trainer.model.get_pruned_model(
            args.pruning_rate, args.folding_cfg, args.independent_prune, trainer.device
        )
        trainer.model = pruned_model
        print("Evaluating model before retraining...")
        pruned_top1 = trainer.eval_model()
        print(f"Pruned {args.pruning_rate} Top-1 = {pruned_top1}")
        # retrain pruned model
        if args.retrain: 
            print("Retraining pruned model...")
            trainer.train_model(retrain_pruned=True)
            retrained_pruned_top1 = trainer.eval_model()
            print(f"Pruned {args.pruning_rate} Retrained Top-1 = {retrained_pruned_top1}")

        layers_weight_and_bias= [
            (layer.weight.data.shape, layer.bias.data.shape)
            for layer in trainer.model.net
            if isinstance(layer, (QuantConv1d, QuantLinear))
        ]
        print(layers_weight_and_bias)

    elif not args.save_onnx:
        print("Training model")
        print(model)
        trainer.train_model()

    if args.export is not None:
        trainer.model.export_model(args.export)


if __name__ == "__main__":
    main()
