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
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from logger import EvalEpochMeters, Logger, TrainingEpochMeters
from vpn_dataset import VpnDataset


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# for each number of classes, a different dataset is used
# always uses FlowAllLayers, which provides the best accuracy results
# feature:
# vpn -> True = vpn, False = novpn
# malware -> True = benign, False = malware
def get_dataset(dataset_type: str, num_classes: int, feature: bool = None):
    datasets = {
        "vpn_2_classes": Path("data/VPN/2class/SessionAllLayers"),
        "vpn_6_classesTrue": Path("data/VPN/6class_vpn/SessionAllLayers"),
        "vpn_6_classesFalse": Path("data/VPN/6class_novpn/SessionAllLayers"),
        "vpn_12_classes": Path("data/VPN/12class/SessionAllLayers"),
    }
    key = f"{dataset_type}_{num_classes}_classes{feature if feature is not None else ''}"
    if key not in datasets:
        raise ValueError(f"Could not find dataset {key}")
    if not datasets[key].exists:
        raise ValueError(f"Directory {datasets[key]} does not exist! Check for missing files.")
    return datasets[key]


class Trainer(object):
    def __init__(self, model, args):
        self.model = model
        if args.pretrained is not None:
            self.model.load_state_dict(
                torch.load(args.pretrained, map_location="cpu")["state_dict"]
            )

        # Init arguments
        self.args = args
        experiment_name = Path(
            f"{args.dataset}_{self.model.num_classes}_classes_"
            f"{args.bitwidth}_bits_{str(self.args.feature) + '_' if self.args.feature is not None else ''}"
            f"{str(args.pruning_rate).replace('.', '_') if self.args.prune else ''}"
        )
        self.output_dir_path: Path = args.outdir / experiment_name

        if not args.dry_run or not args.evaluate:
            self.checkpoints_dir_path = self.output_dir_path / "checkpoints"
            os.makedirs(self.output_dir_path, exist_ok=True)
            os.makedirs(self.checkpoints_dir_path, exist_ok=True)
            # save the command line argments in json for auto documentation
            with open(self.output_dir_path / "args.json", "w") as f:
                f.write(json.dumps(dict(args), default=lambda x: "N/A", indent=True))

        self.logger = Logger(self.output_dir_path, args.dry_run)

        # Randomness
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        dataset_dir = get_dataset(args.dataset, args.num_classes, args.feature)

        transf = transforms.Compose(
            [
                # normalize inputs to {-1,1} to better adapt to FINN
                transforms.Lambda(lambda x: (x / 255.0) * 2 - 1),
                # dataset comes in 28x28, but needs to be in 1d
                transforms.Lambda(lambda x: torch.flatten(x)),
                # add an additional dimension to represent a single channel
                transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
            ]
        )

        train_dataset = VpnDataset(root=dataset_dir, train=True, transform=transf)
        test_dataset = VpnDataset(root=dataset_dir, train=False, transform=transf)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0
        self.best_training_acc = 0

        # Setup device
        if args.gpus is not None and not args.cpu:
            args.gpus = [int(i) for i in args.gpus.split(",")]
            self.device = "cuda:" + str(args.gpus[0])
            torch.backends.cudnn.benchmark = True

            self.device = torch.device(self.device)
        else:
            self.device = "cpu"
            self.device = torch.device("cpu")

        if args.gpus is not None and len(args.gpus) == 1:
            model = model.to(device=self.device)
        if args.gpus is not None and len(args.gpus) > 1:
            model = nn.DataParallel(model, args.gpus)
        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
        )
        if args.pretrained is not None:
            self.optimizer.load_state_dict(
                torch.load(args.pretrained, map_location=self.device)["optim_dict"]
            )

    def save_checkpoint(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optim_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": self.best_val_acc,
                "best_training_acc": self.best_training_acc,
                "pruning_rate": self.args.pruning_rate,
            },
            best_path,
        )

    def train_model(self, retrain_pruned=False):
        # training starts
        if self.args.detect_nan:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.starting_epoch, self.args.epochs):
            # Set to training mode
            self.model.train()
            self.criterion.train()

            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()

            for i, (input, target) in enumerate(self.train_loader):
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                # Training batch starts
                start_batch = time.time()

                output = self.model(input)

                loss = self.criterion(output, target)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 1))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(
                        epoch_meters, epoch, i, len(self.train_loader)
                    )

                # training batch ends
                start_data_loading = time.time()

            # Perform eval
            top1avg = self.eval_model(epoch)

            if prec1 >= self.best_training_acc:
                self.best_training_acc = prec1
            # checkpoint
            if top1avg >= self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                self.save_checkpoint(epoch, f"best{'_pruned' if self.args.prune else ''}.tar")
            elif not self.args.dry_run or self.args.prune:
                self.save_checkpoint(epoch, "checkpoint.tar")

        if retrain_pruned and not self.args.dry_run:
            self.save_checkpoint(epoch, f"pruned_{self.args.pruning_rate}.tar")
        # training ends
        print("best eval accuracy: ", self.best_val_acc)
        print("best training accuracy: ", self.best_training_acc)

    @torch.no_grad()
    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        if self.args.cpu:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)

        for i, (input, target) in enumerate(self.test_loader):

            end = time.time()

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            # compute loss
            loss = self.criterion(output, target)
            eval_meters.loss_time.update(time.time() - end)

            pred = output.data.argmax(1, keepdim=True)
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100.0 * correct.float() / input.size(0)

            _, prec5 = accuracy(output, target, topk=(1, 1))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            # Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        return eval_meters.top1.avg
