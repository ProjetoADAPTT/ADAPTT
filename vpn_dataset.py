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

import torch
from torchvision.datasets import VisionDataset

import codecs
import string
import os
import os.path
import numpy as np


class VpnDataset(VisionDataset):

    # classes = {
    #     "Email": 0,
    #     "Chat": 1,
    #     "Streaming": 2,
    #     "Transfer": 3,
    #     "VoIP": 4,
    #     "P2P": 5,
    #     "VPN Email": 6,
    #     "VPN Chat": 7,
    #     "VPN Streaming": 8,
    #     "VPN File Transfer": 9,
    #     "VPN VoIP": 10,
    #     "VPN P2P": 11,
    # }

    def __init__(self, root, download=False, train=False, transform=None, target_transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root (string): Directory containing GTSRB folder.
            download (bool): just to make it compatible with torchvision builders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.root, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.root, label_file))

        return data, targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # print(type(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype(">i2"), "i2"),
    12: (torch.int32, np.dtype(">i4"), "i4"),
    13: (torch.float32, np.dtype(">f4"), "f4"),
    14: (torch.float64, np.dtype(">f8"), "f8"),
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True, as_numpy=False):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    if not as_numpy:
        return torch.from_numpy(parsed.astype(m[2])).view(*s)
    return parsed.astype(m[2]).reshape(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert x.dtype == torch.uint8
    assert x.ndimension() == 1
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    return x
