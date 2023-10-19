from typing import Dict, Type

from torch import nn

CONV: Dict[bool, Type[nn.Module]] = {
    True: nn.Conv3d,
    False: nn.Conv2d,
}

CONVT: Dict[bool, Type[nn.Module]] = {
    True: nn.ConvTranspose3d,
    False: nn.ConvTranspose2d,
}

BN: Dict[bool, Type[nn.Module]] = {
    True: nn.BatchNorm3d,
    False: nn.BatchNorm2d,
}

MODE: Dict[bool, str] = {
    True: "trilinear",
    False: "bilinear",
}
