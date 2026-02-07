#The implementation from the current EP-PINN paper. Need this to compare.

import torch
import torch.nn as nn
from torchdiffeq import odeint
import deepxde as dde

