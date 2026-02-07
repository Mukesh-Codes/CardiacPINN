import scipy.io as sio
import h5py
import numpy as np
import torch
from .PINN_initial import Geometry
import deepxde as dde
from typing import Callable

#Need to update the MATLAB script for more iterations.

class Data():
    def __init__(self, pde: Callable, geomtime):
        self.pde = pde
        with h5py.File('spiral_data.mat', 'r') as f:
            self.keys = list(f.keys())
            self.data = [f[key] for key in self.keys]
            self.dt = float(f['dt'][()])
            self.Vsav = np.array(f['Vsav'])
            self.Wsav = np.array(f['Wsav'])
        self.ddedata = dde.data.TimePDE(geomtime, pde, [], num_domain = 8000, num_boundary=400, num_inital=800)
