import scipy.io
import deepxde as dde
import numpy as np
import math

class system_dynamics():
    def __init__(self):
        self.a = 0.01
        self.b = 0.15
        self.D = 0.1
        self.k = 10 #decide on a better value later if necessary.
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.epsilon = 0.002

