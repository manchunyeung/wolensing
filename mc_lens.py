import numpy as np
import sympy as sp
from scipy.optimize import minimize
from abc import ABC, abstractmethod

class LensModel(ABC):
    
    def __init__(self, y0, y1 = 0, x_min = None):
        self.y0 = y0
        self.y1 = y1   
        if not isinstance(x_min, np.ndarray):
            self.x_min = self.find_minima()
        else:
            self.x_min = x_min
        self.T_min = self.T_2D_raw_wrap(self.x_min)
        
    @abstractmethod
    def Psi_2D(self, x0, x1):
        raise NotImplementedError("Must override Psi_2D in child class")
    
    def T_2D_raw(self, x0, x1):
        T = ((x0 - self.y0)**2 + (x1 - self.y1)**2)/2 - self.Psi_2D(x0, x1)
        return T
    
    def T_2D_raw_wrap(self, x):
        return self.T_2D_raw(*x)
    
    def T_2D(self, x0, x1):
        T = self.T_2D_raw(x0, x1) - self.T_min
        return T
    
    def find_minima(self, x_guess = np.array([0.1, 0.1])):
        res = minimize(self.T_2D_raw_wrap, x_guess)
        return res.x
       
        
class NFWLens(LensModel):
    
    def __init__(self, kappa, y0, y1 = 0, x_min = None):
        self.kappa = kappa        
        super().__init__(y0, y1, x_min)
        Psi, x, k = sp.symbols(('\Psi', 'x', '\kappa'))
        Psi_expr_low = k / 2 * (sp.log(sp.Abs(x)/2)**2 - sp.atanh(sp.sqrt(1-x**2))**2)
        Psi_expr_hi = k / 2 * (sp.log(sp.Abs(x)/2)**2 + sp.atan(sp.sqrt(1-x**2))**2)
        self.Psi_expr_list = [Psi_expr_low, Psi_expr_hi]
        self.Psi_piecewise_break = [1]
    
    def Psi_1D(self, x):
        with np.errstate(invalid = "ignore"):
            Psi = np.where(np.abs(x)<1,
                         self.kappa / 2 * (np.log(np.abs(x)/2)**2 - np.arctanh(np.sqrt(1-x**2))**2),
                         self.kappa / 2 * (np.log(np.abs(x)/2)**2 + np.arctan(np.sqrt(x**2 - 1))**2))
        if len(Psi.shape) == 0:
            return Psi.item()
        else:
            return Psi
    
    def Psi_2D(self, x0, x1):
        x2 = x0**2 + x1**2
        x = np.sqrt(x2)
        with np.errstate(invalid = "ignore"):
            if np.all(x <= 1):
                Psi = self.kappa / 2 * (np.log(np.abs(x)/2)**2 - np.arctanh(np.sqrt(1-x2))**2)
            elif np.all(x >= 1):
                Psi = self.kappa / 2 * (np.log(np.abs(x)/2)**2 + np.arctan(np.sqrt(x2 - 1))**2)
            else:
                Psi = np.where(np.abs(x)<1,
                             self.kappa / 2 * (np.log(np.abs(x)/2)**2 - np.arctanh(np.sqrt(1-x2))**2),
                             self.kappa / 2 * (np.log(np.abs(x)/2)**2 + np.arctan(np.sqrt(x2 - 1))**2))
        return Psi
    
    
