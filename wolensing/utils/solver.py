import numpy as np

def newtons_method(T, x, y):
    Hessian(x, y).T * derivative(x, y)
