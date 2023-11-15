import numpy as np

def handleScalarInput(variable, func):
    if isinstance(variable, np.ndarray):
        return func(variable)
    if isinstance(variable, list):
        return func(np.array(variable))
    return func(np.array([variable]))[0]