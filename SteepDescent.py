"""
for Steepest Descent Algorithm
"""

from functions.py import recurse, gradient
import numpy as np

recurse_time: int = 0
x0 = 0.0
f = lambda x: x**2

result = recurse(time = recurse_time, s = -gradient(f, x0), x0, f)
 
