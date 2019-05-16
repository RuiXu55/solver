import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, font_manager
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

x  = [1e1,1e2,1e3,1e4]
y0 =[0.86,1.09,1.60,1.87]
y1 =[0.47,0.73,0.87,0.75]
plt.scatter(x,y0)

font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : size}
matplotlib.rc('font', **font)
plt.tick_params(pad=10)

plt.show()

