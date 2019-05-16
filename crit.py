import sys
import numpy as np
import scipy.special as sp
from scipy.optimize import root
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, font_manager
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import f as f

##################################################################### 
################ Parameter Initialization ###########################
num     = 500           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
kyrho   = np.logspace(-2.0,2.0,num) # k_prp*rho_i
kxH     = np.logspace(-1.0,3.0,num) # k_prp*rho_i
fzeta   = np.ones((num,num),dtype=complex)  # store zeta
dlnp_dlnt = 1.0
######################################################################
################### Main Code Starts Here ############################
for i in range(0,num):
  print 'i=',i
  al = kyrho[i]**2/2.0
  kxH[i] = kyrho[i]/dlnp_dlnt*\
     np.sqrt((1.5-dlnp_dlnt-al*f.dive(0,al,1)/sp.ive(0,al))\
     /((2.0*(1.0+tau)/sp.ive(0,al)-1.0)**2-1.0))

######################## Plot the results #############################
plt.loglog(kxH,kyrho,'k-',linewidth=2)
#plt.loglog(kyrho,-kxH,'k--',linewidth=2)

plt.legend(loc=0,prop={'size':20})
plt.xlabel(r'$k_y \rho_i$')
plt.ylabel(r'$ k_\parallel H$')

plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

font = {'family' : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight' : 'light',
        'size'   : 24}
matplotlib.rc('font', **font)

#plt.axis([1e-2,1e2,1e-2,1e1])
fig = plt.gcf()
#fig.set_size_inches(8, 5)
plt.savefig('temp.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


