import NGKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc, font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
import f
import scipy.special as sp

################ Parameter Initialization ###########################
num     = 50           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
ind     = 1
beta0 = np.logspace(2,4,100)
for m in range(0,100):
  beta    = beta0[m]
  d_H     = 1.e-5
  rho_H   = d_H*np.sqrt(beta)

  kprho   = np.logspace(-3.0,2.0,num) # k_prp*rho_i
  fzeta   = np.ones((2*num,num),dtype=complex)  # store zeta
  fz      = np.ones((2*num,num),dtype=complex)  # store zeta
  tf      = np.ones(num,dtype=complex)

  maxgrowth = 0.0
  maxkprho = 0.0
  maxkxrho = 0.0

  dlnt_dlnp=1.0/3.0
  ######################################################################
  ################### Main Code Starts Here ############################

  kxrho  = np.logspace(-4,-6,num)
  for j in range(0,num):
    print( 'j=',j)
    for i in range(0,num):
      if i==0 and j==0:
        zeta = complex(1e-3,0.1)
      elif j==0:
        zeta = fz[j,i-1]
        #if kprho[i]>40:
        #  zeta = tf[i]
      else:
        zeta = fz[j-1,i]
      if m>0:
        zeta = fz[j,i]

      data  = (beta,kxrho[j],-kprho[i],tau,z,mu,rho_H,ind,phi)
      sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-6)
      fz[j,i] = complex(sol.x[0],sol.x[1])

      fzeta[num-j-1,i] = fz[j,i]*kxrho[j]/rho_H
      if fzeta[num-j-1,i].imag>maxgrowth and fzeta[num-j-1,i].imag<20.0:
        maxgrowth = fzeta[num-j-1,i].imag
        maxkxrho = kxrho[j]
        maxkprho = kprho[i]
      print 'fz',fzeta[num-j-1,i],kprho[i]

  kxrho  = np.logspace(-4,-2,num)
  for j in range(0,num):
    print( 'j=',j)
    for i in range(0,num):
      if i==0 and j==0:
        zeta = complex(1e-4,0.4)
      elif j==0:
        zeta = fz[j,i-1]
      else:
        zeta = fz[j-1,i]
      if m>0:
        zeta = fz[j,i]

      data  = (beta,kxrho[j],-kprho[i],tau,z,mu,rho_H,ind,phi)
      sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
      fz[j,i] = complex(sol.x[0],sol.x[1])
      fzeta[num+j,i] = fz[j,i]*kxrho[j]/rho_H
      if fzeta[num+j,i].imag>maxgrowth and fzeta[num+j,i].imag<20:
        maxgrowth = fzeta[num+j,i].imag
        maxkxrho = kxrho[j]
        maxkprho = kprho[i]
        print 'fz',fzeta[num+j,i]

  ######################################################################
  ######################## Plot the results #############################
  extents = (1e-6/rho_H,1e-2/rho_H,kprho[0],kprho[-1])
  cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
  #cmap = cm.get_cmap('viridis')    # 11 discrete colors

  #plt.loglog(kxH,1,'w--',linewidth=1.5)
  xposition = 5.0
  plt.axvline(xposition,color='r',linestyle='--',linewidth=1.5)
  plt.axis(extents)

  fz = np.ma.masked_where(abs(fzeta.imag<0),fzeta.imag)
  Z = fz.data.transpose()*np.sqrt(3.0)
  print 'maxgrowh=',maxgrowth*np.sqrt(3.0)
  print 'maxkxrho=',maxkxrho
  print 'maxkprho=',maxkprho
  myplt = plt.imshow(Z,extent= extents,cmap=cmap,
        aspect='auto',origin='lower',norm=LogNorm(),vmin=0.1,vmax=10.0)
  plt.xscale('log')
  plt.yscale('log')

  sizeOfFont = 12
  fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
          'weight' : 'normal', 'size' : sizeOfFont}
  ticks_font = font_manager.FontProperties(family='Helvetica', style='normal',
          size=sizeOfFont, weight='normal', stretch='normal')
  rc('text', usetex=True)
  rc('font',**fontProperties)
  font = {'size'  : 24}
  rc('font', **font)
  plt.xlabel(r'$k_\parallel H$')
  plt.ylabel(r'$k_y\rho_i$')
  plt.tick_params(pad=10,direction ='in')
  plt.tick_params(axis='y',direction='out',which='both')
  plt.tick_params(axis='x',direction='out',which='both')
  plt.tick_params(length=8, width=1, which='major')
  plt.tick_params(length=5, width=1, which='minor')

  ax = plt.gca()
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="8%", pad=0.2)
  cbar = plt.colorbar(myplt, cax=cax)

  fig = plt.gcf()
  #plt.gca().tight_layout()
  #plt.savefig('mticmp1.eps',format='eps',dpi=300,bbox_inches='tight')
  plt.show()
