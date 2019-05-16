# This is a python Module, calculating dielectric magnetic tensor
# the dielectric tensor here is the stratified case for k_perp = 0
# Written by Rui Xu. Dec. 2015

import f
import sys
import numpy as np
def SOLVE(z0,*data):
  b0,krho0,tau,z,mu,rH0,ind = data
  # first for ion, second for electron
  beta  = [b0,b0/tau]   # Ion beta, electron beta 
  rho_H = [rH0,-rH0*np.sqrt(mu*z/tau)]
  # zeta_{ion} and zeta_{electron} respectively
  ze   = [complex(z0[0],z0[1]),\
         complex(z0[0],z0[1])*np.sqrt(mu*tau)]
  krho = [krho0,-krho0*np.sqrt(mu*z/tau)]
  kd   = [krho[0]/np.sqrt(beta[0]),\
         krho[1]/np.sqrt(beta[1])]
  j       = complex(0.0,1.0)        
  if ind==0:
    kh = np.inf
  elif ind==1:
    kh = krho[0]/rho_H[0]
    kt = 3.0*kh
  elif ind==2:
    kh = krho[0]/rho_H[0]
    kt = -kh
  D = np.zeros((3,3),dtype=complex)
  # sum over species
  for k in range(0,2):
   for n in range(-1,2,2):
      zz = ze[k]+n/krho[k]
      D[1,2] += 0.5*n*j*ze[k]*f.p(zz,0)/kd[k]**2
      D[2,1] += -0.5*n*j*ze[k]*f.p(zz,0)/kd[k]**2
      D[2,2] += -0.5*ze[k]*f.p(zz,0)/kd[k]**2

   D[0,0] += -2.0*ze[k]**2*f.p(ze[k],1)/kd[k]**2
   D[0,1] += ze[k]*rho_H[k]*f.p(ze[k],1)/kd[k]**2
   D[0,2] += 0.0
   D[1,0] += ze[k]*rho_H[k]*f.p(ze[k],1)/kd[k]**2
   D[1,1] += -beta[k]/(2.0*kh*kt)-\
             ze[k]*rho_H[k]**2/2.0*f.p(ze[k],0)/kd[k]**2
   D[2,0] += 0.0 

  D[2,2] += 1.0
  D[1,1] += D[2,2]

  fz    = np.linalg.det(D)/ze[0]**4
  return (fz.real,fz.imag)

def SOL(z0,*data):
  be,kH = data
  ze = complex(z0[0],z0[1])
  j = complex(0,1.0)
  # kH = 100 mfp
  fz = (-ze**2 +1.0/be)*(-j*ze+kH)-j*ze/(15*kH**2)-1.0/(3.0*kH)
  return (fz.real,fz.imag)

def SOL1(z0,*data):
  be,kH = data
  ze = complex(z0[0],z0[1])
  j = complex(0,1.0)
  # kH = 100 mfp
  fz = (-ze**2 +1.0/be)*(-j*ze+kH/10.0)-j*ze/(15*kH**2)-1.0/(30.0*kH)
  return (fz.real,fz.imag)

def SOL2(z0,*data):
  be,kH = data
  ze = complex(z0[0],z0[1])
  j = complex(0,1.0)
  km = kH/10.0
  fz1 = (-ze**2 +1.0/be)*(-j*ze+km*10.0)-j*ze/(15*kH**2)-10.0/3.0*km/kH**2
  fz2 = 0.6*ze*(ze+2.0/3.0*km)*(-j*ze+50.0/3.0*km)*(1.0/be-ze**2+2.0/3.0/kH**2)
  fz = fz1-fz2
  fz = fz1
  return (fz.real,fz.imag)

def SOL3(z0,*data):
  be,kH = data
  ze = complex(z0[0],z0[1])
  j = complex(0,1.0)
  km = kH/100.0
  fz1 = (-ze**2 +1.0/be)*(-j*ze+km*10.0)-j*ze/(15*kH**2)-10.0/3.0*km/kH**2
  fz2 = 0.6*ze*(ze+2.0/3.0*km)*(-j*ze+50.0/3.0*km)*(1.0/be-ze**2+2.0/3.0/kH**2)
  fz = fz1-fz2
  fz = fz1
  return (fz.real,fz.imag)


def SOL4(z0,*data):
  be,kH = data
  ze = complex(z0[0],z0[1])
  j = complex(0,1.0)
  km = kH/1.0
  fz1 = (-ze**2 +1.0/be)*(-j*ze+km*10.0)-j*ze/(15*kH**2)-10.0/3.0*km/kH**2
  fz2 = 0.6*ze*(ze+2.0/3.0*km)*(-j*ze+50.0/3.0*km)*(1.0/be-ze**2+2.0/3.0/kH**2)
  fz = fz1-fz2
  fz = fz1
  return (fz.real,fz.imag)





def SOLVE1(z0,*data):
  b0,kxr0,tau,z,mu,rH0,ind = data
  # first for ion, second for electron
  beta  = [b0,b0/tau]   # Ion beta, electron beta 
  rho_H = [rH0,-rH0*np.sqrt(mu*z/tau)]
  ze    = [complex(z0[0],z0[1]),\
          complex(z0[0],z0[1])*np.sqrt(mu*tau)]
  kxrho = [kxr0, -kxr0*np.sqrt(mu*z/tau)]
  kxH   = kxrho[0]/rho_H[0]
  kd    = [kxrho[0]/np.sqrt(beta[0]),\
          kxrho[1]/np.sqrt(beta[1])]
  j     = complex(0.0,1.0)        

  if ind==1:
    kh = kxrho[0]/rho_H[0]
    kt = 3.0*kh
    kh_kt = 1.0/3.0

  xi = [0.0,0.0]
  pre1 = [1.0,-1.0]
  fz = 0.0
  for k in range(0,2):
    xi[k] = 1.0-pre1[k]*(f.p(ze[0],1)-f.p(ze[1],1))/(f.p(ze[0],1)+f.p(ze[1],1))

  for k in range(0,2):
    fz += (ze[0]**2-1.0/beta[0])+1.0/kh**2/3.0+ze[k]*f.p(ze[k],0)*xi[k]/kh**2

  return (fz.real,fz.imag)

