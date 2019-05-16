# This is a python Module, calculating dielectric magnetic tensor
# the dielectric tensor here is the stratified case
# Written by Rui Xu. Dec. 2015
import f
import sys
import numpy as np
def SOLVE(z0,*data):
  b0,kxr0,kpr0,tau,z,mu,rH0,ind,phi = data
  # first for ion, second for electron
  beta  = [b0,b0/tau]   # Ion beta, electron beta 
  rho_H = [rH0,-rH0*np.sqrt(mu*z/tau)]
  # zeta_{ion} and zeta_{electron} respectively
  ze    = [complex(z0[0],z0[1]),\
          complex(z0[0],z0[1])*np.sqrt(mu*tau)]
  kxrho = [kxr0, -kxr0*np.sqrt(mu*z/tau)]
  kprho = [kpr0, -kpr0*np.sqrt(mu*z/tau)]
  kpH   = kprho[0]/rho_H[0]
  kzH   = kpH*np.sin(phi)
  krho  = [np.sqrt(kxr0**2+kpr0**2),\
          -np.sqrt(kxrho[1]**2+kprho[1]**2)]
  kd    = [krho[0]/np.sqrt(beta[0]),\
          krho[1]/np.sqrt(beta[1])]
  j     = complex(0.0,1.0)        

  xp    = kprho[0]/krho[0]
  x     = kxrho[0]/krho[0]
  ky_kx = xp*np.cos(phi)/x

  D4    = [[x**2-1.0,x*xp*np.cos(phi),       \
            x*xp*np.sin(phi)],
            [x*xp*np.cos(phi), xp**2*np.cos(phi)**2-1.0,\
            xp**2*np.cos(phi)*np.sin(phi)],
            [x*xp*np.sin(phi), xp**2*np.cos(phi)*np.sin(phi),\
            xp**2*np.sin(phi)**2-1.0]]

  if ind==0:
    #kh = 1e200#np.inf
    kh = krho[0]/rho_H[0]
    kt = np.inf
    kh_kt = 0.0
  elif ind==1:
    kh = krho[0]/rho_H[0]
    kt = 3.0*kh
    kh_kt = 1.0/3.0
  elif ind==2:
    kh = krho[0]/rho_H[0]
    kt = -kh
    kh_kt = -1.0

  D = np.zeros((3,3),dtype=complex)

  # sum over species
  for k in range(0,2):
     vd = rho_H[k]/2.0
     #z0 = ze[k]+vd*ky_kx
     z0  = ze[k]
     omp = min(1,ind)*ky_kx/(2.0*ze[k])*rho_H[k]
     omt = -omp*(kh_kt)

     D[0,0] += 2.*ze[k]*f.p(z0,2)/kd[k]**2
     D[0,1] += 2.*ze[k]*(-vd*f.p(z0,1)\
               +j*np.sin(phi)*kprho[k]/2.0*f.p(z0,1))/kd[k]**2
     D[0,2] += 2*ze[k]*(-j*np.cos(phi)*kprho[k]/2.0*f.p(z0,1))/kd[k]**2
     D[1,0] += 2*ze[k]*(-vd*f.p(z0,1)\
               -j*np.sin(phi)*kprho[k]/2.0*f.p(z0,1))/kd[k]**2
     D[1,1] += 2*ze[k]*(kxrho[k]**2/2.0*z0\
               +(vd**2+np.sin(phi)**2*kprho[k]**2/2.0)*f.p(z0,0))/kd[k]**2
     D[1,2] += 2*ze[k]*(j*kxrho[k]/2.0\
               +j*np.cos(phi)*kprho[k]/2.0*vd*f.p(z0,0)-\
               kprho[k]**2/2.0*np.sin(phi)*np.cos(phi)*f.p(z0,0))/kd[k]**2
     D[2,0] += 2*ze[k]*(j*np.cos(phi)*kprho[k]/2.0*f.p(z0,1))/kd[k]**2
     D[2,1] += 2*ze[k]*(-j*kxrho[k]/2.0\
               -j*np.cos(phi)*kprho[k]/2.0*vd*f.p(z0,0)-\
               kprho[k]**2/2.0*np.sin(phi)*np.cos(phi)*f.p(z0,0))/kd[k]**2
     D[2,2] += 2*ze[k]*(np.cos(phi)**2*kprho[k]**2/2.0*f.p(z0,0)+\
               kxrho[k]**2/2.0*z0)/kd[k]**2
     ####
     D[1,1] += beta[k]/(2*kh*kt)
  D += np.matrix(D4)

  fz = np.linalg.det(D)#/z0**2
  return (fz.real,fz.imag)




def SOLVE1(z0,*data):
  b0,kxr0,kpr0,tau,z,mu,rH0,ind,phi = data
  # first for ion, second for electron
  beta  = [b0,b0/tau]   # Ion beta, electron beta 
  rho_H = [rH0,-rH0*np.sqrt(mu*z/tau)]
  # zeta_{ion} and zeta_{electron} respectively
  ze    = [complex(z0[0],z0[1]),\
          complex(z0[0],z0[1])*np.sqrt(mu*tau)]
  kxrho = [kxr0, -kxr0*np.sqrt(mu*z/tau)]
  kprho = [kpr0, -kpr0*np.sqrt(mu*z/tau)]
  kxH   = kxrho[0]/rho_H[0]
  kpH   = kprho[0]/rho_H[0]
  kzH   = kpH*np.sin(phi)
  krho  = [np.sqrt(kxr0**2+kpr0**2),\
          -np.sqrt(kxrho[1]**2+kprho[1]**2)]
  kd    = [krho[0]/np.sqrt(beta[0]),\
          krho[1]/np.sqrt(beta[1])]
  j     = complex(0.0,1.0)        

  xp    = kprho[0]/krho[0]
  x     = kxrho[0]/krho[0]

  if ind==0:
    kh = krho[0]/rho_H[0]
    kt = np.inf
    kh_kt = 0.0
  elif ind==1:
    kh = krho[0]/rho_H[0]
    kt = 3.0*kh
    kh_kt = 1.0/3.0

  x1 = 0
  x2 = 0
  x3 = 0
  x4 = 0
  xi = [0.0,0.0]
  xxi = [0.0,0.0]

  pre1 = [1.0,-1.0]

  ff = (ze[0]**2-(1.0+(xp*np.sin(phi)/x)**2)/beta[0])*2.0
  for k in range(0,2):
    xi[k] = 1.0-pre1[k]*(f.p(ze[0],1)-f.p(ze[1],1))/(f.p(ze[0],1)+f.p(ze[1],1))
    xxi[k] = xi[k]+ kzH**2*(1.0+xi[k])
  
  x11 = 0.0
  x12 = 0.0
  x21 = 0.0
  for k in range(0,2):
    x1 += 1.0/kxH**2*(1.0/3.0+ze[k]*f.p(ze[k],0)*xxi[k])
    x2 += (ze[0]**2-(1.0+(xp*np.cos(phi)/x)**2)/beta[0])+\
          (xp*np.cos(phi)/x)**2*(ze[k]*f.p(ze[k],0)*(1.0+xi[k]))
    #x2 += (xp*np.cos(phi)/x)**2*(ze[k]*f.p(ze[k],0)*(1.0+xi[k]))

    #x3 += (xp*np.cos(phi)/x)**2*(ze[k]*f.p(ze[k],0)*xi[k]/kxH)**2
    #x4 += (xp**4*np.cos(phi)**2*np.sin(phi)**2)/x**4*(ze[k]*f.p(ze[k],0)*(1.0+xi[k]))**2
  fz = (ff+x1)#*x2-x3-x4
  return (fz.real,fz.imag)

