# This is a python Module, calculating dielectric magnetic tensor
# the dielectric tensor here is the stratified case for k_perp = 0
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
    kh = np.inf
    kh_kt = 0
  elif ind==1:
    kh = krho[0]/rho_H[0]
    kt = 3.0*kh
    kh_kt = 1.0/3.0
  elif ind==2:
    kh = krho[0]/rho_H[0]
    kt = -kh
    kh_kt = -1.0

  D = np.zeros((3,3),dtype=complex)

  rkd    = [1.0,(kd[1]/kd[0])**2]
  # sum over species
  for k in range(0,2):
     vd = rho_H[k]/2.0
     z0 = ze[k]+vd*ky_kx
     omp = min(1,ind)*ky_kx/(2.0*ze[k])*rho_H[k]
     omt = -omp*(kh_kt)

     D[0,0] += 2*ze[k]*f.p(z0,2)/rkd[k]
     D[0,1] += 2*ze[k]*(-vd*f.p(z0,1)\
               +j*np.sin(phi)*kprho[k]/2.0*f.p(z0,1))/rkd[k]
     D[0,2] += 2*ze[k]*(-j*np.cos(phi)*kprho[k]/2.0*f.p(z0,1))/rkd[k]
     D[1,0] += 2*ze[k]*(-vd*f.p(z0,1)\
               -j*np.sin(phi)*kprho[k]/2.0*f.p(z0,1))/rkd[k]
     D[1,1] += 2*ze[k]*(kxrho[k]**2/2.0*z0\
               +(vd**2+np.sin(phi)**2*kprho[k]**2/2.0)*f.p(z0,0))/rkd[k]
     D[1,2] += 2*ze[k]*(j*kxrho[k]/2.0\
               -j*np.cos(phi)*kprho[k]/2.0*vd*f.p(z0,0)-\
               kprho[k]**2/4.0*np.sin(2*phi)*f.p(z0,0))/rkd[k]
     D[2,0] += 2*ze[k]*(j*np.cos(phi)*kprho[k]/2.0*f.p(z0,1))/rkd[k]
     D[2,1] += 2*ze[k]*(-j*kxrho[k]/2.0\
               +j*np.cos(phi)*kprho[k]/2.0*vd*f.p(z0,0)-\
               kprho[k]**2/4.0*np.sin(2*phi)*f.p(z0,0))/rkd[k]
     D[2,2] += 2*ze[k]*(np.cos(phi)**2*kprho[k]**2/2.0*f.p(z0,0)+\
               kxrho[k]**2/2.0*z0)/rkd[k]
     ####
     D[1,1] += beta[k]/(2*kh*kt)*kd[0]**2
  print D
  D += np.matrix(D4)*kd[0]**2
  print D
  print ze[0]
  print '1',D[0,0]*(D[1,1]*D[2,2]-D[1,2]*D[2,1])
  print '2',-D[0,1]*(D[1,0]*D[2,2]-D[1,2]*D[2,0])
  print '3',D[0,2]*(D[1,0]*D[2,1]-D[1,1]*D[2,0])
  print 'yy*zz',D[1,1]*D[2,2]
  print 'yz*zy',D[1,2]*D[2,1]
  print 'yx*zz',D[1,0]*D[2,2]
  print 'yz*zx',D[1,2]*D[2,0]
  fz    = np.linalg.det(D)/ze[0]**2
  return (fz.real,fz.imag)

