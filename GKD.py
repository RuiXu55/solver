# This file calculate dispersion relation and return its real part and 
# imaginary part separately. 

import f
import sys
import numpy as np
import scipy.special as sp

def SOLVE(zeta0,*data):
  beta0,kxr0,kpr0,tau,z,mu,rho_H0,ind,phi = data
  d_H0  = rho_H0/np.sqrt(beta0)
  kxd   = kxr0/np.sqrt(beta0)
  kpd   = kpr0/np.sqrt(beta0)

  beta  = [beta0,beta0/tau]             # Ion beta, electron beta 
  d_H   = [d_H0,-d_H0*np.sqrt(mu*z)]    # d_H for ion and e-
  # zeta_{ion} and zeta_{electron} respectively
  ze    = [complex(zeta0[0],zeta0[1]),\
          complex(zeta0[0],zeta0[1])*np.sqrt(mu*tau)]

  # kh=-k/(dlnP/dz),kt=-k/(dlnT/dz),kn=-k/(dlnn/dz)
  if ind == 0:  # Homogeneous Plasma
    d_H[0] = 0
    d_H[1] = 0
    kh = kt= kn  = 0
  elif ind==1:  # Stratified Plasma with dlnT/dz<0
    kh   = np.sqrt(kxd**2+kpd**2)/d_H[0]
    kt   = 3.0*kh
    #kt   = 1.*kh
  elif ind==2:  # Stratified Plasma with dlnT/dz>0
    kh   = np.sqrt(kxd**2+kpd**2)/d_H[0] 
    kt   = -kh
    kn   = 0.5*kh
  # Initialize those A,B,C etc in the dispersion relation
  A1 = B1 = C1 = D1 = E1 = F1 = 0
  # Different prefix for each term
  # First is ion which is always zero and second is electron
  pre1  = [1.0, tau/z]
  pre2  = [1.0, -1.0]
  pre3  = [1.0, z/tau]
  # Sum over ions and electrons,where k is species index
  for k in range(0,2):
      # zeta here taking into account of drift velocity
      zeta  = ze[k] + d_H[k]*np.sqrt(beta[k])/2.0*kpd*np.cos(phi)/kxd
      # a = (k_perp*rho_i)**2/2.0
      if k ==0: # sum over ion
        a   = kpd**2*beta[0]/2.0
      elif k==1: # sum over electron
        a   = (kpd*np.sqrt(mu*z))**2*beta[1]/2.0
      # omp = omega_ps/omega,omt = omega_ts/omega,omn=omega_ns/omega
      if ind==0:
        omp = omt = omn =0
      else:
        omp = kpd*np.cos(phi)/kxd/(2.0*ze[k])*np.sqrt(beta[k])*d_H[k]
        omt = -kpd*np.cos(phi)/kxd/(2.0*ze[k])*np.sqrt(beta[k])*d_H[k]*(kh/kt)
      # sp.ive(n,a_s)= I_n(a_s)*exp(-a_s)
      A1   += pre1[k]*(1.0+ze[k]*(f.p(zeta,0)*sp.ive(0,a)*(1.0-1.5*omt)+\
              omt*(f.p(zeta,2)*sp.ive(0,a)+a*f.p(zeta,0)*f.dive(0,a,1))))
      B1   += pre1[k]*(1.0-sp.ive(0,a)*(1.0-omt)-omt*a*f.dive(0,a,1)\
              -omp*ze[k]*(f.p(zeta,0)*sp.ive(0,a)*(1.0-1.5*omt)+\
              omt*(f.p(zeta,2)*sp.ive(0,a)+a*f.p(zeta,0)*f.dive(0,a,1))))
      C1   += -pre2[k]*ze[k]*(f.p(zeta,0)*f.dive(0,a,1)*(1-0.5*omt)+\
              omt*(f.p(zeta,2)*f.dive(0,a,1)+a*f.p(zeta,0)*f.dive(0,a,2))) 
      D1   += -pre3[k]*2.*ze[k]*(f.p(zeta,0)*f.dive(0,a,1)*(1+0.5*omt)+\
              omt*(f.p(zeta,2)*f.dive(0,a,1)+a*f.dive(0,a,2)*f.p(zeta,0)))
      #print 'ze',2.0*ze[k],'z0',f.p(zeta,0),'g.',f.dive(0,a,1),'omt',omt,'z2/z0',f.p(zeta,2)/f.p(zeta,0)
      #print 'g2g',a*f.dive(0,a,2)/f.dive(0,a,1)

      E1   += -pre2[k]*(f.dive(0,a,1)+omt*a*f.dive(0,a,2)+\
              ze[k]*omp*(f.p(zeta,0)*f.dive(0,a,1)*(1-0.5*omt)+\
              omt*(f.p(zeta,2)*f.dive(0,a,1)+a*f.p(zeta,0)*f.dive(0,a,2))))
      F1   += pre1[k]*omp*(f.p(zeta,1)*sp.ive(0,a)*(1-1.5*omt)+\
              omt*(f.p(zeta,3)*sp.ive(0,a)+ a*f.p(zeta,1)*f.dive(0,a,1))) 
  
  # The dispersion relation is caculated the way same from the paper 
  #D1 = 0.0
  ombar = ze[0]*np.sqrt(beta[0])
  x1 = (kpd**2*beta[0]/2.0)*A1/ombar**2-A1*F1-A1*B1+B1**2
  x2 = 2.0*A1/beta[0]-A1*D1+C1**2
  x3 = (A1*E1+B1*C1)**2
  fz = (x1*x2 - x3)#*beta[0]
  #print 'A1',A1,'B1',B1,'C1',C1,'D1',D1,'E1',E1,'F1',F1
  #print 'x1',x1,'x2',x2,'x3',x3,'fz',fz

  return (fz.real, fz.imag)

  D = np.zeros((3,3),dtype=complex)
  D[0,0] = A1
  D[0,1] = A1-B1
  D[0,2] = C1
  D[1,0] = A1-B1
  D[1,1] = A1-B1+F1-(kpd**2*beta[0])/2.0/ombar**2
  D[1,2] = C1+E1
  D[2,0] = C1
  D[2,1] = C1+E1
  D[2,2] = D1-2.0/beta[0]
  fz1    = np.linalg.det(D)#/ze[0]**3



def SOLVE2(zeta0,*data):
  beta,kxr,kpr,tau,z,mu,rho_H,ind,phi = data
  ze    = [complex(zeta0[0],zeta0[1]),\
          complex(zeta0[0],zeta0[1])*np.sqrt(mu*tau)]

  omp = kpr/kxr/(2.0*ze[0])*rho_H
  omt = kpr/kxr/(2.0*ze[0])*rho_H*1.0/3.0
  fz = complex(1.,1.)*ze[0]**(3/2.0)*(np.sqrt(np.pi*mu)/2.0*omt**2*
      (omt/2.0-omp))**(1./2.)-ze[0]

  return (fz.real, fz.imag)
