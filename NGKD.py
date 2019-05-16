import f
import sys
import time
import numpy as np
import scipy.special as sp

def SOLVE(zeta0,*data):
  kxr,kpr,tau,z,mu,rho_H0,phi = data
  rho_H   = [rho_H0,-rho_H0*np.sqrt(mu*z)]    # d_H for ion and e-
  ze    = [complex(zeta0[0],zeta0[1]),\
          complex(zeta0[0],zeta0[1])*np.sqrt(mu*tau)]
  kh   = np.sqrt(kxr**2+kpr**2)/rho_H[0]
  A1 = 0.0
  for k in range(0,1):
    zeta  = ze[k] + rho_H[k]/2.0*kpr/kxr
    #print 'zeta=',zeta,'ze',ze[k]
    if k ==0: # sum over ion
      a   = kpr**2/2.0
    elif k==1: # sum over electron
      a   = (kpr*np.sqrt(mu*z))**2/2.0

    omt = kpr/kxr/(2.0*ze[k])*rho_H[k]
    A1   += 2.0+ze[k]*(f.p(zeta,0)*sp.ive(0,a)*(1.0+1.5*omt)-\
            omt*(f.p(zeta,2)*sp.ive(0,a)+a*f.p(zeta,0)*f.dive(0,a,1)))
  fz = A1
  return (fz.real, fz.imag)

def SOLVE1(zeta0,*data):
  kxr,kpr,tau,z,mu,rho_H,phi,dlnt_dlnp = data
  ze   = complex(zeta0[0],zeta0[1])
  a    = kpr**2/2.0
  omt  = -kpr/kxr/2.0*rho_H*dlnt_dlnp
  omp  = -kpr/kxr/2.0*rho_H
  fz   = (ze+omp)**3*(tau+1.0-sp.ive(0,a))-(ze+omp-omt)*sp.ive(0,a)/2.0-\
         sp.ive(0,a)*((ze+omp)**2+0.5)*(-omp+omt*(1.-a*f.dive(0,a,1)/sp.ive(0,a)))
  return (fz.real, fz.imag)

def SOLVE3(zeta0,*data):
  kxr,kpr,tau,z,mu,rho_H0,phi,dlnt_dlnp = data
  rho_H   = [rho_H0,-rho_H0*np.sqrt(mu*z)]    # d_H for ion and e-
  ze    = [complex(zeta0[0],zeta0[1]),\
          complex(zeta0[0],zeta0[1])*np.sqrt(mu*tau)]
  fz = 0.0
  for k in range(0,2):
    a    = kpr**2/2.0
    if k==1:
      a   = (kpr*np.sqrt(mu*z))**2/2.0
    omt  = -kpr/kxr/2.0*rho_H[k]*dlnt_dlnp
    omp  = -kpr/kxr/2.0*rho_H[k]
    zeta = ze[k] + omp
    z2 = f.p(zeta,2)
    z0 = f.p(zeta,0)
    fz += 1.0+z0*sp.ive(0,a)*(ze[k]+1.5*omt)-\
            omt*(z2*sp.ive(0,a)+a*z0*f.dive(0,a,1))
  return (fz.real, fz.imag)


def SOLVE4(zeta0,*data):
  beta0,kxr,kpr,tau,z,mu,rho_H0,ind,phi = data

  beta  = [beta0,beta0/tau]             # Ion beta, electron beta 
  rho_H = [rho_H0,-rho_H0*np.sqrt(mu*z)]    # d_H for ion and e-
  ze    = [complex(zeta0[0],zeta0[1]),\
          complex(zeta0[0],zeta0[1])*np.sqrt(mu*tau)]
  A1 = B1 = C1 = D1 = E1 = F1 = 0
  pre1  = [1.0, tau/z]
  pre2  = [1.0, -1.0]
  pre3  = [1.0, z/tau]
  # Sum over ions and electrons,where k is species index
  for k in range(0,2):
      # zeta here taking into account of drift velocity
      omp = kpr/kxr/(2.0*ze[k])*rho_H[k]
      omt = -kpr/kxr/(2.0*ze[k])*rho_H[k]*(1.0/3.)
      zeta  = ze[k]*(1.0+omp)
      #print 'zeta',zeta
      # a = (k_perp*rho_i)**2/2.0
      if k ==0: # sum over ion
        a   = kpr**2/2.0
      elif k==1: # sum over electron
        a   = (kpr*np.sqrt(mu*z))**2/2.0
      # omp = omega_ps/omega,omt = omega_ts/omega,omn=omega_ns/omega
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
      E1   += -pre2[k]*(f.dive(0,a,1)+omt*a*f.dive(0,a,2)+\
              ze[k]*omp*(f.p(zeta,0)*f.dive(0,a,1)*(1-0.5*omt)+\
              omt*(f.p(zeta,2)*f.dive(0,a,1)+a*f.p(zeta,0)*f.dive(0,a,2))))
      F1   += pre1[k]*omp*(f.p(zeta,1)*sp.ive(0,a)*(1-1.5*omt)+\
              omt*(f.p(zeta,3)*sp.ive(0,a)+ a*f.p(zeta,1)*f.dive(0,a,1))) 
  
  # The dispersion relation is caculated the way same from the paper 
  ombar = ze[0]*np.sqrt(beta[0])
  x1 = (kpr**2/2.0)*A1/ombar**2-A1*F1-A1*B1+B1**2
  x2 = 2.0*A1/beta[0]-A1*D1+C1**2
  x3 = (A1*E1+B1*C1)**2
  fz = (x1*x2 - x3)
  #fz = A1
  return (fz.real, fz.imag)

