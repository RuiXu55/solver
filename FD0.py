import f
import sys
import scipy
import numpy as np
import scipy.special as sp

def SOLVE(zi0,*data):
  beta0,kxr0,kpr0,tau,z,mu,rho_H0,ind,phi,m= data
  # zeta_{ion} and zeta_{electron} respectively
  # first column is for ions, second colum is e-
  ze     = [complex(zi0[0],zi0[1]),\
           complex(zi0[0],zi0[1])*np.sqrt(mu*tau)]  # omega/(k_prl*vths)
  
  d_H0   = rho_H0/np.sqrt(beta0)
  kxd0   = kxr0/np.sqrt(beta0)
  kpd0   = kpr0/np.sqrt(beta0)

  kxd    = [kxd0, -kxd0*np.sqrt(mu*z)] # k_prl*skin_depth
  kpd    = [kpd0, -kpd0*np.sqrt(mu*z)] # k_perp*skin_depth
  beta   = [beta0, beta0/tau]          # plasma beta=8*pi*P/B**2
  d_H    = [d_H0, -d_H0*np.sqrt(mu*z)] # skin_depth/scale_height
  kd     = [np.sqrt(kxd[0]**2+kpd[0]**2),\
           -np.sqrt(kxd[1]**2+kpd[1]**2)] # k*skin_depth

  # xp = k_perp/k_tot, x = k_prl/k_tot, ky = k_perp*cos(phi)
  xp     = kpd[0]/kd[0]
  x      = kxd[0]/kd[0]

  if ind ==0:     # Homogeneous Plasma
    d_H[0]=0
    d_H[1]=0
    kh =kt = 1e100
    kh_o_kt = 0 
  elif ind ==1:   # Stratified Plasma with dlnT/dz<0
    kh  = kd[0]/d_H[0]
    kh_o_kt = 1.0/3.0
    kt = kh/kh_o_kt
  elif ind ==2:   # Stratified Plasma with dlnT/dz>0
    kh  = kd[0]/d_H[0]
    kh_o_kt  = -1
    kt = kh/kh_o_kt

  # Initialize the dielectric tensor and its derivative
  j      = complex(0,1.0)    # use as a imaginary number
  D      = np.zeros((3,3),dtype=complex) 
  xx = complex(0,0)
  pre1  = [1.0, tau/z]
  pre2  = [1.0, -1.0]
  pre3  = [1.0, z/tau]

  # Sum over different species(ion+e-)
  for k in range(0,2):
    # Initialize the temporary dielectric tensor and its derivative
    # a = (k_perp*rho_i)**2/2.0
    vd = d_H[k]*np.sqrt(beta[k])/2.0
    kprp = kpd[k]*np.sqrt(beta[k])
    kprl = kxd[k]*np.sqrt(beta[k])
    a = kprp**2/2.0
    # summation over modified bessel function n.
    for n in range(-m,m+1): 
      kyvd = vd*xp/x*np.cos(phi)
      zz   = ze[k] + kyvd - n/kprl
      # omt = omega_ts/omega
      omt  = -xp*np.cos(phi)/x/ze[k]*vd*kh_o_kt
      omp  = xp*np.cos(phi)/x/ze[k]*vd

      # tensor component and derivative for different n & k
      D[0,0] += pre1[k]*(ze[k]*(f.p(zz,0)*sp.ive(n,a)*(1.0-1.5*omt)+\
                omt*f.p(zz,2)*sp.ive(n,a)+a*omt*f.p(zz,0)*f.dive(n,a,1)))

      D[0,1] += pre1[k]*((1.0+(x/xp)**2*omp-(x/xp)**2*n/(ze[k]*kprl))\
                *(f.p(zz,1)*sp.ive(n,a)*(1.0-1.5*omt)+omt*f.p(zz,3)*\
                sp.ive(n,a)+omt*a*f.p(zz,1)*f.dive(n,a,1))\
                -ze[k]*(x/xp*n/(ze[k]*kprl)-x/xp*omp)**2*(f.p(zz,0)*\
                sp.ive(n,a)*(1.0-1.5*omt)+omt*f.p(zz,2)*sp.ive(n,a)\
                +a*omt*f.p(zz,0)*f.dive(n,a,1)))

      D[0,2] += -pre2[k]*(ze[k]*(f.p(zz,0)*f.dive(n,a,1)*(1.0-0.5*omt)\
                +omt*f.p(zz,2)*f.dive(n,a,1)+a*omt*f.p(zz,0)*f.dive(n,a,2))) 

      D[1,0] += pre1[k]*(f.p(zz,1)*sp.ive(n,a)*(1.0-1.5*omt)\
                +omt*f.p(zz,3)*sp.ive(n,a)+omt*a*f.p(zz,1)*f.dive(n,a,1)) 

      D[1,1] += pre1[k]*((1.0+omp/xp**2-n/(xp**2*ze[k]*kprl))\
                *(f.p(zz,1)*sp.ive(n,a)*(1.0-1.5*omt)+omt*f.p(zz,3)*sp.ive(n,a)\
                +a*omt*f.p(zz,1)*f.dive(n,a,1)))

      D[1,2] += -pre2[k]*(f.p(zz,1)*f.dive(n,a,1)*(1.0-0.5*omt)\
                +omt*f.p(zz,3)*f.dive(n,a,1)+a*omt*f.p(zz,1)*f.dive(n,a,2))

      D[2,0] += -pre2[k]*(ze[k]*(f.p(zz,0)*f.dive(n,a,1)*(1.0-0.5*omt)\
                +omt*f.p(zz,2)*f.dive(n,a,1)+a*omt*f.p(zz,0)*f.dive(n,a,2))) 

      D[2,1] += -pre2[k]*((ze[k]+\
                kyvd*(1-(x/xp)**2)-n/kprl*(1-(x/xp)**2))*(f.p(zz,0)*\
                f.dive(n,a,1)*(1-0.5*omt)+omt*f.p(zz,2)*f.dive(n,a,1)+\
                a*omt*f.p(zz,0)*f.dive(n,a,2)))

      D[2,2] += -pre3[k]*2.0*ze[k]*xp**2*((f.p(zz,0)*f.dive(n,a,1)*(1.0+0.5*omt)\
                +omt*f.p(zz,2)*f.dive(n,a,1)+a*omt*f.p(zz,0)*f.dive(n,a,2))\
                -n**2/(2.0*a**2)*(f.p(zz,0)*sp.ive(n,a)*(1.0-1.5*omt)\
                +omt*f.p(zz,2)*sp.ive(n,a)+a*omt*f.p(zz,0)*f.dive(n,a,1)))

    # Some extra piece without sum over n
    D[0,0] += pre1[k]
    D[0,1] += pre1[k]*(x/xp)**2*omp*omt
    # Only for ions.
    if k ==0:
      ombar = ze[k]*np.sqrt(beta[k])
      D[1,1] += -a/ombar**2/xp**2
      D[2,2] += -2.0/beta[k]
  fz    = scipy.linalg.det(D)
  return fz.real**2+fz.imag**2


