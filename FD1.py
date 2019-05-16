import f
import sys
import scipy
import numpy as np
import scipy.special as sp

def SOLVE(z0,*data):
  b0,kpr0,tau,z,mu,r_H0,ind,phi,m=data
  # first column is for ions, second colum is e-
  ze     = [complex(z0[0],z0[1]),\
           complex(z0[0],z0[1])*np.sqrt(mu*tau)]  # omega/(k_prl*vths)
  krho   = [kpr0, -kpr0*np.sqrt(mu*z/tau)]
  rho_H  = [r_H0, -r_H0*np.sqrt(mu*z/tau)]

  beta   = [b0, b0/tau]          # plasma beta=8*pi*P/B**2

  D4     = [[-1.0, 0.0, 0.0],
            [0.0, np.cos(phi)**2-1.0, np.cos(phi)*np.sin(phi)],
            [0.0, np.cos(phi)*np.sin(phi),np.sin(phi)**2-1.0]]

  if ind ==0:     # Homogeneous Plasma
    rho_H[0]=rho_H[1]=0
    kh = np.inf
    kh_kt = 0 
  elif ind ==1:   # Stratified Plasma with dlnT/dz<0
    kh  = krho[0]/rho_H[0]
    kh_kt = 1.0/3.0
  elif ind ==2:   # Stratified Plasma with dlnT/dz>0
    kh  = krho[0]/rho_H[0]
    kh_kt  = -1
  # Initialize the dielectric tensor and its derivative
  j      = complex(0,1.0)    # use as a imaginary number
  D      = np.zeros((3,3),dtype=complex)
  FF     = np.zeros((2,2),dtype=complex)
  # Sum over different species(ion+e-)
  y1 =y2=y3=0
  for k in range(0,2):
    # Initialize the temporary dielectric tensor and its derivative
    D1    = np.zeros((3,3),dtype=complex)
    x1 =x2=x3= 0
    vd = rho_H[k]/2.0
    a = krho[k]**2/2.0
    # summation over modified bessel function n.
    # omt = omega_ts/omega
    omt  = -np.cos(phi)*vd/ze[k]*kh_kt
    for n in range(-m,m+1): 
      rat  = 2.0*ze[k]/(ze[k]+np.cos(phi)*vd-n/krho[k])
      # tensor component and derivative for different n & k
      #D1[0,0]  += rat*(0.5*(sp.ive(n,a)+a*omt*f.dive(n,a,1)))
      #D1[0,1]  += 0.0
      #D1[0,2]  += 0.0
      #D1[1,0]  += 0.0
      D1[1,1]  += rat*(((n*np.cos(phi)/krho[k]-vd)**2 + \
                  n**2*np.sin(phi)**2/(2.0*a))\
                  *(sp.ive(n,a)*(1.-omt)+a*omt*f.dive(n,a,1))\
                  -np.sin(phi)**2*a*(f.dive(n,a,1)*\
                  (1.0+omt)+a*omt*f.dive(n,a,2)))

      D1[1,2]  += rat*(j*(n/2.0-0*np.cos(phi)*krho[k]/2.0*vd)\
                  *(f.dive(n,a,1)+a*omt*f.dive(n,a,2)))
                  #-n*np.sin(phi)/krho[k]*vd*\
                  #(sp.ive(n,a)*(1-omt)+a*omt*f.dive(n,a,1)))
                  #+0.5*np.sin(2.0*phi)*a*(f.dive(n,a,1)*(1.+omt)\
                  #+a*omt*f.dive(n,a,2)))
      

      D1[2,1]  += rat*(-j*(n/2.0-0*np.cos(phi)*krho[k]/2.0*vd)\
                  *(f.dive(n,a,1)+a*omt*f.dive(n,a,2)))
                  #-n*np.sin(phi)/krho[k]*vd*\
                  #(sp.ive(n,a)*(1-omt)+a*omt*f.dive(n,a,1)))
                  #+0.5*np.sin(2.0*phi)*a*(f.dive(n,a,1)*(1.+omt)\
                  #+a*omt*f.dive(n,a,2)))
      D1[2,2]  += rat*(n**2/2.0/a*(sp.ive(n,a)*(1-omt)+a*omt*f.dive(n,a,1))\
                  -np.cos(phi)**2*a*(f.dive(n,a,1)*(1+omt)+a*omt*f.dive(n,a,2)))
    FF[0,0] += D1[1,1]/(krho[k]**2/beta[k])
    FF[0,1] += D1[1,2]/(krho[k]**2/beta[k])
    FF[1,0] += D1[2,1]/(krho[k]**2/beta[k])
    FF[1,1] += D1[2,2]/(krho[k]**2/beta[k])
    y1   += x1/(krho[k]**2/beta[k])
    y2   += x2/(krho[k]**2/beta[k])
    if ind>0:  # inhomogeneous plasma
      FF[0,0] -= beta[k]/(2.0*kh**2/kh_kt)
  D4 = np.matrix(D4)
  FF[0,0] -=D4[1,1]
  FF[0,1] -=D4[1,2]
  FF[1,0] -=D4[2,1]
  FF[1,1] -=D4[2,2]
  '''
  print 'y2',y2
  print 'kd',krho[0]/np.sqrt(beta[0])
  print 'omt',omt
  print 'rat',2.0*ze[k]/(ze[k]+np.cos(phi)*vd)
  sys.exit()
  '''
  #fz    = scipy.linalg.det(D)#/ze[0]**2
  fz    = scipy.linalg.det(FF)#/ze[0]**2
  return (fz.real, fz.imag)

def SOLVE(z0,*data):
  b0,kpr0,tau,z,mu,r_H0,ind,phi,m=data
  # first column is for ions, second colum is e-
  ze     = [complex(z0[0],z0[1]),\
           complex(z0[0],z0[1])*np.sqrt(mu*tau)]  # omega/(k_prl*vths)
  krho   = [kpr0, -kpr0*np.sqrt(mu*z/tau)]
  rho_H  = [r_H0, -r_H0*np.sqrt(mu*z/tau)]
  beta   = [b0, b0/tau]          # plasma beta=8*pi*P/B**2

  kd   = [kpr0/np.sqrt(b0), -kpr0*np.sqrt(mu*z/tau)/np.sqrt(b0)]
  kh  = krho[0]/rho_H[0]
  kh_kt  = -1
  j      = complex(0,1.0)    # use as a imaginary number
  D1    = np.zeros((2,2),dtype=complex)
  for k in range(0,2):
    vd = rho_H[k]/2.0
    a = krho[k]**2/2.0
    omt  = -vd/ze[k]*kh_kt
    rat  = 2.0*ze[k]/(ze[k]+vd)
    D1[0,0] += -1.0/kd[k]**2*(ze[k]*vd*8.0*(sp.ive(1,a)*(1.-omt)+\
               a*omt*f.dive(1,a,1))+vd**2*rat*(sp.ive(0,a)*(1.-omt)+a*omt*f.dive(0,a,1))-\
               2.*ze[k]*(ze[k]+vd)/a/krho[k]**2*(sp.ive(1,a)*(1.-omt)+\
               a*omt*f.dive(0,a,1)))+beta[k]/(2.0*kh**2/kh_kt)

    D1[0,1] += 1.0/kd[k]**2*rat*vd*(f.dive(1,a,1)+a*omt*f.dive(1,a,2))\
               +j/kd[k]**2*2.*ze[k]*krho[k]*(f.dive(1,a,1)+a*omt*f.dive(1,a,2))

    D1[1,0] += 1.0/kd[k]**2*rat*vd*(f.dive(0,a,1)+a*omt*f.dive(0,a,2))\
               -j/kd[k]**2*2.*ze[k]*krho[k]*(f.dive(1,a,1)+a*omt*f.dive(1,a,2))
    D1[1,1] += 0.5+1.0/kd[k]**2*a*rat*(f.dive(0,a,1)*(1.+omt)+a*omt*f.dive(0,a,2))

  fz    = scipy.linalg.det(D1)/ze[0]
  return (fz.real, fz.imag)

