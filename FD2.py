# This file calculate dispersion relation, return its determinant
# and its derivative
import f
import math
import sys
import scipy
import numpy as np
import scipy.special as sp

def SOLVE(zi0,*data):
  beta0,d_H0,kxd0,kpd0,tau,mu,z,ind,m,phi = data
  # first column is for ions, second colum is e-
  ze     = [complex(zi0[0],zi0[1]),\
           complex(zi0[0],zi0[1])*np.sqrt(mu*tau)]  # omega/(k_prl*vths)

  kxd    = [kxd0, -kxd0*np.sqrt(mu*z)] # k_prl*skin_depth
  kpd    = [kpd0, -kpd0*np.sqrt(mu*z)] # k_perp*skin_depth
  beta   = [beta0, beta0/tau]          # plasma beta=8*pi*P/B**2
  d_H    = [d_H0, -d_H0*np.sqrt(mu*z)] # skin_depth/scale_height
  kd     = [np.sqrt(kxd[0]**2+kpd[0]**2),\
           -np.sqrt(kxd[1]**2+kpd[1]**2)] # k*skin_depth

  # xp = k_perp/k_tot, x = k_prl/k_tot, ky = k_perp*cos(phi)
  xp     = kpd[0]/kd[0]
  x      = kxd[0]/kd[0]
  D4     = [[x**2-1.0,         x*xp*np.cos(phi),       \
            x*xp*np.sin(phi)],
            [x*xp*np.cos(phi), xp**2*np.cos(phi)**2-1.0,\
            xp**2*np.cos(phi)*np.sin(phi)],
            [x*xp*np.sin(phi), xp**2*np.cos(phi)*np.sin(phi),\
             xp**2*np.sin(phi)**2-1.0]]

  if ind ==0:     # Homogeneous Plasma
    d_H[0]=0
    d_H[1]=0
    kh = kt = 1e100
    kh_o_kt = 0 
  elif ind ==1:   # Stratified Plasma with dlnT/dz<0
    kh  = kd[0]/d_H[0]
    kh_o_kt = 1.0/3.0
    kt = kh/kh_o_kt
  elif ind ==2:   # Stratified Plasma with dlnT/dz>0
    kh  = kd[0]/d_H[0]
    kh_o_kt  = -1
    kt = kh/kh_o_kt

  test = test1 = 0
  # Initialize the dielectric tensor and its derivative
  j      = complex(0,1.0)    # use as a imaginary number
  D      = np.zeros((3,3),dtype=complex)
  pre1  = [1.0, tau/z]
  pre2  = [1.0, -1.0]
  pre3  = [1.0, z/tau]
  A1 = B1 = C1= D1 = E1 = F1 = CE= Db =0
  sign = [1.0,-1.0]
  # Sum over different species(ion+e-)
  for k in range(0,2):
    # Initialize the temporary dielectric tensor and its derivative
    D1    = np.zeros((3,3),dtype=complex)
    dD1   = np.zeros((3,3),dtype=complex)
    # a = (k_perp*rho_i)**2/2.0
    vd = d_H[k]*np.sqrt(beta[k])/2.0
    kprp = kpd[k]*np.sqrt(beta[k])
    kprl = kxd[k]*np.sqrt(beta[k])
    a = kprp**2/2.0
    if k==0:
      a0 = kprp**2/2.0

    # omt = omega_ts/omega
    #omt  = -xp*np.cos(phi)/x/(ze[k])*vd*kh_o_kt
    if ind==0:
      omp=omt=0
    else:
      omp = xp*np.cos(phi)/x/(2.0*ze[k])*np.sqrt(beta[k])*d_H[k]
      omt = -xp*np.cos(phi)/x/(2.0*ze[k])*np.sqrt(beta[k])*d_H[k]*kh_o_kt

    # summation over modified bessel function n.
    for n in range(-m,m+1): 
      kyvd = vd*xp/x*np.cos(phi)
      zz   = ze[k] + kyvd - n/kprl
      # tensor component and derivative for different n & k
      D1[0,0]  += f.p(zz,2)*sp.ive(n,a)*(-omt*1.5+1.0)+\
                  omt*f.p(zz,4)*sp.ive(n,a)\
                  + omt*a*f.dive(n,a,1)*f.p(zz,2)

      D1[0,1]  += (n*np.cos(phi)/np.sqrt(2.0*a)*sign[k]-vd)\
                  *(f.p(zz,1)*sp.ive(n,a)\
                  *(1.0-omt*1.5)+omt*(f.p(zz,3)*sp.ive(n,a)+\
                  a*f.dive(n,a,1)*f.p(zz,1)))\
                  -j*np.sin(phi)*np.sqrt(a/2.)*sign[k]*(f.p(zz,1)*f.dive(n,a,1)\
                  *(1.0-omt*0.5)+omt*(f.p(zz,3)*f.dive(n,a,1)+\
                  a*f.dive(n,a,2)*f.p(zz,1))) 

      #test  += n*np.cos(phi)\
      #            *f.p(zz,1)
      #if n ==0:
      #  print 'n==0'
      #else:
      #  test1 += -ze[0]*(kxd[k]*np.sqrt(beta[k]))**3/n**3
      #print 'test',test,'test1',test1

      D1[0,2]  += sign[k]*(n*np.sin(phi)/np.sqrt(2.0*a)*(f.p(zz,1)*sp.ive(n,a)\
                  *(1.0-omt*1.5)+omt*(f.p(zz,3)*sp.ive(n,a)+\
                  a*f.dive(n,a,1)*f.p(zz,1)))\
                  +j*np.cos(phi)*np.sqrt(a/2.0)*(f.p(zz,1)*f.dive(n,a,1)\
                  *(1.0-omt*0.5)+omt*(f.p(zz,3)*f.dive(n,a,1)+\
                  a*f.dive(n,a,2)*f.p(zz,1))))


      D1[1,0]  += (n*np.cos(phi)/np.sqrt(2.0*a)*sign[k]-vd)\
                  *(f.p(zz,1)*sp.ive(n,a)\
                  *(1.0-omt*1.5)+omt*(f.p(zz,3)*sp.ive(n,a)+\
                  a*f.dive(n,a,1)*f.p(zz,1)))\
                  +j*np.sin(phi)*np.sqrt(a/2.)*sign[k]*(f.p(zz,1)*f.dive(n,a,1)\
                  *(1.0-omt*0.5)+omt*(f.p(zz,3)*f.dive(n,a,1)+\
                  a*f.dive(n,a,2)*f.p(zz,1))) 

      D1[1,1]  += ((n*np.cos(phi)/np.sqrt(2.0*a)*sign[k]-vd)**2+\
                  n**2*np.sin(phi)**2/2.0/a)\
                  *(f.p(zz,0)*sp.ive(n,a)*(1.-omt*1.5)+omt*(f.p(zz,2)*sp.ive(n,a)\
                  +a*f.dive(n,a,1)*f.p(zz,0)))\
                  -np.sin(phi)**2*a*(f.p(zz,0)*f.dive(n,a,1)*\
                  (1.0+omt*0.5)+omt*(f.p(zz,2)*f.dive(n,a,1)\
                  +a*f.dive(n,a,2)*f.p(zz,0))) 

      D1[1,2]  += j*(n/2.0-np.cos(phi)*np.sqrt(a/2.)*sign[k]*vd)\
                  *(f.p(zz,0)*f.dive(n,a,1)\
                  *(1.0-omt*0.5)+omt*(f.p(zz,2)*f.dive(n,a,1)+\
                  a*f.dive(n,a,2)*f.p(zz,0)))\
                  -n*np.sin(phi)/np.sqrt(2.0*a)*sign[k]*vd*\
                  (f.p(zz,0)*sp.ive(n,a)\
                  *(1.-omt*1.5)+omt*(f.p(zz,2)*sp.ive(n,a)+\
                  a*f.dive(n,a,1)*f.p(zz,0)))\
                  +0.5*np.sin(2.0*phi)*a*(f.p(zz,0)*f.dive(n,a,1)*(1.+0.5*omt)\
                  +omt*(f.p(zz,2)*f.dive(n,a,1)+a*f.dive(n,a,2)*f.p(zz,0)))

      D1[2,0]  += sign[k]*(n*np.sin(phi)/np.sqrt(2.0*a)*(f.p(zz,1)*sp.ive(n,a)\
                  *(1.0-omt*1.5)+omt*(f.p(zz,3)*\
                  sp.ive(n,a)+a*f.dive(n,a,1)*f.p(zz,1)))\
                  -j*np.cos(phi)*np.sqrt(a/2.)*(f.p(zz,1)*f.dive(n,a,1)\
                  *(1.0-omt*0.5)+omt*(f.p(zz,3)*f.dive(n,a,1)+\
                  a*f.dive(n,a,2)*f.p(zz,1))))

      D1[2,1]  += -j*(n/2.0-np.cos(phi)*np.sqrt(a/2.)*sign[k]*vd)\
                  *(f.p(zz,0)*f.dive(n,a,1)\
                  *(1.0-omt*0.5)+omt*(f.p(zz,2)*f.dive(n,a,1)+\
                  a*f.dive(n,a,2)*f.p(zz,0)))\
                  -n*np.sin(phi)/np.sqrt(2.0*a)*sign[k]*vd\
                  *(f.p(zz,0)*sp.ive(n,a)\
                  *(1.-omt*1.5)+omt*(f.p(zz,2)*sp.ive(n,a)+a*f.dive(n,a,1)*f.p(zz,0)))\
                  +0.5*np.sin(2.0*phi)*a*(f.p(zz,0)*f.dive(n,a,1)*(1.+0.5*omt)\
                  +omt*(f.p(zz,2)*f.dive(n,a,1)+a*f.dive(n,a,2)*f.p(zz,0)))

      D1[2,2]  += n**2/2.0/a*(f.p(zz,0)*sp.ive(n,a)*\
                  (1.-omt*1.5)+omt*(f.p(zz,2)*sp.ive(n,a)\
                  +a*f.dive(n,a,1)*f.p(zz,0)))\
                  -np.cos(phi)**2*a*(f.p(zz,0)*f.dive(n,a,1)*\
                  (1.+omt*0.5)+omt*(f.p(zz,2)*f.dive(n,a,1)\
                  +a*f.dive(n,a,2)*f.p(zz,0)))


    D[1,0] += pre1[k]/ze[k]*(D1[0,0]+xp/x*np.cos(phi)*D1[0,1]+xp/x*np.sin(phi)*D1[0,2])

    D[1,1] += pre1[k]/ze[k]*(D1[0,0]-x/xp*np.cos(phi)*D1[0,1]-x/xp*np.sin(phi)*D1[0,2])
    if k==0:
       ombar = ze[0]*np.sqrt(beta[0])
       D[1,1] += -a/ombar**2

    D[1,2] += pre2[k]*2.0*j/(kpd[k]*np.sqrt(beta[k]))*(np.cos(phi)*D1[0,2]-np.sin(phi)*D1[0,1]) 

    D[0,0] += pre1[k]/ze[k]*(D1[0,0]+xp/x*np.cos(phi)*(D1[0,1]+D1[1,0])+\
              xp/x*np.sin(phi)*(D1[0,2]+D1[2,0])+(xp/x)**2*\
              (np.cos(phi)**2*D1[1,1]+np.sin(phi)**2*D1[2,2])+\
              (xp/x)**2*np.sin(phi)*np.cos(phi)*(D1[1,2]+D1[2,1]))-pre1[k]*omp*omt#*np.cos(phi)**2

    D[0,1] += pre1[k]/ze[k]*(D1[0,0]+xp/x*np.cos(phi)*(D1[1,0]-(x/xp)**2*D1[0,1])+\
             xp/x*np.sin(phi)*(D1[2,0]-(x/xp)**2*D1[0,2])-np.sin(phi)*np.cos(phi)*\
             (D1[1,2]+D1[2,1])-(np.sin(phi)**2*D1[2,2]+np.cos(phi)**2*D1[1,1]))+\
             +pre1[k]*(x/xp)**2*omt*omp

    D[0,2] += pre2[k]*j*sign[k]*np.sqrt(2.0/a)*(xp/x*(np.cos(phi)**2*D1[1,2]-np.sin(phi)**2*D1[2,1])\
              +np.cos(phi)*D1[0,2]-np.sin(phi)*D1[0,1]-xp/x*np.sin(phi)*np.cos(phi)*(D1[1,1]-D1[2,2])\
              )+ pre2[k]*j*sign[k]*np.sqrt(2.0/a)*ze[k]*x/xp*np.tan(phi)*omp*omt

    D[2,0] += pre2[k]*2.0*j/(kpd[k]*np.sqrt(beta[k]))*(np.sin(phi)*D1[1,0]-np.cos(phi)*D1[2,0]-\
              xp/x*(np.cos(phi)**2*D1[2,1]-np.sin(phi)**2*D1[1,2])-(xp/x)*np.sin(phi)*np.cos(phi)*(\
              D1[2,2]-D1[1,1]))-2.0*j/(kpd[k]*np.sqrt(beta[k]))*np.tan(phi)*pre2[k]*omp*omt*ze[k]*(x/xp)

    D[2,1] += pre2[k]*2.0*j/(kpd[k]*np.sqrt(beta[k]))*(-np.cos(phi)*D1[2,0]+np.sin(phi)*D1[1,0]\
              +x/xp*(np.cos(phi)**2*D1[2,1]-np.sin(phi)**2*D1[1,2])+ x/xp*np.sin(phi)*np.cos(phi)*\
              (D1[2,2]-D1[1,1]))+ 2.0*j/(kpd[k]*np.sqrt(beta[k]))*np.tan(phi)*pre2[k]*omt*omp*ze[k]*\
              (x/xp)**3
    if ind>0: # inhomogeneous
      D[2,2] += pre3[k]*2.0*ze[k]*(np.cos(phi)**2/a*D1[2,2]+\
                np.sin(phi)**2/a*D1[1,1]) - pre3[k]*np.sin(phi)*np.cos(phi)*\
                2.0*ze[k]*(D1[1,2]+D1[2,1])/a - pre3[k]*np.sin(phi)**2*(kd[0]/kpd[0])**2/(kh*kt)
    else: # homogenous
      D[2,2] += pre3[k]*2.0*ze[k]/a*(np.cos(phi)**2*D1[2,2]+\
                np.sin(phi)**2*D1[1,1]) - pre3[k]*np.sin(phi)*np.cos(phi)*\
                2.0*ze[k]*(D1[1,2]+D1[2,1])/a 
    if k==0:
      D[2,2] += -2.0/beta[0]*(kd[0]/kpd[0])**2

 

    A1 += pre1[k]/ze[k]*(D1[0,0]+xp/x*np.cos(phi)*(D1[0,1]+D1[1,0])+\
          xp/x*np.sin(phi)*(D1[0,2]+D1[2,0])+(xp/x)**2*(np.cos(phi)**2*D1[1,1]+\
          np.sin(phi)**2*D1[2,2])+(xp/x)**2*np.sin(phi)*np.cos(phi)*(D1[1,2]+D1[2,1]))\
          -pre1[k]*omp*omt
   
    F1 += pre1[k]/ze[k]*(-(x/xp+xp/x)*np.cos(phi)*D1[0,1]-(x/xp+xp/x)*np.sin(phi)*D1[0,2])

    B1 += -pre1[k]*omp*omt+pre1[k]/ze[k]*(xp/x*np.cos(phi)*D1[1,0]+xp/x*np.sin(phi)*D1[2,0]+\
          (xp/x)**2*(np.cos(phi)**2*D1[1,1]+np.sin(phi)**2*D1[2,2])+(xp/x)**2*np.sin(phi)*np.cos(phi)\
          *(D1[1,2]+D1[2,1]))

    C1 += pre2[k]*2.0/(kpd[k]*np.sqrt(beta[k]))*(j*xp/x*(np.cos(phi)**2*D1[1,2]-np.sin(phi)**2*D1[2,1])\
          +j*(np.cos(phi)*D1[0,2]-np.sin(phi)*D1[0,1])-j*xp/x*np.sin(phi)*np.cos(phi)*(D1[1,1]-D1[2,2]))\
          +j*pre2[k]*np.sin(phi)*np.cos(phi)/kh/kt*(kd[k]*np.sqrt(beta[k])/2.0/ze[k]/x)

    Db += -1.0/xp**2/beta[0] - pre3[k]*np.sin(phi)**2/xp**2/kh/kt\
          + pre3[k]*2.0*ze[k]/a*(np.cos(phi)**2*D1[2,2]+np.sin(phi)**2*D1[1,1])\
          -np.sin(phi)*np.cos(phi)*pre3[k]*2.0*ze[k]/a*(D1[1,2]+D1[2,1])

    CE += pre2[k]*2.0*j/(kpd[k]*np.sqrt(beta[k]))*(np.cos(phi)*D1[0,2]-np.sin(phi)*D1[0,1])
    #print 'D101',D1[0,1],'D102',D1[0,2]
  print 'Full'
  print 'A1',A1
  print 'A1-B1',A1-B1
  print 'C1',C1
  print 'Db',Db
  print 'CE',CE
  ombar = ze[0]*np.sqrt(beta[0])
  print 'A1-B1+F1',A1-B1+F1-a0/ombar**2
  print 'F1',F1

  print 'D',D
  sys.exit()
  #print 'D',D
  fz    = scipy.linalg.det(D)#/ze[0]**2
  if math.isnan(fz.real) or math.isnan(fz.imag):
    fz = D[0,0]*(D[1,1]*D[2,2]-D[1,2]*D[2,1])-D[0,1]*(D[1,0]*D[2,2]-D[1,2]*D[2,0])+D[0,2]*(D[1,0]*D[2,1]-D[1,1]*D[2,0])
  return (fz.real, fz.imag)

