# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:59:20 2014

@author: Attilio Di Nisio

History:

- spectral_est_mod_06 20140814.
- spectral_est_mod_05 20140814.
- spectral_est_mod_04 20140620. I2MTC 2014 SI
- spectral_est_mod_03 20140614.
- spectral_est_mod_02 .
- spectral_est_mod_01 20140331.

"""
import pdb; dbs =  pdb.set_trace
from numpy import *
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.signal

def lp_filter_01(fs):
    # Filter deisgn
    #filt_b = scipy.signal.firwin(121,1.0/(fs/2))
    #filt_a = np.array([1])
    cut_off = 20.0
    filt_b,filt_a = scipy.signal.butter(7, cut_off/(fs/2))
    freqz_n = 4000
    w_norm, filt_H = scipy.signal.freqz(filt_b, filt_a,freqz_n)
    w_ind = cut_off/4/(fs/2)*freqz_n; 
    delay_sec = -np.angle(filt_H[w_ind])/(w_norm[w_ind]*fs) # approx delay in seconds
    return filt_a, filt_b, w_norm, filt_H, delay_sec

def lp_filter_02(fs):
    # Filter deisgn
    #filt_b = scipy.signal.firwin(121,1.0/(fs/2))
    #filt_a = np.array([1])
    cut_off = 20.1
    filt_b,filt_a = scipy.signal.butter(7, cut_off/(fs/2))
    freqz_n = 4000
    w_norm, filt_H = scipy.signal.freqz(filt_b, filt_a,freqz_n)
    w_ind = cut_off/20/(fs/2)*freqz_n; 
    delay_sec = -np.angle(filt_H[w_ind])/(w_norm[w_ind]*fs) # approx delay in seconds
    return filt_a, filt_b, w_norm, filt_H, delay_sec

def inst_phase_01(t,x,f_ref, filt_a, filt_b):
    """Return unwrapped instantaneous phase"""
    xI = x * np.cos(2*np.pi*f_ref*t)
    xQ = x * -np.sin(2*np.pi*f_ref*t)
    xI_filt  = scipy.signal.lfilter(filt_b, filt_a,xI)
    xQ_filt  = scipy.signal.lfilter(filt_b, filt_a,xQ)
    x_iphase = np.arctan2(xQ_filt, xI_filt)
    x_diff_iphase = np.diff(x_iphase)
    x_udiff_iphase = x_diff_iphase.copy()
    x_udiff_iphase[x_udiff_iphase>np.pi] -= 2*np.pi
    x_udiff_iphase[x_udiff_iphase<-np.pi] += 2*np.pi
    x_udiff_iphase = np.insert(x_udiff_iphase, 0, x_iphase[0])
    x_uiphase = np.cumsum(x_udiff_iphase)
    return x_uiphase

def se_base_resampling_01(x, g_est, f_ref, delay_n):
    #spectral estimation with base resampling
    g_est_i = (g_est[:-1] + g_est[1:])/2
    ind_U = zeros(len(g_est_i))
    g_est_L = zeros(len(g_est_i))
    T_ref_ext = 1/f_ref
    ind = 0
    for i in range(len(g_est_i)):
        g_U = g_est_i[i] - T_ref_ext
        g_est_L[i] = g_U
        while g_est_i[ind] <= g_U:
            ind += 1
        ind_U[i] = ind

    K = 2 #max order of armonics
    a_est = np.empty((len(x),K),complex); a_est.fill(nan)
    for i in range(len(x)-int(delay_n)-1):
        if ind_U[i+delay_n] - delay_n <= 0:
            continue
        for k in range(1,K+1):
            a_est[i,k-1]= 1/T_ref_ext * sum(  x[ind_U[i+delay_n]-delay_n+1:i+1] \
                * exp(1.0j*2*pi*k*f_ref*g_est[ind_U[i+delay_n]+1:i+delay_n+1]) \
                * ( g_est_i[ind_U[i+delay_n]+1:i+delay_n+1] - \
                    g_est_i[ind_U[i+delay_n]:i+delay_n] ) \
                   ) \
                 + 1/T_ref_ext * x[ind_U[i+delay_n]-delay_n] \
                     * exp(1.0j*2*pi*k*f_ref*g_est[ind_U[i+delay_n]]) \
                     * ( g_est_i[ind_U[i+delay_n]] - g_est_L[i+delay_n] )
    return a_est


def se_base_resampling_02(x, gd, f_ref, delay_n):
    #spectral estimation with base resampling
    #assert(issorted(dg))
    
    g_ = empty((len(gd)+2)); 
    g_[0] = -np.inf
    g_[1:-1] = gd
    g_[-1] = np.inf
    
    L_ = empty(gd.shape); L_.fill(nan)
    R_ = empty(gd.shape); R_.fill(nan)
    W = 1/f_ref
    WL_ = gd - W/2; #WL(not isfinite(ge)) = np.inf
    WR_ = gd + W/2; #WR(not isfinite(ge)) = -np.inf
    indL = 0
    indR = 0
    for i in range(0,len(gd)):
        lim = WL_[i]
        while g_[indL] < lim:
          indL += 1
        L_[i] = indL
        lim = WR_[i]
        while g_[indR] < lim:
          indR += 1
        R_[i] = indR
        
    L_[L_>=len(g_)-1] = np.nan
    L_[L_<=2] = np.nan
    R_[R_>=len(g_)-1] = np.nan
    R_[R_<=2] = np.nan

    DR = int(np.round(delay_n))
    ge = empty((len(gd)+DR)); 
    ge[0:DR] = np.nan
    ge[DR:] = gd
    WL = ge - W/2; #WL(not isfinite(ge)) = np.inf
    WR = ge + W/2; #WR(not isfinite(ge)) = -np.inf

    #note ge[i+DR] = ge[a]= g_[i+1] = g_[b];  a = b + DR -1
    #note ge[i+DR] = gd[i]
    L = empty(ge.shape); L.fill(nan)
    R = empty(ge.shape); R.fill(nan)
    L[DR:] = L_ + DR - 1
    R[DR:] = R_ + DR - 1

    for i in range(1,len(ge)):
      if isfinite(L[i]):
        if not ( ge[L[i]-1]<WL[i] and ge[L[i]]>=WL[i]) :
          print (i, L[i], ge[L[i]-1], WL[i], ge[L[i]] )
    for i in range(1,len(ge)):
      if isfinite(R[i]):
        if not( ge[R[i]-1]<WR[i] and ge[R[i]]>=WR[i]):
          print (i, R[i], ge[R[i]-1], WR[i], ge[R[i]] )

    K = 2 #max order of armonics
    a_est = np.empty((len(x),K),complex); a_est.fill(nan)
    for i in range(0,len(x)):
        if (not isfinite(L[i])) or (not isfinite(R[i])):
            continue
        for k in range(1,K+1):
            s = ge[R[i]]
            s = ge[R[i]-1]
            try:
              s = x[R[i]]
            except:
              print (R[i], len(x))
              raise
            s = x[R[i]-1]
            a_est[i,k-1]= 1/2/W * ( \
              sum(  x[L[i]:R[i]] \
                * exp(1.0j*2*pi*k/W*ge[L[i]:R[i]]) \
                * ( ge[L[i]+1:R[i]+1] - ge[L[i]-1:R[i]-1]) \
                ) \
              + x[R[i]] * exp(1.0j*2*pi*k/W*ge[R[i]])  * (WR[i]-ge[R[i]-1]) \
              - x[R[i]-1] * exp(1.0j*2*pi*k/W*ge[R[i]-1])  * (ge[R[i]]-WR[i]) \
              - x[L[i]] * exp(1.0j*2*pi*k/W*ge[L[i]])  * (WL[i]-ge[L[i]-1]) \
              + x[L[i]-1] * exp(1.0j*2*pi*k/W*ge[L[i]-1])  * (ge[L[i]]-WL[i]) \
              )
                
    return a_est
    
def se_base_resampling_03(x, gd, f_ref, delay_n, K):
    #spectral estimation with base resampling
    #assert(issorted(dg))
    
    g_ = empty((len(gd)+2)); 
    g_[0] = -np.inf
    g_[1:-1] = gd
    g_[-1] = np.inf
    
    L_ = empty(gd.shape); L_.fill(nan)
    R_ = empty(gd.shape); R_.fill(nan)
    W = 1/f_ref
    WL_ = gd - W/2; #WL(not isfinite(ge)) = np.inf
    WR_ = gd + W/2; #WR(not isfinite(ge)) = -np.inf
    indL = 0
    indR = 0
    for i in range(0,len(gd)):
        lim = WL_[i]
        while g_[indL] < lim:
          indL += 1
        L_[i] = indL
        lim = WR_[i]
        while g_[indR] < lim:
          indR += 1
        R_[i] = indR
        
    DR = int(np.round(delay_n))
    ge = empty((len(gd))); ge.fill(nan)
    ge[0:len(ge)-DR] = gd[DR:]
    WL = ge - W/2; #WL(not isfinite(ge)) = np.inf
    WR = ge + W/2; #WR(not isfinite(ge)) = -np.inf

    #note ge[i] = gd[i+DR] = g_[i+DR+1]; 
    #note WL_[i+DR] = gd[i+DR] - W/2 = ge[i] - W/2 = WL[i]
    #note g_[L_[i+DR]-1] < WL_[i+DR] <= g_[L_[i+DR]]
    #note ge[L_[i+DR]-DR-2] < WL[i] <= ge[L_[i+DR]-DR-1]
    #note ge[L[i]-1] < WL[i] <= ge[L[i]] 
    #note L[i] = L_[i+DR]-DR-1    
    
    L = empty(ge.shape); L.fill(nan)
    R = empty(ge.shape); R.fill(nan)
    L[0:len(ge)-DR] = L_[DR:] - DR - 1
    R[0:len(ge)-DR] = R_[DR:] - DR - 1

    L[L>=len(ge)-1] = np.nan
    L[L<=0] = np.nan
    R[R>=len(ge)-1] = np.nan
    R[R<=0] = np.nan

    """
    for i in range(1,len(ge)):
      if isfinite(L[i]):
        if not ( ge[L[i]-1]<WL[i] and ge[L[i]]>=WL[i]) :
          print (i, L[i], ge[L[i]-1], WL[i], ge[L[i]] )
    for i in range(1,len(ge)):
      if isfinite(R[i]):
        if not( ge[R[i]-1]<WR[i] and ge[R[i]]>=WR[i]):
          print (i, R[i], ge[R[i]-1], WR[i], ge[R[i]] )
    """
    a_est = dt_integr_02(x, ge, WL, WR, L, R, W, K)
    return a_est

def dt_integr_01(x, ge, WL, WR, L, R, W, K):
  #roughtly approximated boundary    
  a_est = np.empty((len(x),K+1),complex); a_est.fill(nan)
  for k in range(1,K+1):
    f = 1/2/W * x * exp(1.0j*2*pi*k/W*ge)
    for i in range(0,len(x)):
      if (not isfinite(L[i])) or (not isfinite(R[i])):
        continue
      a_est[i,k]= \
        sum(  f[L[i]:R[i]] * ( ge[L[i]+1:R[i]+1] - ge[L[i]-1:R[i]-1]) ) + \
        + f[R[i]] * (WR[i]-ge[R[i]-1]) \
        - f[R[i]-1] * (ge[R[i]]-WR[i]) \
        - f[L[i]] * (WL[i]-ge[L[i]-1]) \
        + f[L[i]-1] * (ge[L[i]]-WL[i])

  return a_est

def dt_integr_02(x, ge, WL, WR, L, R, W, K):    
  #linearly approximated boundary    
  a_est = np.empty((len(x),K+1),complex); a_est.fill(nan)
  for k in range(1,K+1):
    f = 1/2/W * x * exp(1.0j*2*pi*k/W*ge)
    for i in range(0,len(x)):
      if (not isfinite(L[i])) or (not isfinite(R[i])):
        continue
      a_est[i,k]= \
        sum(  f[L[i]:R[i]] * ( ge[L[i]+1:R[i]+1] - ge[L[i]-1:R[i]-1]) ) + \
        + f[R[i]]  * (WR[i]-ge[R[i]-1])**2 / (ge[R[i]]-ge[R[i]-1])\
        - f[R[i]-1] * (ge[R[i]]-WR[i])**2 / (ge[R[i]]-ge[R[i]-1])\
        - f[L[i]] * (WL[i]-ge[L[i]-1])**2 / (ge[L[i]]-ge[L[i]-1]) \
        + f[L[i]-1] * (ge[L[i]]-WL[i])**2 / (ge[L[i]]-ge[L[i]-1]) 
        
  return a_est
        
def dt_integr_03(x, ge, WL, WR, L, R, W, K):    
  # Experimentation with Simpson's integration
  a_est = np.empty((len(x),K+1),complex); a_est.fill(nan)
  for k in range(1,K+1):
    f = 1/2/W * x * exp(-1.0j*2*pi*k/W*ge)
    for i in range(0,len(x)):
      if (not isfinite(L[i])) or (not isfinite(R[i])):
        continue
      a_est[i,k]= \
        scipy.integrate.simps(2*f[L[i]:R[i]],ge[L[i]:R[i]]) \
        + f[R[i]-1]  * (ge[R[i]]-ge[R[i]-1]) \
        + f[R[i]]  * (WR[i]-ge[R[i]-1])**2 / (ge[R[i]]-ge[R[i]-1]) \
        - f[R[i]-1] * (ge[R[i]]-WR[i])**2 / (ge[R[i]]-ge[R[i]-1]) \
        + f[L[i]] * (ge[L[i]]-ge[L[i]-1]) \
        - f[L[i]] * (WL[i]-ge[L[i]-1])**2 / (ge[L[i]]-ge[L[i]-1]) \
        + f[L[i]-1] * (ge[L[i]]-WL[i])**2 / (ge[L[i]]-ge[L[i]-1]) 
        
  return a_est


def se_phase_resampling_01(x, gd, K, t1_smpl, f_ref, WN2, t2_start, g_delay, x_f = None, use_fft = False):
  """Spectral estimation with phase resampling
  signal interpolation to obtain uniform sampling for tranformed time
  t1_out: where estimate is calculated in reference U
  a_est: estimated components
  gd_out: estimate of g(t1_out)
  
  NOTE x_f eventually used instead of interpolated x for performance comaprison purposes.
  """
  W = 1/f_ref
  fs2 = f_ref * WN2
  t2 = t2_start +   np.arange( np.ceil( (gd[0]-t2_start)*fs2), 
                               np.ceil( (gd[-1]-t2_start)*fs2)) / fs2
  tau = np.interp(t2, gd, t1_smpl, left = np.nan, right = np.nan)
  t_int = tau - g_delay
  if x_f is None:
    x_int = np.interp(t_int, t1_smpl, x, left = np.nan, right = np.nan)
  else:
    x_int = x_f(t_int)
    
  gd_out = t2 + .5/f_ref
  t1_out = np.interp(gd_out, gd, t1_smpl, 
                     left = np.nan, right = np.nan) - g_delay
  

  N2 = len(t2)
  a_est = np.empty((N2,WN2),complex); a_est.fill(nan)
  if use_fft:
    p = np.arange(0, WN2); p = p - WN2*(p > WN2/2)
    for i in range(0,N2-WN2-1):
      f = 1/WN2 * x_int[i:i+WN2]
      f[0]  = (x_int[i] + x_int[i+WN2]) /2/WN2
      a_est[i,:] = np.fft.fft(f)
      a_est[i,:] *= exp(-1.0j*2*pi*p*f_ref*t2[i]) #time shift
  else:
    for k in range(1,K+1):
      f = x_int * exp(-1.0j*2*pi*k/W*t2)
      for i in range(0,N2-WN2-1):
        a_est[i,k]  = 1/(WN2) * (sum(f[i:i+WN2+1])- .5*f[i] - .5*f[i+WN2])
  
  return t1_out, a_est, gd_out
    
def se_phase_integr_04(x_f, g_f, ti, N_OS, W, WN2, K, use_fft = False):
  """ High accuracy integration in phase domain.
  OS = number of points for oversampling of g_f
  ti = instants where the ouput is calculated, i.e. the center of integration
  window.

  This function is similar to se_phase_resampling_01 called by passing    
  gd caclculated with high resolution and the function x_f, i.e.
  se_phase_resampling_01(None, g_f(t), K, t, 1/W, WN2, t2_start, 0.0, x_f, False)
  with high resolution t vector. The difference is that with
  se_phase_resampling_01 the instant to which the ouput belongs can't be
  predetermined.
  """
  N = len(ti)
  a_est = np.empty((N,WN2),complex); a_est.fill(nan)
  WL = g_f(ti) - W/2
  WR = g_f(ti) + W/2
  t_start = scipy.optimize.newton(lambda t:g_f(t) - (ti[0]-W), ti[0]-W)
  t_end = scipy.optimize.newton(lambda t:g_f(t) - (ti[-1]+W), ti[-1]+W)

  if use_fft:
    p = np.arange(0, WN2); p = p - WN2*(p > WN2/2)
  else:
    p = None
  
  for i in range(0,N):
    fun = lambda t: g_f(t)-WL[i]
    tL = scipy.optimize.brentq(fun, t_start, t_end)
    fun = lambda t: g_f(t)-WR[i]
    tR = scipy.optimize.brentq(fun, t_start, t_end)
    #print(g_f(tR)-g_f(tL)-W)
  
    t = np.linspace(tL,tR,N_OS)
    gt = g_f(t)
    gt2 = np.linspace(gt[0], gt[-1], WN2+1)  
    t2 = np.interp(gt2, gt, t)

    if use_fft:
        f = 1/WN2 * x_f(t2[0:-1])
        f[0]  = (x_f(t2[0]) + x_f(t2[-1])) /2/WN2
        a_est[i,:] = np.fft.fft(f)
        a_est[i,:] *= exp(-1.0j*2*pi*p*f_ref*t2[0]) #time shift
    else:
      for k in range(1,K+1):
        f_re = lambda t_: x_f(t_) * np.cos(-2*pi*k/W*g_f(t_))
        f_im = lambda t_: x_f(t_) * np.sin(-2*pi*k/W*g_f(t_))
        a_est[i,k]  = (1/(len(t2)-1) * (sum(f_re(t2)) + 1.0j*sum(f_im(t2)) \
                                        - .5*f_re(t2[0]) - .5*f_re(t2[-1]) \
                                        - .5j*f_im(t2[0]) - .5j*f_im(t2[-1]) )  )
  return a_est

def se_time_resampling_01(x_f, g_f, ti, W, WN):
  """ Spectral estimation with time resampling
  """
  N = len(ti)
  f_ref = 1/W
  WL = g_f(ti) - W/2
  WR = g_f(ti) + W/2
  t_start = scipy.optimize.newton(lambda t:g_f(t) - (ti[0]-W), ti[0]-W)
  t_end = scipy.optimize.newton(lambda t:g_f(t) - (ti[-1]+W), ti[-1]+W)

  a_est = np.empty((N,WN),complex); a_est.fill(nan)
  p = np.arange(0, WN); p = p - WN*(p > WN/2)
  for i in range(0,N-WN-1):
    fun = lambda t: g_f(t)-WL[i]
    tL = scipy.optimize.brentq(fun, t_start, t_end)
    fun = lambda t: g_f(t)-WR[i]
    tR = scipy.optimize.brentq(fun, t_start, t_end)
    t = np.linspace(tL,tR,WN+1)
    f = 1/WN * x_f(t)
    f[0]  = (f[0]+f[-1]) /2
    a_est[i,:] = np.fft.fft(f[0:-1])
    a_est[i,:] *= exp(-1.0j*2*pi*p*f_ref*tL) #time shift

  return a_est

def stfs_H1_01(x, PN, WN):
  """firist component of short time fourier series
  PN number of samples in the reference period
  WN number of samples in each window
  """
  # given samples x[i], dft[i] refers to the window centered in i + dft_shift
  if mod(WN,2)==0:
    dft_shift = 0.5
  else:
    dft_shift = 0.0
    
  n = np.arange(len(x))
  ex = np.exp(-1.0j*2*np.pi/PN*n)
  y = (1/PN)*x*ex
  s = np.cumsum(y)
  #TODO Avoid accumulation errors when calculating dft from s on large records.
  dft = np.empty(x.shape, complex); dft.fill(nan)
  dft[(WN-1)//2] = s[WN-1]
  dft[(WN-1)//2+1:-(WN+1)//2+1] = s[WN:]-s[:-WN] 
  dft *= np.exp(+1.0j*2*np.pi/PN*dft_shift) * ex.conj()
  return (dft, dft_shift)

def phasor_4P(X, PN, d):
  """Phasor estimation. Implementation of method 4-parameter described in [2008premerlani].
  
  X is dft on moving windows
  PN number of samples in the reference period  
  d integer is separation between windows. d = PN/2 in [2012barchi]
  
  [2008premerlani] W. Premerlani, B. Kasztenny, M. Adamiak, "Development and Implementation of a Synchrophasor Estimator Capable of Measurements Under Dynamic Conditions," IEEE Transactions on Power Delivery,  vol.23, no.1, pp.109,123, Jan. 2008. doi: 10.1109/TPWRD.2007.910982
  [2012barchi] G. Barchi ..., An improved dynamic synchrophasor estimator

  """
  Y =  np.empty(X.shape, complex); Y.fill(nan)
  Y[d:] = X[d:] - 1.0j/(2*d*np.sin(2*np.pi/PN)) * (X[d:] - X[:-d]).conj()
  return Y

def phasor_6P(X, N):
  """Phasor estimation. Implementation of method 6-parameter described in [2008premerlani].
  
  X is dft on moving windows
  PN number of samples in the reference period  
  
  [2008premerlani] W. Premerlani, B. Kasztenny, M. Adamiak, "Development and Implementation of a Synchrophasor Estimator Capable of Measurements Under Dynamic Conditions," IEEE Transactions on Power Delivery,  vol.23, no.1, pp.109,123, Jan. 2008. doi: 10.1109/TPWRD.2007.910982

  """
  Y =  np.empty(X.shape, complex); Y.fill(nan)
  Y[2*N:] = X[2*N:] \
          - 1.0j/(2*N*np.sin(2*np.pi/N)) * (3/2*X[2*N:] - 2*X[N:-N] + .5*X[0:-2*N]).conj() \
          - (1-1/N)/24    * (X[2*N:] - 2*X[N:-N] + X[0:-2*N]) \
          - np.cos(2*np.pi/N)/(2*N**2*np.sin(2*np.pi/N)**2) * (X[2*N:] - 2*X[N:-N] + X[0:-2*N]).conj()
          
  return Y
  
  
  
  