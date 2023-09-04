# Modified by Renato de Castro Rabelo Profeta - renato.profeta@tu-ilmenau.de
# Fixed the missing normalization to rxx[0] and adapted the outputs to the fix.
# This function now returns the normalized coefficients
#Modified by Gerald Schuller to enable running from the terminal with the file name as argument
#Aug. 2018



def adaptive_prefilter_quant(N,P,olap,q, filename):
# Author : K Suresh sureshk@ieee.org
# Adaptive Pre Filter followed by Quantization    
# Interpolates Reflex coefficients across Frame 
# Lattice FIR implementation
# Autocorrelation sequence is derived from psychoacoustic threshold
# Reflexion coefficients are derived from ACF using Levinson-Durbin Recursion
# N = Frame Size, P = Filter Order, olap = Interpolation Depth , q = quantization (can be a float)
# filename: name of the input sound file
# Returns filt_lattice=filter coefficients (ki),filter_output=filtered signal,olap = extend of interpolation used 
# filtered_spectrum = DFT Spectrum of the filtered signal ,input_spectrum = DFT Spectrum of the input signal, rate=sampling rate
# filtered signal is written to the wavfile filt_quantized.wav


   import numpy as np
   import math
   import scipy.io.wavfile
   from scipy import signal as signal
   import levinson_durbin
   from levinson_durbin import levinson_durbin
   import psycho_acoustic_model
   from psycho_acoustic_model import psycho
   #import inputFile 
   #from inputFile import fileDialog
   import matplotlib.pyplot as plt
   
   def lattice_firfilter(x,k_current,k_old,k_next,olap,state):
   #
   # x is the signal to be filtered
   # k is the lattice coefficient vector
      P=k_current.size
      input_length=x.size
      NX=input_length-1
      X=np.zeros((2,input_length)).astype(float)
      x_shift=np.zeros((input_length)).astype(float)
      output=np.zeros((input_length)).astype(float)
      tmp_state=np.zeros(P).astype(float)
      x_shift[1:input_length]=x[0:NX]
      k=np.zeros(P).astype(float)
      K=np.zeros((2,2)).astype(float)
      K[0,0]=K[1,1]=1
      X[0,:]=x
      X[1,:]=x_shift
      x_tmp0=x_tmp1=k_tmp=0.0
      
      for m in range(0,P):
         X[1,0]=state[m]
         k[m]=0.5*(k_old[m]+k_current[m])
         for n in range(olap):
            k_tp=k[m]+n*(k_current[m]-k_old[m])/(2*olap)
            x_tmp0=X[0,n]+k_tp*X[1,n]
            x_tmp1=X[1,n]+k_tp*X[0,n]
            X[0,n]=x_tmp0
            X[1,n]=x_tmp1
         
         k_tp=k_current[m]
         for n in range(olap,input_length-olap):
            
            x_tmp0=X[0,n]+k_tp*X[1,n]
            x_tmp1=X[1,n]+k_tp*X[0,n]
            X[0,n]=x_tmp0
            X[1,n]=x_tmp1
         for n in range(input_length-olap,input_length):
            k_tp=k_current[m]+(n-input_length+olap)*(k_next[m]-k_current[m])/(2*olap)
            x_tmp0=X[0,n]+k_tp*X[1,n]
            x_tmp1=X[1,n]+k_tp*X[0,n]
            X[0,n]=x_tmp0
            X[1,n]=x_tmp1
         if m<P-1:
            tmp_state[m+1]=X[1,NX]
         x_shift[1:input_length]=X[1,0:NX]
         X[1,:]=x_shift
      tmp_state[0]=x[NX]
      output[0:input_length]=X[0,:]
      return output,tmp_state
   
   
   #filename=fileDialog()
   rate, x = scipy.io.wavfile.read(filename)
   #x=np.fromfile(open(filename),np.int16)[24:]
   blks=x.size/N
   num_blocks=np.floor(blks).astype(int)
   z=x[0:num_blocks*N]
   fil_coef=np.zeros(0).astype(float)
   filt_output=np.zeros(0).astype(float)
   residue=np.zeros(0).astype(float)
   reconst=np.zeros(0).astype(float)
   filt_lattice=np.zeros((P,blks)).astype(float)
   print(filt_lattice.shape)
   state = np.zeros((P)).astype(float)
   state=b_prev=np.zeros(P).astype(float)
   rxx=psycho(z[0:N],rate,N)
   reflex_coef_current,cnt=levinson_durbin(rxx,P)
   reflex_coef_old = reflex_coef_current
   filt_lattice[:,0]=reflex_coef_current
   filter_output=np.zeros((N,num_blocks)).astype(float)
   filtered_spectrum=np.zeros((N,num_blocks)).astype(complex)
   input_spectrum=np.zeros((N,num_blocks)).astype(complex)
   norm_coeff=np.zeros(num_blocks).astype(float)
   for n in range(num_blocks):
      if n<num_blocks-1:
         rxx=psycho(z[(n+1)*N:(n+2)*N],rate,N)
         reflex_coef_next,count=levinson_durbin(rxx,P)
         filt_lattice[:,n+1]=reflex_coef_next
      tmp_latticefiltered,tmp_state=lattice_firfilter(z[n*N:n*N+N],reflex_coef_current,reflex_coef_old,reflex_coef_next,olap,state)
      # Divide by sqrt (rxx[0]) -> normalization
      tmp_latticefitered_norm=tmp_latticefiltered / np.sqrt(rxx[0])
      state=tmp_state
      tmp_inpspectrum=np.fft.fft(z[n*N:n*N+N])
      tmp_spectrum=np.fft.fft(tmp_latticefitered_norm)
      tmp_quantized=np.rint(tmp_latticefitered_norm/q)
      reflex_coef_old=reflex_coef_current
      reflex_coef_current=reflex_coef_next   
      filter_output[:,n]=tmp_quantized
      filtered_spectrum[:,n]=tmp_spectrum
      input_spectrum[:,n]=tmp_inpspectrum
      norm_coeff[n]=np.sqrt(rxx[0])
   plt.plot(np.reshape(filter_output,N*num_blocks,'f'),label="pre-filter output")
   plt.legend()
   plt.show()
   towav=np.reshape(filter_output,N*num_blocks,'f').astype(float)
   towav=np.squeeze(np.asarray(towav))
   avg_bits=1+sum(np.log2(np.array(np.abs(towav[towav!=0]))))/towav.size
   filter_output=q*filter_output
   towav=np.int16(towav)   
   print(avg_bits)
   scipy.io.wavfile.write('filt_quantized.wav',rate,towav)
   return filt_lattice,filter_output,olap,filtered_spectrum,input_spectrum, norm_coeff, rate

if __name__ == '__main__':
   import sys
   import os
   import scipy
   if sys.version_info[0] < 3:
      # for Python 2
      import cPickle as pickle
   else:
      # for Python 3
      import pickle
   
   if (len(sys.argv) <2):
      print('\033[93m'+"Need audio file as argument!"+'\033[0m') #warning in yellow font
   filename=sys.argv[1]
   print("audiofile=", filename)
   name,ext=os.path.splitext(filename)
   #new extension for compressed file:
   preffile=name+'prefiltered.wav'
   
   q=0.1  #Quantization step size, q=1 corresponds roughly to quantization noise at the
          #computed masking threshold, smaller quantization step sizes mean better quality
   filt_lattice,filter_output,olap,filtered_spectrum,input_spectrum, norm_coeff, rate=adaptive_prefilter_quant(N=128,P=11,olap=32,q=q, filename=filename)
   print("prefiltered output written to: 'filt_quantized.wav'")
   print("and to file:", preffile)
   #scipy.io.wavfile.write(preffile, 32000,(1/q*pref[0]).astype(np.int16))
   #Writing side info to picke file:
   with open('prefilt_sidefinfo.pickle', 'wb') as sidefile: #open compressed file
      pickle.dump(filt_lattice, sidefile, protocol=-1)
      pickle.dump(olap, sidefile, protocol=-1)
      pickle.dump(q, sidefile, protocol=-1)
      pickle.dump(norm_coeff, sidefile, protocol=-1)

   
   
   
