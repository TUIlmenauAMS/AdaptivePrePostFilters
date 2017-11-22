def adaptive_prefilter(N,P,olap):
# Author : K Suresh sureshk@ieee.org
# Adaptive Pre Filter 
# Interpolates Reflex coefficients across Frame 
# Lattice FIR implementation
# Autocorrelation sequence is derived from psychoacoustic threshold
# Reflexion coefficients are derived from ACF using Levinson-Durbin Recursion
# N = Frame Size, P = Filter Order, olap = Interpolation Depth
# Returns filt_lattice=filter coefficients (ki),filter_output=filtered signal,olap = extend of interpolation used 
# Currentky supports 16000 Hz sampling rate only


	import numpy as np
	import math
	import scipy.io.wavfile
	from scipy import signal as signal
	import levinson_durbin
	levinson_durbin
	from levinson_durbin import levinson_durbin
	import psycho_acoustic_model
	psycho_acoustic_model
	from psycho_acoustic_model import psycho
	import inputFile 
	inputFile
	from inputFile import fileDialog
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
			k[m]=0.5*(k_old[m]+k_current[m])-(k_current[m]-k_old[m])/(2*olap)
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
				k_tp=k[m]+(n-input_length+olap)*(k_next[m]-k_current[m])/(2*olap)
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
	
	x=np.fromfile(open(fileDialog()),np.int16)[24:]
	blks=x.size/N
	num_blocks=np.floor(blks).astype(int)
	z=x[0:num_blocks*N]
	fil_coef=np.zeros(0).astype(float)
	filt_output=np.zeros(0).astype(float)
	residue=np.zeros(0).astype(float)
	reconst=np.zeros(0).astype(float)
	filt_lattice=np.zeros((P,blks)).astype(float)
	print filt_lattice.shape
	reflex_coef_old = state = np.zeros((P)).astype(float)
	state=b_prev=np.zeros(P).astype(float)
	rxx=psycho(z[0:N],16000,N)
	reflex_coef_current,cnt=levinson_durbin(rxx,P)
	filt_lattice[:,0]=reflex_coef_current
	#filter_output=np.zeros((0).astype(float)
	filter_output=np.zeros((N,num_blocks)).astype(float)
	norm_coeff=np.zeros(num_blocks).astype(float)
	for n in range(num_blocks):
		if n<num_blocks-1:
			rxx=psycho(z[(n+1)*N:(n+2)*N],16000,N)
			reflex_coef_next,count=levinson_durbin(rxx,P)
			filt_lattice[:,n+1]=reflex_coef_next
		tmp_latticefiltered,tmp_state=lattice_firfilter(z[n*N:n*N+N],reflex_coef_current,reflex_coef_old,reflex_coef_next,olap,state)
		# Divide by sqrt (rxx[0]) -> normalization
		tmp_latticefitered_norm=tmp_latticefiltered / np.sqrt(rxx[0])
		state=tmp_state
		tmp_quantized=np.rint(tmp_latticefiltered_norm)
		reflex_coef_old=reflex_coef_current
		reflex_coef_current=reflex_coef_next
		#filter_output=np.append(filter_output,tmp_quantized)	
		filter_output[:,n]=tmp_quantized
		norm_coeff[n]=np.sqrt(rxx[0])
	plt.plot(np.reshape(filter_output,N*num_blocks,'f'),label="pre-filter output")
	plt.legend()
	return filt_lattice,filter_output,olap, norm_coeff
