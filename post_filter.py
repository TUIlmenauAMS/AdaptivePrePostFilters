# Modified by Renato de Castro Rabelo Profeta - renato.profeta@tu-ilmenau.de
# Fixed the missing normalization to rxx[0] and adapted the outputs to the fix.
# This function now takes the normilized coefficients as an input.

def post_filter(fil_coef,filtered,olap,fs, norm_coeff):
# Author : K Suresh sureshk@ieee.org
# Adaptive  Post Filtering, inverse filter for pre-filtered signal	
# Interpolates Reflex coefficients across Frame 
# fil_coef=filter coefficients used in pre-filter in clolumns, filtered = filtered frames in columns,
# olap = Interpolation Depth used in pre-filter
# fs= sampling frequency 
# Returns reconst = Reconstructed Signal
# Reconstructed signal is written to the wavfile reconstructed.wav


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
	
	
	
	def lattice_iirfilter(x,k_current,k_old,k_next,olap,b_prev):
	#
	# x is the signal to be filtered
	# k is the lattice coefficient vector
		P=k_current.size
		x=np.array(x)
		x=np.append([0],x)
		input_length=x.size
		NX=input_length-1
		output=np.zeros((input_length)).astype(float)
		k_1=np.zeros((P,1)).astype(float)
		k_1=0.5*(k_old+k_current)
		incr1=incr2=np.zeros((P,1)).astype(float)
		incr1=(k_current-k_old)/(2*olap)
		incr2=(k_next-k_current)/(2*olap)
		b_out=np.zeros((P,1)).astype(float)
		b_sequence=np.zeros((P,input_length)).astype(float)
		b_sequence[:,0]=b_prev
		error_sequence=np.zeros((P,input_length)).astype(float)
		error_sequence[0,0]=b_prev[P-1]
		for n in range(1,input_length):
			if n<olap+1:
				k=k_1+(n-1)*incr1
			elif (olap)<n<(input_length-olap):
				k=k_current
			else:
				k=k_current+(n-input_length+olap)*incr2			
			for m in range(0,P):
				if m==0:
					error_sequence[P-1-m,n]=x[n]-k[P-1-m]*b_sequence[P-2-m,n-1]	
					b_sequence[P-1-m,n]=k[P-1-m]*error_sequence[P-1-m,n]+b_sequence[P-2-m,n-1]
				else:
					if m!=P-1:
						error_sequence[P-1-m,n]=error_sequence[P-m,n]-k[P-1-m]*b_sequence[P-2-m,n-1]	
						b_sequence[P-m-1,n]=k[P-1-m]*error_sequence[P-1-m,n]+b_sequence[P-2-m,n-1]
					else:
						error_sequence[P-1-m,n]=error_sequence[P-m,n]-k[P-1-m]*error_sequence[P-1-m,n-1]	
						b_sequence[P-m-1,n]=k[P-1-m]*error_sequence[P-1-m,n]+error_sequence[P-1-m,n-1]
		output=error_sequence[0,1:]
		b_out=b_sequence[:,NX]
		b_out[P-1]=error_sequence[0,NX]
		return output,b_out
	shp=filtered.shape
	N=shp[0]	
	num_blocks=shp[1]
	
	reconst=np.zeros(0).astype(float)	
	reflex_coef_old =reflex_coef_current=fil_coef[:,0]
	P=reflex_coef_current.size
	b_prev=np.zeros(P).astype(float)
	state = np.zeros((P)).astype(float)
	for n in range(num_blocks):
		if n<num_blocks-1:
			reflex_coef_next=fil_coef[:,n+1]
		tmp_quantized=filtered[:,n]*norm_coeff[n]
		tmp_recon,tmp_b=lattice_iirfilter(tmp_quantized,reflex_coef_current,reflex_coef_old,reflex_coef_next,olap,b_prev)
		b_prev=tmp_b
		reflex_coef_old=reflex_coef_current
		reflex_coef_current=reflex_coef_next
		reconst=np.append(reconst,tmp_recon)
	reconst=np.int16(reconst)
	scipy.io.wavfile.write('reconstructed.wav',fs,reconst)
	plt.plot(reconst,label="reconstructed")
	plt.legend()
	return reconst
