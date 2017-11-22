def levinson_durbin(r,P):
# Author : K Suresh sureshk@ieee.org
# calculates lpc coefficeints from autocorrelation sequence 
# r = autocorrelation sequence
# P = order of the filter
# returns reflexion coefficients coefficients [k_1, ... k_P]
# returns count, the number of instances in which estimated k_i 
# exceeded unity. ( When k_i exceeeds unity, k_i = 0 is assigned to avoid instabilty of the reconstruction filter) 
	import numpy as np
	import math
	import scipy  
	N = r.size
	#r=r/r[0]
	E = r[0].astype(float)
	a=a_tmp=np.zeros(P+1).astype(float)
	count=0	
	a[0]=a_tmp[0]=1
	k = np.zeros(P+1).astype(float)
	ktp=E_tmp=0.0
	for i in range(1,P+1):
		if i==1:
			if E>0:
				k[i]=-r[i]/E
				a[i]=k[i]
				E_tmp=E*(1-k[i]*k[i])
				E=E_tmp
		else:
			ktp=r[i]
			for j in range(1,i):
				ktp=ktp+a[j]*r[i-j]
			if E>0:
				k[i]=-ktp/E
				ktp=0.0
				if np.abs(k[i])>1:
					k[i]=0
		                        count+=1
				for l in range(1,i):
					a_tmp[l]=a[l]+k[i]*a[i-l]
				a=a_tmp
				a[i]=k[i]
				E_tmp=E*(1-k[i]*k[i])
				E=E_tmp
			else:
				ktp=0		
	return k[1:P+1],count
