# Adaptive Pre- and Post- Filters
Adaptive Pre and Post Filters based on Perceptual Audio Coding Using Adaptive Pre- and Post-Filters and Lossless Compression by G. Schuller

[Perceptual audio coding using adaptive pre-and post-filters and lossless compression](https://www.idmt.fraunhofer.de/content/dam/idmt/en/documents/Personal%20Websites/Schuller/publications/tsap9-02.pdf)
<br>GDT Schuller, B Yu, D Huang, B Edler
IEEE Transactions on Speech and Audio Processing 10 (6), 379-390

## Description

### adaptive_prefilter_quant.py 

* Adaptive Pre Filter followed by Quantization 	
* Interpolates Reflex coefficients across Frame 
* Lattice FIR implementation
* Autocorrelation sequence is derived from psychoacoustic threshold
* Reflexion coefficients are derived from ACF using Levinson-Durbin Recursion

#### Usage 
filt_lattice,filter_output,olap,filtered_spectrum,input_spectrum, norm_coeff, rate=adaptive_prefilter_quant(N,P,olap,q)<br>
The filtered signal is written to the wavfile filt_quantized.wav<br>

Arguments:

N = frame size = 128<br> 
P = filter order , usally 10 or 11<br> 
overlap = interpolation depth for filter coefficients. For a frame size of 128, you can choose<br> 
          overlap in the range 16-32, though it supports values from 1-63.<br>
q = is used to change quantization level of the filtered signal,  can give values 2,4,8,...512. <br>
     ( \hat(x) = q* trunc (x/q )) ie filter output is divided by q, truncates, and then   multiplied by q. <br>

Returns:

filt_lattice=filter coefficients (ki)<br>
filter_output=filtered signal<br>
olap = extend of interpolation used<br> 
filtered_spectrum = DFT Spectrum of the filtered signal<br>
input_spectrum = DFT Spectrum of the input signal<br>
norm_coeff = normalized coefficients
rate = sampling rate of the audio output


### post_filter.py

Reconstruct signal back from pre-filtered signal. 

#### Usage
reconst = post_filter(fil_coef,filtered,olap,fs,norm_coeff)

Arguments:

fil_coef = filter coeefficinet matrix obtined from pre filter<br>
filtered = filtered signal from pre filter<br> 
olap = ineterpolation depth used in pre filter<br>
fs = sampling frequency, used in wavewrite <br>
norm_coeff = normalized coefficients <br>

Returns:

The reconstructed signal is returned in reconst.

Reconstructed signal is written to reconstructed.wav

### FilterTest-Interact.ipynb
### FilterApplication-Interact.ipynb

Interactive jupyter notebooks to test and demonstrate the pre- and post- filters.

In case you run into problems running jupyter widgets try:

pip install ipywidgets <br>
jupyter nbextension enable --py widgetsnbextension<br>

## Contributors
* **K Suresh**   sureshk@ieee.org
* **R Profeta**  renato.profeta@tu-ilmenau.de
* **G Schuller** gerald.schuller@tu-ilmenau.de
