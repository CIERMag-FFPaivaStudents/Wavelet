#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Linear phase plot for different mother wavelets
"""



import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns



def graph(wavelet_str,ax):

    
    
    wavelet= pywt.Wavelet(wavelet_str)
    (phi,psi,x)=wavelet.wavefun(level=10)

    (dec_lo,dec_hi,rec_lo,rec_hi)=wavelet.filter_bank


    w, h = signal.freqz(rec_lo)

    h_u = np.unwrap(np.angle(h))
    
    w = w/np.pi
    
    # print(wavelet_str)
    ax.plot(w,h_u,label=wavelet_str)
    ax.plot([w[0],w[-1]],[h_u[0],h_u[-1]],'k--')
    ax.set_ylabel('Phase(rad)')
    ax.set_xlabel('Normalized Frequency')
    # ax.set_title(wavelet_str)
    
if __name__=='__main__':
    

    # fig = plt.figure(figsize=(10,8))
    # sns.set()
    # sns.set_context('talk')
    # sns.set_style('white')

    # ax = fig.add_subplot(1,1,1)
    # wavelet_str='coif4'
        
    # wavelet_list = ['sym4','coif4','db4']
    # for wavelet_str in wavelet_list:
    #     graph(wavelet_str,ax)
    # plt.legend(loc='best')
    


    fig = plt.figure(figsize=(30,16))
    sns.set()
    sns.set_context('talk')
    sns.set_style('white')
    
    
        
    wavelet_list = ['sym4','coif4','db4']
    for i in range(len(wavelet_list)):
        ax = fig.add_subplot(2,len(wavelet_list),i+4)
        graph(wavelet_list[i],ax)
        
        
    for i in range(len(wavelet_list)):
        ax = fig.add_subplot(2,len(wavelet_list),i+1)
        wavelet= pywt.Wavelet(wavelet_list[i])
        (phi,psi,x)=wavelet.wavefun(level=10)
        ax.plot(x,psi)
        ax.set_title(wavelet_list[i]+' - Scale Function')

    # plt.savefig('Fig19.png')
    
    
    # sr = 44100
    # w, h = signal.freqz(b=[1, -1], a=1)
    # x = w * sr * 1.0 / (2 * np.pi)
    # # y = 20 * np.log10(abs(h))
    # plt.figure(figsize=(10,5))
    # # plt.semilogx(x, y)
    # plt.plot(x,20*np.abs(h))
    # plt.ylabel('Amplitude [dB]')
    # plt.xlabel('Frequency [Hz]')
    # plt.title('Frequency response')
    # plt.grid(which='both', linestyle='-', color='grey')
    # plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])
    # plt.show()


