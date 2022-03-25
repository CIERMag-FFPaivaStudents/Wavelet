#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This evaluates the best way to estimate the standar deviation of a signal using wavelets
"""

import numpy as np
import wavelet_1D
import signal_1D
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.ndimage import convolve


def SNR_Measure(I,Ihat):
    n = np.sum(np.power(I,2))
    diff = np.subtract(I,Ihat)
    d = np.sum(np.power(diff,2))
    SNR = 10*np.log10(n/d)
    return SNR

def CoC_Measure(I,Ihat):
    meanI = np.mean(I)
    meanIhat = np.mean(Ihat)
    n = np.sum(np.multiply(I-meanI,Ihat-meanIhat))
    d = np.sqrt(np.sum(np.power(I-meanI,2))*np.sum(np.power(Ihat-meanIhat,2)))
    CoC = n/d
    return CoC

def EPI_Measure(I, Ihat):
    laplacian_kernel = np.array([[[0, 0, 0],[0, 1, 0], [0, 0, 0]],
                                [[0, 1, 0],[1, -6, 1],[0, 1, 0]],
                                [[0, 0, 0],[0, 1, 0], [0, 0, 0]]])

    lap_I = convolve(I, laplacian_kernel)
    lap_Ihat = convolve(Ihat, laplacian_kernel)

    mean_lap_I = np.mean(lap_I)
    mean_lap_Ihat = np.mean(lap_Ihat)

    n = np.sum((lap_I-mean_lap_I)*(lap_Ihat-mean_lap_Ihat))
    d = np.sqrt(np.sum((lap_I-mean_lap_I)**2)*
            np.sum((lap_Ihat-mean_lap_Ihat)**2)) 
    
    EPI = n/d
    return EPI

def MSSIM_Measure(I, Ihat):
    x,y,z = np.mgrid[-2:3,-2:3,-2:3]
    d2 = x**2+y**2+z**2
    sigma=1.5
    gaus_const = 1/((2*np.pi)**(3/2)*sigma**(1/2))
    gaussian_kernel = gaus_const*np.exp(-d2/(2*sigma**2)) 
    gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel) 

    C1 = 0.01*np.max(I)
    C2 = 0.03*np.max(I)

    I_conv = convolve(I,gaussian_kernel)
    Ihat_conv = convolve(Ihat,gaussian_kernel)

    mu_I_sq = I_conv**2
    mu_Ihat_sq = Ihat_conv**2
    mu_IIhat = I_conv*Ihat_conv

    sigma_I = convolve(np.power(I,2),gaussian_kernel)-mu_I_sq
    sigma_Ihat = convolve(np.power(Ihat,2),gaussian_kernel)-mu_Ihat_sq
    sigma_IIhat = convolve(I*Ihat,gaussian_kernel)-mu_IIhat

    SSIM = ((2*mu_IIhat + C1)*(2*sigma_IIhat+C2))/((
            mu_I_sq +mu_Ihat_sq+C1)*(sigma_I+sigma_Ihat+C2))

    return np.mean(SSIM)





if __name__ == '__main__':

    num_iterations=1
    
    
    
    delta_time=0.5
    T2=100
    freq=0.25 
    size_array=2048
    mean_set=0
    levels=5
    wavelet='db4'

    sigma_input_list=np.arange(0,0.6,0.01)
    
    length_input=len(sigma_input_list)


    for j in range(length_input):
        # continue
        sigma_input=sigma_input_list[j]
        
        sigma_output_matrix=np.zeros((num_iterations,8))
        for i in range(num_iterations):
            # print(j,i)
            signal_pure,time=signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq)
            
            signal_noise = signal_1D.Add_Noise1D(signal_pure,mean_set,sigma_input)

            snr=SNR_Measure(signal_noise,signal_pure)
            coc = CoC_Measure(signal_noise,signal_pure)
            epi = EPI_Measure(signal_noise,signal_pure)
            ssim = MSSIM_Measure(signal_noise,signal_pure)
        



#     # Escrevendo na tabela
#     Nome='Sigmas'
#     tabela = open(Nome+'.txt','w')
    
#     ctes='#Número de Simulações={}, Wavelet Mãe={}, Número de Níveis={},p={},Número de pontos={}\n'.format(num_iterations,wavelet,levels,cut_point,size_array)
#     # print(ctes)
#     tabela.write(ctes)
    
#     nomes_metricas=['Msigma{}\tSsigma{}'.format(i,i) for i in range(8)]
#     var='#'
#     var+='\t'.join(nomes_metricas)
#     var+='\n'
#     # print(var)
#     tabela.write(var)
    
#     pontos=''
#     for j in range(length_input):
#         for k in range(8):
#           m,s=sigma_output_mean_list[j][k],sigma_output_std_dev_list[j][k]
#           pontos+='{}\t{}'.format(m,s)
#           pontos+= '\t'
#         pontos+='\n'
#     # print(pontos)
#     tabela.write(pontos)
    
#     tabela.close()
    
    
    

    
    SIGMA = np.loadtxt('Sigmas.txt').T
    
    symbols=['h','*','o','^','s','p','H']

    plt.figure(figsize=(20,15))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    
    m0,s0 = SIGMA[0:2]
    for i in range(7):
        # plt.clf()
        mi,si = SIGMA[2*(i+1):2*(i+2)]
        plt.errorbar(m0,mi,yerr=si,marker=symbols[i],mew=0,label='sigma{}'.format(i+1))
        plt.plot(m0,m0,'k-',zorder=3)
        plt.xlim(-0.05,0.65)
        plt.ylim(-0.05,0.7)
        plt.savefig('Grafico sigma{}.png'.format(i+1))
    plt.xlabel(r'$\sigma_{input} (a.u.)$',fontsize=20)
    plt.ylabel(r'$\sigma_{output} (a.u.)$',fontsize=20)
    plt.legend(loc='best')
    plt.savefig('Tudo junto.png')
