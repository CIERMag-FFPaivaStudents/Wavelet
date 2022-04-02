
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


if __name__ == '__main__':

#     num_iterations=1000
    
    
    
#     delta_time=0.5
#     T2=100
#     freq=0.25 
#     size_array=2048
#     mean_set=0
#     levels=5
#     wavelet='db4'

#     sigma_input_list=np.arange(0,0.6,0.01)
    
#     length_input=len(sigma_input_list)



#     sigma_output_mean_list=np.zeros((length_input,8))
#     sigma_output_std_dev_list=np.zeros((length_input,8))

#     for j in range(length_input):
#         # continue
#         sigma_input=sigma_input_list[j]
        
#         sigma_output_matrix=np.zeros((num_iterations,8))
#         for i in range(num_iterations):
#             # print(j,i)
#             signal_pure,time=signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq)
            
#             signal_noise = signal_1D.Add_Noise1D(signal_pure,mean_set,sigma_input)

            
#             # continue
#             approx_coef_list, detail_coef_list = wavelet_1D.DWT_1D(signal_noise,wavelet,levels)
            

#             sigma_output_list = np.zeros(8)
            
#             sigma_output_list[0] = sigma_input
            
            
#             sigma1 = np.std(signal_noise)
#             sigma_output_list[1]=sigma1
            
            
#             cut_point = 30
#             end_slice=int((100-cut_point)*size_array/100)
#             noise_estimate=signal_noise[end_slice:]
#             sigma2=np.std(noise_estimate)
#             sigma_output_list[2]=sigma2
            
            
#             sigma3 = 0
#             for k in range(0, levels):
#                 sigma3+=np.std(detail_coef_list[k+1])
#             sigma3 /=levels
#             sigma_output_list[3]=sigma3
            
            
#             median_factor = 0.675
#             sigma4 = np.median(np.abs(detail_coef_list[levels]))
#             sigma4/=median_factor
#             sigma_output_list[4]=sigma4
            
            
#             concatenated_detail_coef = np.hstack((detail_coef_list[1:]))
#             concatenated_detail_coef = np.abs(concatenated_detail_coef)
#             sigma5 = np.median(concatenated_detail_coef)
#             sigma5 /= median_factor
#             sigma_output_list[5]=sigma5
            
            
#             median_list=np.zeros(levels)
#             for k in range(0,levels):
#                 median_list[k] = np.median(np.abs(detail_coef_list[k+1]))
#             sigma6 = np.median(median_list)/median_factor
#             sigma_output_list[6]=sigma6
            
            
#             sigma7 = median_list.mean()/median_factor
#             sigma_output_list[7]=sigma7
            
            
#             sigma_output_matrix[i]=sigma_output_list

#         # continue
#         sigma_output_matrix = sigma_output_matrix.T

#         sigma_output_mean=[sigma_output_list.mean() for sigma_output_list in sigma_output_matrix]
#         sigma_output_mean = np.array(sigma_output_mean).T
#         sigma_output_mean_list[j]=sigma_output_mean
    
#         sigma_output_std_dev=[np.std(sigma_output_list) for sigma_output_list in sigma_output_matrix]
#         sigma_output_std_dev = np.array(sigma_output_std_dev).T
#         sigma_output_std_dev_list[j]=sigma_output_std_dev

# # sigma_output_mean_list=sigma_output_mean_list.T
# # sigma_output_std_dev_list=sigma_output_std_dev_list.T



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

    plt.figure(figsize=(40,30))
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
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.savefig('Grafico sigma{}.png'.format(i+1))
    plt.xlabel(r'$\sigma_{input} (a.u.)$',fontsize=40)
    plt.ylabel(r'$\sigma_{output} (a.u.)$',fontsize=40)
    plt.legend(loc='best',fontsize=40)
    plt.savefig('Tudo junto.png')
