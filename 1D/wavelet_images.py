#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Plot the wavelet filter banks

"""

import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__=='__main__':
    

    plt.style.use('default')
    
    wavelet_list = ['db1','db4','db8','coif5','coif10','coif17','sym8','sym18','sym20']
    
    xlim_list = [15,15,15,100,100,100,40,40,40]



    
    # Simple code 
    
    wavelet='sym10'
        
        
    # a,b=1,4
    # fotos=plt.figure(figsize=(5*b,5*a))
    # sns.set()
    # sns.set_context('talk')
    # sns.set_style('white')
    
    
    
    
    # f1=fotos.add_subplot(a,b,1)
    # f2=fotos.add_subplot(a,b,2)
    # f3=fotos.add_subplot(a,b,3)
    # f4=fotos.add_subplot(a,b,4)

    # w= pywt.Wavelet(wavelet)
    # (phi,psi,x)=w.wavefun(level=10)

    # f1.grid()
    # f1.plot(x,phi)
    # f1.set_title('Scale Function' + '('+r'$\varphi$'+')')

    # f2.grid()
    # f2.plot(x,psi)
    # f2.set_title('Wavelet Function' + '('+r'$\psi$'+')')
    # (dec_lo,dec_hi,rec_lo,rec_hi)=w.filter_bank 
    # f3.stem(dec_lo)

    # f3.grid()
    # f3.set_title(r'$h_\varphi$')
    # f4.stem(dec_hi)

    # f4.grid()
    # f4.set_title(r'$h_\psi$')
    # plt.suptitle(wavelet.capitalize())
    # plt.tight_layout()
        
    a,b=1,3
    fotos=plt.figure(figsize=(18*b,12*a))
    sns.set()
    sns.set_context('talk')
    sns.set_style('white')
    i=1
    for wavelet in ['db5','coif5','sym5']:

        fi=fotos.add_subplot(a,b,i)
    
        w= pywt.Wavelet(wavelet)
        (phi,psi,x)=w.wavefun(level=10)
    
        fi.plot(x,psi,'k-',linewidth=10)
        fi.set_xticks([])
        fi.set_yticks([])
        i+=1
    plt.savefig('every wavelet.svg')
    # just the waveforms
    
    # for wavelet in wavelet_list:
    
    #     a,b=1,2
    #     fig=plt.figure(figsize=(5*b,5*a))
        
    
    #     sns.set()
    #     sns.set_context('talk')
    #     sns.set_style('white')
        
        
    #     ax1 = fig.add_subplot(a,b,1)
    #     ax2 = fig.add_subplot(a,b,2)

        
    
    #     w= pywt.Wavelet(wavelet)
    #     (phi,psi,x)=w.wavefun(level=10)
        
        
    #     ax1.grid()
    #     ax1.plot(x,phi)
    #     ax1.set_title('Scale Function' + '('+r'$\varphi$'+')')
        
    #     ax2.grid()
    #     ax2.plot(x,psi)
    #     ax2.set_title('Wavelet Function' + '('+r'$\psi$'+')')
        
    #     plt.suptitle(wavelet.capitalize())
    #     plt.tight_layout()

    #     os.chdir('Figures/Wavelets')
    #     # plt.savefig(wavelet.capitalize()+'_1.png')
    #     os.chdir('..')
    #     os.chdir('..')
    #     # plt.close()
    
    #just the filter banks
    # for wavelet in wavelet_list:
    #     a,b=1,2
    #     fig=plt.figure(figsize=(5*b,5*a))

    #     w= pywt.Wavelet(wavelet)
    
    #     sns.set()
    #     sns.set_context('talk')
    #     sns.set_style('white')
        
        
    #     ax3 = fig.add_subplot(a,b,1)
    #     ax4 = fig.add_subplot(a,b,2)
        
        
    #     (dec_lo,dec_hi,rec_lo,rec_hi)=w.filter_bank 
        
    #     ax3.stem(dec_lo)
    #     ax3.grid()
    #     ax3.set_title(r'$h_\varphi$')
    
    #     ax4.stem(dec_hi)
    #     ax4.grid()
    #     ax4.set_title(r'$h_\psi$')
        
        
    #     plt.suptitle(wavelet.capitalize())
    #     plt.tight_layout()
    #     os.chdir('Figures/Wavelets')
    #     plt.savefig(wavelet.capitalize()+'_2.png')
    #     os.chdir('..')
    #     os.chdir('..')
        
    #     plt.close()
        

    #just the scale function
    # i=0
    # # for wavelet in [wavelet_list[0]]:
    # for wavelet in wavelet_list:
    
    #     a,b=1,2
    #     fig=plt.figure(figsize=(10,5))
        
    
    #     sns.set()
    #     sns.set_context('talk')
    #     sns.set_style('white')
        
        
    #     ax2 = fig.add_subplot(a,b,1)
    #     ax4 = fig.add_subplot(a,b,2)

        
    
    #     w= pywt.Wavelet(wavelet)
    #     (phi,psi,x)=w.wavefun(level=10)
    #     (dec_lo,dec_hi,rec_lo,rec_hi)=w.filter_bank 
        
        
        

        
    #     ax2.grid()
    #     ax2.plot(x,psi)
    #     ax2.set_title('Wavelet Function' + '('+r'$\psi$'+')',fontsize = 28)
    #     ax2.set_xlim(0,xlim_list[i])
    #     ax2.set_ylim(-1.5,1.5)
    #     ax2.tick_params(axis='x', labelsize=26)
    #     ax2.tick_params(axis='y', labelsize=26)
    #     # ax2.set_xticks(color='blue')
    #     # ax2.set_xlim(100)
        
    #     ax4.stem(dec_hi)
    #     ax4.grid()
    #     ax4.set_title(r'$h_\psi$',fontsize = 28)
    #     ax4.set_xticks([])
    #     ax4.set_ylim(-0.8,0.8)
    #     ax4.tick_params(axis='x', labelsize=26)
    #     ax4.tick_params(axis='y', labelsize=26)
    #     plt.suptitle(wavelet.capitalize(),fontsize=30)
    #     plt.tight_layout()

    #     os.chdir('Figures/Wavelets/bla')
    #     plt.savefig(wavelet.capitalize()+'_psi.svg')
    #     os.chdir('..')
    #     os.chdir('..')
    #     os.chdir('..')
    #     plt.close()
        
    #     i+=1
        
    # for wavelet in wavelet_list:
    #     a,b=1,2
    #     fig=plt.figure(figsize=(5*b,5*a))

    #     w= pywt.Wavelet(wavelet)
    #     (phi,psi,x)=w.wavefun(level=10)
    #     (dec_lo,dec_hi,rec_lo,rec_hi)=w.filter_bank 
        
    
    #     sns.set()
    #     sns.set_context('talk')
    #     sns.set_style('white')
        
        
    #     ax1 = fig.add_subplot(a,b,1)
    #     ax3 = fig.add_subplot(a,b,2)
        
 

        
    #     ax1.grid()
    #     ax1.plot(x,phi)
    #     ax1.set_title('Scale Function' + '('+r'$\varphi$'+')')
        
        
        
    #     ax3.stem(dec_lo)
    #     ax3.grid()
    #     ax3.set_title(r'$h_\varphi$')
    

        
        
        # plt.suptitle(wavelet.capitalize())
        # plt.tight_layout()
        # os.chdir('Figures/Wavelets')
        # plt.savefig(wavelet.capitalize()+'_phi.png')
        # os.chdir('..')
        # os.chdir('..')
        
        # plt.close()
