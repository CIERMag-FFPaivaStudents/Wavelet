#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Calculates evaluation metrics for 2D signal filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
from signal_2D import Add_Noise2D
import seaborn as sns
from scipy.ndimage import convolve
import os
import SimpleITK as sitk

def SNR_Measure(I,Ihat):
    """Measure of signal to noise ratio as defined in Coupé et al.
    
    Parameters
    ----------
    I: array
        Input 2D array of noiseless image.
    I_hat: array
        Input 1D array of filtered image.
        
    Return
    ------
    SNR: float
        Signal to noise ratio of the filtered image.
        
    References
    ----------
        COUPÉ, P.; MANJÓN, J. V.; ROBLES, M.; COLLINS, D. L. Adaptive
        multiresolution non-local means filter for three-dimensional magnetic resonance
        image denoising. IET Image Processing, [S. l.], v. 6, n. 5, p. 558, 2012. DOI:
        10.1049/iet-ipr.2011.0161. Available at: https://digitallibrary.theiet.org/content/journals/10.1049/iet-ipr.2011.0161.
    """
    n = np.sum(np.power(I,2))
    diff = np.subtract(I,Ihat)
    d = np.sum(np.power(diff,2))
    SNR = 10*np.log10(n/d)
    return SNR

def CoC_Measure(I,Ihat):
    """Measure of the coefficient of correlation as defined in Coupé et al.
    
    Parameters
    ----------
    I: array
        Input 2D array of noiseless image.
    I_hat: array
        Input 1D array of filtered image.
        
    Return
    ------
    CoC: float
        Coefficient of correlation between the noiseless image and the filtered image.
        
    References
    ----------
        COUPÉ, P.; MANJÓN, J. V.; ROBLES, M.; COLLINS, D. L. Adaptive
        multiresolution non-local means filter for three-dimensional magnetic resonance
        image denoising. IET Image Processing, [S. l.], v. 6, n. 5, p. 558, 2012. DOI:
        10.1049/iet-ipr.2011.0161. Available at: https://digitallibrary.theiet.org/content/journals/10.1049/iet-ipr.2011.0161.
    """
    meanI = np.mean(I)
    meanIhat = np.mean(Ihat)
    n = np.sum(np.multiply(I-meanI,Ihat-meanIhat))
    d = np.sqrt(np.sum(np.power(I-meanI,2))*np.sum(np.power(Ihat-meanIhat,2)))
    CoC = n/d
    return CoC

def EPI_Measure(I, Ihat):
    """Measure of edge preservation index as defined in Coupé et al.
    
    Parameters
    ----------
    I: array
        Input 2D array of noiseless image.
    I_hat: array
        Input 1D array of filtered image.
        
    Return
    ------
    EPI: float
        Edge preservation index of filtered image.
        
    References
    ----------
        COUPÉ, P.; MANJÓN, J. V.; ROBLES, M.; COLLINS, D. L. Adaptive
        multiresolution non-local means filter for three-dimensional magnetic resonance
        image denoising. IET Image Processing, [S. l.], v. 6, n. 5, p. 558, 2012. DOI:
        10.1049/iet-ipr.2011.0161. Available at: https://digitallibrary.theiet.org/content/journals/10.1049/iet-ipr.2011.0161.
    """
    
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
    """Measure of mean structure similarity index measure as defined in Wang et al.
    
    Parameters
    ----------
    I: array
        Input 2D array of noiseless image.
        
    I_hat: array
        Input 1D array of filtered image.
        
    Return
    ------
    MSSIM: float
        Measure of mean structure similarity index measure of the filtered image in relation to the noiseledd image.
        
    References
    ----------
        WANG, Z.; BOVIK, A. C.; SHEIKH, H. R.; SIMONCELLI, E. P. Image Quality
        Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on
        Image Processing, [S. l.], v. 13, n. 4, p. 600–612, 2004. DOI: 10.1109/TIP.2003.819861.
        Available at: http://ieeexplore.ieee.org/document/1284395/.
    """
    
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
    
    
    noise_level_list=np.arange(0,100,5)

    length_input=len(noise_level_list)

    num_metrics=4

    metrics_mean_list=np.zeros((length_input,num_metrics+1))
    metrics_std_dev_list=np.zeros((length_input,num_metrics+1))
    
    os.chdir('..')
    path = os.path.abspath('BrainWeb')

    pond = 't1'
    name = pond+'.nii'
    image = sitk.ReadImage(path+'/'+name)
    I=sitk.GetArrayViewFromImage(image)
    slice_num=90
    I=np.flip(I[slice_num])


    name_mask='t1_otsu_3d.nii'
    path_mask = os.path.abspath('BrainWeb')
    mask = sitk.ReadImage(path_mask+'/'+name_mask)
    mask=sitk.GetArrayViewFromImage(mask)
    # mask = mask>0
    mask=mask.astype(bool)
    mask=mask[slice_num]

    os.chdir('2D')
    
    # J = Add_Noise2D(I,noise_level,mask)
    for j in range(length_input):
        # continue
        noise_level=noise_level_list[j]
        
        metrics_matrix=np.zeros((num_iterations,num_metrics+1))
        
        for i in range(num_iterations):
            
            
            J = Add_Noise2D(I,noise_level,mask)

            

            

            metrics_list = np.zeros(num_metrics+1)
            
            metrics_list[0] =   noise_level
            
            SNR = SNR_Measure(I, J)
            metrics_list[1] = SNR
            
            CoC = CoC_Measure(I, J)
            metrics_list[2] = CoC
            
            EPI = EPI_Measure(I, J)
            metrics_list[2] = EPI
            
            MSSIM = MSSIM_Measure(I, J)
            metrics_list[1] = MSSIM
            

            

            metrics_matrix[i]=metrics_list

        # # continue
        # metrics_matrix = metrics_matrix.T

        # metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
        # metrics_mean = np.array(metrics_mean).T
        # metrics_mean_list[j]=metrics_mean
    
        # metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
        # metrics_std_dev = np.array(metrics_std_dev).T
        # metrics_std_dev_list[j]=metrics_std_dev



    # # Escrevendo na tabela
    # Nome='Metricas_2D'
    # tabela = open(Nome+'.txt','w')
    
    
    # nomes_metricas=['Sigma_mean','Sigma_std_dev','SNR_mean','SNR_std_dev','CoC_mean','CoC_std_dev','EPI_mean','EPI_std_dev','MSSIM_mean','MSSIM_std_dev']
    # var='#'
    # var+='\t'.join(nomes_metricas)
    # var+='\n'
    # tabela.write(var)
    
    # pontos=''
    # for j in range(length_input):
    #     for k in range(num_metrics+1):
    #       m,s=metrics_mean_list[j][k],metrics_std_dev_list[j][k]
    #       pontos+='{}\t{}'.format(m,s)
    #       pontos+= '\t'
    #     pontos+='\n'
    # tabela.write(pontos)
    
    # tabela.close()
    
    
    

    
    # Metrics = np.loadtxt('Metricas.txt').T
    
    # symbols=['h','*','o','^','s','p','H']
    # plt.figure(figsize=(12,9))
    # sns.set()
    # sns.set_style('ticks')
    # sns.set_context('talk')
    
    # nomes_metricas =['MSE','SNR','PRD']
    # m0,s0 = Metrics[0:2]
    # for i in range(num_metrics):
    #     plt.clf()
    #     mi,si = Metrics[2*(i+1):2*(i+2)]
    #     plt.errorbar(m0,mi,yerr=si,color='k',marker='o',linestyle='none')
    #     plt.xlabel('$\sigma_{input} (a.u.)$')
    #     plt.ylabel(nomes_metricas[i])
    #     plt.savefig('Grafico '+nomes_metricas[i]+'.png')

        