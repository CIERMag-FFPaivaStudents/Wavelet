
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to get an MRI simulation and add noise to it.
It also plots a bunch of figures
"""
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import SimpleITK as sitk
import seaborn as sns
import signal_2D
import seaborn as sns


def DWT_2D(image,wavelet,levels):
    """Calculates the two dimensional Discrete Wavelet Transform of a 2D magnetic ressonance image.
    Parameters
    ----------
    image: array
        MRI image.
        
    wavelet: Wavelet object or name string
        Desired mother wavelet in the DWT.
        
    levels: int
        Number of decomposition levels.
        
    Return
    ------
    approx_coef_list: array
        List of approximation coefficients from the lowest level to the highest.
    
    hor_detail_coef_list: array
        List of horizontal detail coefficients from the lowest level to the highest.
    
    vert_detail_coef_list: array
        List of vertical detail coefficients from the lowest level to the highest.

    vert_detail_coef_list: array
        List of diagonal detail coefficients from the lowest level to the highest.
    """
    
    approx_coef_list = (levels+1)*[None]
    hor_detail_coef_list = (levels+1)*[None]
    vert_detail_coef_list = (levels+1)*[None]
    diag_detail_coef_list = (levels+1)*[None]

    approx_coef_list[0]=image
    hor_detail_coef_list[0]=0*image
    vert_detail_coef_list[0]=0*image
    diag_detail_coef_list[0]=0*image
    
    for level in range(levels):
        coeffs2 = pywt.dwt2(approx_coef_list[level], wavelet)
        approx_coef, (hor_detail_coef, vert_detail_coef, diag_detail_coef) = coeffs2
        approx_coef_list[level+1]=approx_coef
        hor_detail_coef_list[level+1] = hor_detail_coef
        vert_detail_coef_list[level+1] = vert_detail_coef
        diag_detail_coef_list[level+1] = diag_detail_coef

    return approx_coef_list,hor_detail_coef_list,vert_detail_coef_list,diag_detail_coef_list


def Multilevel_plot(aprox_list,hor_list,vert_list,diag_list,levels_dwt):
    """Creates a plot of the most common way to visualize 2D DWT coefficients.
    ----------
    approx_coef_list: array
        List of approximation coefficients from the lowest level to the highest.
    
    hor_detail_coef_list: array
        List of horizontal detail coefficients from the lowest level to the highest.
    
    vert_detail_coef_list: array
        List of vertical detail coefficients from the lowest level to the highest.

    vert_detail_coef_list: array
        List of diagonal detail coefficients from the lowest level to the highest.
        
    levels_dwt: int
        Number of decomposition levels of the DWT transform.
        
    Return
    ------
    """
    
    fig=plt.figure(figsize=(31,36))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')

    gs = fig.add_gridspec(2**levels_dwt,2**levels_dwt)

    ax = fig.add_subplot(gs[0:1,0:1])
    ax.imshow(aprox_list[1],cmap='gray')
    ax.set_xticks([]),ax.set_yticks([])

    for i in range(levels_dwt):
        ax = fig.add_subplot(gs[0:2**i,2**i:2**(i+1)])
        ax.imshow(hor_list[-(i+1)],cmap='gray')
        ax.set_xticks([]),ax.set_yticks([])
        
        ax = fig.add_subplot(gs[2**i:2**(i+1),0:2**i])
        ax.imshow(vert_list[-(i+1)],cmap='gray')
        ax.set_xticks([]),ax.set_yticks([])
        
        ax = fig.add_subplot(gs[2**i:2**(i+1),2**i:2**(i+1)])
        ax.imshow(diag_list[-(i+1)],cmap='gray')
        ax.set_xticks([]),ax.set_yticks([])
    
    plt.tight_layout()
    return None




if __name__ == '__main__':
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
    mask = mask>0
    mask=mask.astype(bool)
    mask=mask[slice_num]
    
    os.chdir('2D')
    
    
    mother_wavelet='db'
    number_wavelet=4
    wavelet=mother_wavelet+'%d'%number_wavelet

    levels_dwt=3

    J=signal_2D.Add_Noise2D(I,20,mask)
    noise=J-I

    levels_dwt = 2

    aprox_list,hor_list,vert_list,diag_list=DWT_2D(J,wavelet,levels_dwt)

    Multilevel_plot(aprox_list,hor_list,vert_list,diag_list,levels_dwt)

    os.chdir('Figures')
    plt.savefig('Multilevel_2.png',bbox_inches='tight')




