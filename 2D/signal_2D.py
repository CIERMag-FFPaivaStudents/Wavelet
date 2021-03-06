
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to get an MRI simulation and adds noise to it.
It also plots a two figures, one to illustrate the signal and another one to show noise
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import seaborn as sns

def Add_Noise2D(I,noise_level,mask):
    """Adds rician noise to a 2D magnetic ressonance image as described in Wiest-Daesslé et al.
    Parameters
    ----------
    I: array 
        MRI image.
        
    noise_level: float]
        Desired noise level of corrupted image.
        
    mask: array
        Region of interest binary mask.
        
    Return
    ------
    signal_noise: array
        Noisy simulated free induction decay 1D signal.

    References
    ----------
        WIEST-DAESSLÉ, Nicolas; PRIMA, Sylvain; COUPÉ, Pierrick; MORRISSEY,
        Sean Patrick; BARILLOT, Christian. Rician Noise Removal by Non-Local Means
        Filtering for Low Signal-to-Noise Ratio MRI: Applications to DT-MRI. In: [s.l: s.n.]. p. 
        171–179. DOI: 10.1007/978-3-540-85990-1_21. Available at: http://link.springer.com/10.1007/978-3-540-85990-1_21.
    """
    mask=mask.astype(bool)
    mu = 0
    sigma=float(noise_level/100)*np.mean(I[mask])
    real = (1/np.sqrt(2))*I+np.random.normal(mu,sigma,I.shape)
    imag = (1/np.sqrt(2))*I+np.random.normal(mu,sigma,I.shape)
    noised_image=np.sqrt(real**2 + imag**2)
    return noised_image


def Signal_figure(name,I,mask):
    """Plots a figure designed to show the influences of the image parameters and creates a .png image of it.

    Parameters
    ----------

    name: string
        Desired name of the image.
    
    I: array 
        MRI image.
    
    mask: array
        Region of interest binary mask.

    Return
    ------

    References
    ----------
    """
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')

    fig=plt.figure(figsize=(20,20))

    gs = fig.add_gridspec(2,2)
    
    ax1=fig.add_subplot(gs[0, 0:1])
    ax1.imshow(I,cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Noiseless image',fontsize=40)

    ax2=fig.add_subplot(gs[0, 1:2])
    ax2.imshow(mask,cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Mask',fontsize=40)

    ax3=fig.add_subplot(gs[1, 0:])
    hist, bins = np.histogram(I,80)
    ax3.plot(bins[:-1],hist,'k')
    ax3.fill_between(bins[:-1], hist,color='black')
    ax3.set_title('Noiseless image histogram',fontsize=40)
    ax3.set_ylabel('Number os pixels',fontsize=40)
    ax3.set_xlabel('Value',fontsize=40)
    ax3.set_xlim(0,750)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    os.chdir('Figures')
    plt.savefig(name+'.png')
    os.chdir('..')
    
    return None



def Noise_figure(name,I,mask):
    """Plots a figure designed to show the influences of the noise parameters and creates a .png image of it.

    Parameters
    ----------

    name: string
        Desired name of the image.
    
    I: array 
        MRI image.
    
    mask: array
        Region of interest binary mask.

    Return
    ------

    References
    ----------
    """
    
    fig=plt.figure(figsize=(40,20))
    

    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')

    gs = fig.add_gridspec(5,15)
    
    ax1=fig.add_subplot(gs[0:2,6:9])
    ax1.imshow(I,cmap='gray')
    ax1.set_xticks([]),ax1.set_yticks([])
    ax1.set_title('Imagem sem ruído',fontsize=40)

    for i in range(5):
        ax_image=fig.add_subplot(gs[2:4, 3*i:3*(i+1)])
        noise_level=20*(i+1)
        J=Add_Noise2D(I,noise_level,mask)
        ax_image.imshow(J,cmap='gray')
        ax_image.set_xticks([]),ax_image.set_yticks([])
        ax_image.set_title("{}% de ruído".format(noise_level),fontsize=40)

        ax_hist=fig.add_subplot(gs[4,3*i:3*(i+1)])
        hist, bins = np.histogram(J,80)
        ax_hist.plot(bins[:-1],hist,'k')
        ax_hist.fill_between(bins[:-1], hist,color='black')
        ax_hist.set_title('Histograma da imagem com {}% de ruído'.format(noise_level))
        ax_hist.set_ylabel('Número de pixels',fontsize=30)
        ax_hist.set_xlabel('Valor',fontsize=30)
        ax_hist.set_ylim(0,1555)
        ax_hist.set_xlim(0,2200)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)


    os.chdir('Figures')
    plt.tight_layout()
    plt.savefig(name+'.png')
    os.chdir('..')
    
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

    name = 'Signal figure'
    Signal_figure(name,I,mask)
    
    name='Noise figure'
    Noise_figure(name,I,mask)
