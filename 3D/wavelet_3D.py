
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code extracts a volume from a downloaded 3D image and calculates its wavelet transform
"""


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import pywt


if __name__ == '__main__':
    os.chdir('..')
    path = os.path.abspath('BrainWeb')
    pond = 't1'
    name = pond+'.nii'
    image = sitk.ReadImage(path+'/'+name)
    I=sitk.GetArrayViewFromImage(image)

    os.chdir('3D')

    wavelet='haar'

    levels_dwt=2
    
    coef_list=pywt.wavedecn(I,wavelet,level=levels_dwt)
    
    aprox_coef=coef_list[0]
    detail_coef_dict=coef_list[1]

    os.chdir('Data/Coefficients')


    aprox_coef_name='aaa'
    name_file_aprox=pond+'_'+aprox_coef_name+'_3'+'.nii'
    nift_file_aprox = sitk.GetImageFromArray(aprox_coef)
    

    sitk.WriteImage(nift_file_aprox,name_file_aprox)
    
    for detail_coef_name in detail_coef_dict.keys():
        name_file_detail=pond+'_'+detail_coef_name+'_3'+'.nii'
        nift_file_detail = sitk.GetImageFromArray(detail_coef_dict[detail_coef_name])
