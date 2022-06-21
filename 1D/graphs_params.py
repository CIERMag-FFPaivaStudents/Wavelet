#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is plot graphs.
"""


#Sigma_mean	Sigma_std_dev	MSE0_h_mean	MSE0_h_std_dev	MSE1_h_mean	MSE1_h_std_dev	SNR0_h_mean	SNR0_h_std_dev	SNR1_h_mean	SNR1_h_std_dev	PRD0_h_mean	PRD0_h_std_dev	PRD1_h_mean	PRD1_h_std_dev
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # fs=20
    # num_params=12
    # os.chdir('Data/Metrics_filter')
    # data_name = 'Metrics_filter1D_{}.txt'.format(num_params)
    # data = np.loadtxt(data_name).T
    
    # metrics_num_list = [2,6,10]
    # metrics_name = ['MSE','SNR','PRD']
    
    # os.chdir('..')
    # os.chdir('..')
    # os.chdir('Graphs/Metrics')
    # sigmas = data[0]


    # fig = plt.figure(figsize=(12,9))
    
    # gs = fig.add_gridspec(2,4)
    
    # i=0
    # var = metrics_name[i]
    # k = metrics_num_list[i]
    # ax = fig.add_subplot(gs[0,0:2])
    # ax.plot(sigmas,data[k],'r-',lw=3)
    # ax.plot(sigmas,data[k+2],'g-',lw=3)
    # ax.set_xlabel(r'$\sigma$',fontsize=fs)
    # ax.set_ylabel(var,fontsize=fs)
    # ax.set_xticks([])
    # ax.set_yticks([])


    # i=1
    # var = metrics_name[i]
    # k = metrics_num_list[i]
    # ax = fig.add_subplot(gs[1,1:3])
    # ax.plot(sigmas,data[k],'r-',lw=3)
    # ax.plot(sigmas,data[k+2],'g-',lw=3)
    # ax.set_xlabel(r'$\sigma$',fontsize=fs)
    # ax.set_ylabel(var,fontsize=fs)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # i=2
    # var = metrics_name[i]
    # k = metrics_num_list[i]
    # ax = fig.add_subplot(gs[0,2:4])
    # ax.plot(sigmas,data[k],'r-',lw=3)
    # ax.plot(sigmas,data[k+2],'g-',lw=3)
    # ax.set_xlabel(r'$\sigma$',fontsize=fs)
    # ax.set_ylabel(var,fontsize=fs)
    # ax.set_xticks([])
    # ax.set_yticks([])


    # plt.savefig(' referencia.png',bbox_inches='tight')
    # plt.close()

    
    # os.chdir('..')
    # os.chdir('..')
    
    
    
    fs=15
    for num_params in range(1,25):
        print(num_params)
        os.chdir('Data/Metrics_filter')
        data_name = 'Metrics_filter1D_{}.txt'.format(num_params)
        data = np.loadtxt(data_name).T
        
        metrics_num_list = [2,6,10]
        metrics_name = ['MSE','SNR','PRD']
        metrics_lim = [0,1,-40,30,0,50]

        os.chdir('..')
        os.chdir('..')

        
        os.chdir('Graphs/Metrics/ns')
        sigmas = data[0]
        for i in range(len(metrics_name)):
            var = metrics_name[i]
            k = metrics_num_list[i]
            plt.figure(figsize=(12,9))
            plt.errorbar(sigmas,data[k],data[k+1],marker='o',linestyle='--',label='noisy')
            plt.errorbar(sigmas,data[k+2],data[k+3],marker='o',linestyle='--',label='smooth')
            plt.xlabel(r'$\sigma$',fontsize=fs)
            plt.ylabel(var,fontsize=fs)
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            # plt.ylim(metrics_lim[2*i],metrics_lim[2*i+1])
            plt.legend(loc='best',fontsize=fs)
            plt.savefig(var+' {}.png'.format(num_params),bbox_inches='tight')
            plt.close()
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
    

    metrics_num_list = [2,6,10]
    metrics_name = ['MSE','SNR','PRD']
    metrics_lim = [0,0.3,-30,30,0,25]
        

    F = ['db1','db4','db8']
    T = ['hard','soft']
    N = [2,3,4,6]



    data_name = 'Metrics_filter1D_'
    bla = [0,4,8]
    

    fs=15
    color_plot = ['r','g']
    for i in range(len(metrics_name)):
        var = metrics_name[i]
        k = metrics_num_list[i]
        plt.figure(figsize=(12,9))

        for j in range(2):
            os.chdir('Data/Metrics_filter')
            data = np.loadtxt(data_name + '{}.txt'.format(23*j + 1)).T
            os.chdir('..')
            os.chdir('..')
            sigmas = data[0]
            plt.errorbar(sigmas,data[k+2],data[k+3],marker='o',color=color_plot[j],linestyle='--',label='%d'%(23*j+1))


        plt.xlabel(r'$\sigma$',fontsize=fs)
        plt.ylabel(var,fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend(loc='best',fontsize=fs)
        # plt.ylim(metrics_lim[2*i],metrics_lim[2*i+1])

        os.chdir('Graphs/Metrics')
        plt.savefig(metrics_name[i]+' best_worst.png',bbox_inches='tight')
        plt.close()
        
        os.chdir('..')
        os.chdir('..')

    
    
    
    metrics_num_list = [2,6,10]
    metrics_name = ['MSE','SNR','PRD']
    metrics_lim = [0,0.25,-12,50,0,3.3]
        

    F = ['db1','db4','db8']
    T = ['hard','soft']
    N = [2,3,4,6]

    for t in range(len(T)):
        for n in range(len(N)):
    # t=0
    # n=0
            num = 12*t + 1 + n
            print(T[t],N[n], '-' ,num, num+4,num+8)
        
            data_name = 'Metrics_filter1D_'
            bla = [0,4,8]
            
        
            fs=15
            for i in range(len(metrics_name)):
                var = metrics_name[i]
                k = metrics_num_list[i]
                plt.figure(figsize=(12,9))
                plt.title(T[t]+' thresholding decomposto em {} níveis'.format(N[n]),fontsize=fs)
        
                for f in range(len(F)):
                    os.chdir('Data/Metrics_filter')
                    data = np.loadtxt(data_name + '{}.txt'.format(12*t + 1 + n+4*f)).T
                    os.chdir('..')
                    os.chdir('..')
                    sigmas = data[0]
                    plt.errorbar(sigmas,data[k+2],data[k+3],marker='o',linestyle='--',label=F[f])
        
        
                plt.xlabel(r'$\sigma$',fontsize=fs)
                plt.ylabel(var,fontsize=fs)
                plt.xticks(fontsize=fs)
                plt.yticks(fontsize=fs)
                plt.legend(loc='best',fontsize=fs)
                # plt.ylim(metrics_lim[2*i],metrics_lim[2*i+1])
    
                os.chdir('Graphs/Metrics/Var F')
                plt.savefig('Var F_'+var+'_'+T[t]+' {}.png'.format(N[n]),bbox_inches='tight')
                plt.close()
    
                os.chdir('..')
                os.chdir('..')
                os.chdir('..')



    for f in range(len(F)):
        for n in range(len(N)):
            
            num = 4*f+n+1
            print(F[f],N[n], '-' ,num+12*0, num+12*1)
        
            data_name = 'Metrics_filter1D_'
            bla = [0,4,8]
            
        
            fs=15
            for i in range(len(metrics_name)):
                var = metrics_name[i]
                k = metrics_num_list[i]
                plt.figure(figsize=(12,9))
                plt.title(F[f]+' decomposto em {} níveis'.format(N[n]),fontsize=fs)
        
                for t in range(len(T)):
                    os.chdir('Data/Metrics_filter')
                    data = np.loadtxt(data_name + '{}.txt'.format(num+12*t)).T
                    os.chdir('..')
                    os.chdir('..')
                    sigmas = data[0]
                    plt.errorbar(sigmas,data[k+2],data[k+3],marker='o',linestyle='--',label=T[t])
        
        
                plt.xlabel(r'$\sigma$',fontsize=fs)
                plt.ylabel(var,fontsize=fs)
                plt.xticks(fontsize=fs)
                plt.yticks(fontsize=fs)
                plt.legend(loc='best',fontsize=fs)
                # plt.ylim(metrics_lim[2*i],metrics_lim[2*i+1])

                
                os.chdir('Graphs/Metrics/Var T')
                plt.savefig('Var T_'+var+'_'+F[f]+' {}.png'.format(N[n]),bbox_inches='tight')
                plt.close()
                os.chdir('..')
                os.chdir('..')
                os.chdir('..')



    for f in range(len(F)):
        for t in range(len(T)):
            num = 1 + 12*t + 4*f
            print(T[t],F[f], '-' ,num, num+1,num+2, num+3)
        
            data_name = 'Metrics_filter1D_'
            bla = [0,4,8]
            
        
            fs=15
            for i in range(len(metrics_name)):
                var = metrics_name[i]
                k = metrics_num_list[i]
                plt.figure(figsize=(12,9))
                plt.title(F[f]+' '+T[t],fontsize=fs)
        
                for n in range(len(N)):
                    os.chdir('Data/Metrics_filter')
                    data = np.loadtxt(data_name + '{}.txt'.format(num+n)).T
                    os.chdir('..')
                    os.chdir('..')
                    sigmas = data[0]
                    plt.errorbar(sigmas,data[k+2],data[k+3],marker='o',linestyle='--',label=N[n])
        
        
                plt.xlabel(r'$\sigma$',fontsize=fs)
                plt.ylabel(var,fontsize=fs)
                plt.xticks(fontsize=fs)
                plt.yticks(fontsize=fs)
                plt.legend(loc='best',fontsize=fs)
                # plt.ylim(metrics_lim[2*i],metrics_lim[2*i+1])
            
            
                os.chdir('Graphs/Metrics/Var N')
                plt.savefig('Var N_'+var+'_'+F[f]+' '+T[t]+'.png',bbox_inches='tight')
                os.chdir('..')
                os.chdir('..')
                os.chdir('..')
