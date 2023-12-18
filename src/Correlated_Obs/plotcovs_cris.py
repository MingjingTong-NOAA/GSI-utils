#!/usr/bin/env python

import numpy as np
import os, sys
from matplotlib import pyplot as plt
from scipy.io import FortranFile
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns

def get_xlabel(instr,waven,chnum,xticklabel=10):

    if isinstance(xticklabel, int) and xticklabel > 1:
        xtickevery = xticklabel
        xlabel=waven[::xtickevery]
        xlabel2=chnum[::xtickevery]
        xtick=np.arange(waven.size)[::xtickevery]
    else:
        xlabel=[waven[0]]
        xlabel2=[chnum[0]]
        xtick=[0]
        if 'iasi' in instr:
            """ middle temperature sounding """
            idx = (np.abs(waven - 706.25)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ lower temperature sounding """
            idx = (np.abs(waven - 721.75)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ window """
            idx = (np.abs(waven - 742.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ ozone """
            idx = (np.abs(waven - 1014.5)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            idx = (np.abs(waven - 1062.5)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ humidity """
            idx = (np.abs(waven - 1096.0)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
        else:
            """ window """
            idx = (np.abs(waven - 706.25)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ tropospheric-sensitive sounding """
            idx = (np.abs(waven - 710.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ window """
            idx = (np.abs(waven - 850.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ ozone """
            idx = (np.abs(waven - 980.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            idx = (np.abs(waven - 1095.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ humidity """
            idx = (np.abs(waven - 1300.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ window """
            idx = (np.abs(waven - 1688.125)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            idx = (np.abs(waven - 1,703.125)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
            """ humidity """
            idx = (np.abs(waven - 1750.)).argmin()
            xlabel.append(waven[idx])
            xtick.append(idx)
            xlabel2.append(chnum[idx])
    
        """ last channel """ 
        xlabel.append(waven[-1])
        xlabel2.append(chnum[-1])
        xtick.append(waven.size - 1)
    
    return xtick, xlabel, xlabel2

def savefigure(
        fh=None,
        fname='test',
        format=[
            'png',
            'eps',
            'pdf'],
    orientation='landscape',
        dpi=100):
    '''
    Save a figure in png, eps and pdf formats
    '''

    if fh is None:
        fh = _plt
    if 'png' in format:
        fh.savefig(
            '%s.png' %
            fname,
            format='png',
            dpi=1 *
            dpi,
            orientation=orientation)
    if 'eps' in format:
        fh.savefig(
            '%s.eps' %
            fname,
            format='eps',
            dpi=2 *
            dpi,
            orientation=orientation)
    if 'pdf' in format:
        fh.savefig(
            '%s.pdf' %
            fname,
            format='pdf',
            dpi=2 *
            dpi,
            orientation=orientation)

    return

def check(list):
    return all(i == list[0] for i in list)

def main():

    fighome = '/scratch2/GFDL/gfdlscr/Mingjing.Tong/GSI/GSI-utils/src/Correlated_Obs/figures'
    #datadir = '/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/RadStat/shield_edmfdf'
    datadir = '/scratch2/GFDL/gfdlscr/Mingjing.Tong/noscrub/Rcov'
    instr = 'cris'
    config = {'HL_raw_50km': 'HL_rc_50km',
              'HL_raw_80km': 'HL_raw_80km',
              'HL_raw_100km': 'HL_raw_100km',
              'HL_rc_50km': 'HL_rc_50km',
              'HL_rc_80km': 'HL_rc_80km',
              'HL_rc_100km': 'HL_rc_100km',
              'D_rc': 'D_rc_it1',
              'D_raw': 'D_raw_it1',
              'Diagnosed': 'D_raw_it1',
              'Reconditioned': 'D_rc_it1',
              'inflated': 'D_rc_inf_it1',
              'Reconditioned_it2': 'D_rc_it2',
              'inflated_it2': 'D_rc_inf_it2',
             }
    erradj = ['Diagnosed', 'Reconditioned', 'inflated']
    #erradj = ['HL_raw_50km','HL_raw_80km','HL_raw_100km','D_raw']
    #erradj = ['HL_rc_50km','HL_rc_80km','HL_rc_100km','D_rc']
    sensor = [f'{instr}-fsr_n20']
    stype = ['sea']
    if len(sensor) != len(erradj):
        if len(sensor) == 1:
            sensor = sensor * len(erradj)
        elif len(erradj) == 1:
            erradj = erradj * len(sensor)
        else:
            print ('sensor and erradj have different length')
            sys.exit()
    if len(stype) != len(sensor):
        if len(stype) == 1:
            stype = stype * len(sensor)
        else:
            print ('sensor and stype have different length')
    print (sensor)
    print (stype)
    print (erradj)
    stradj = "-".join(erradj)
    strsen = "-".join(sensor)
    strtyp = "-".join(stype)
    
    fcorr=[]; fwaven=[]; fchnum=[]; finfoerr=[]; ferr=[]; fnpair=[]
    feigs=[]
    for eadj, sen, st in zip(erradj,sensor,stype):
        fcorr.append(f'{datadir}/{config[eadj]}/Error/Rcorr_{sen}_{st}')
        fwaven.append(f'{datadir}/{config[eadj]}/Error/wave_{sen}_{st}')
        fchnum.append(f'{datadir}/{config[eadj]}/Error/chnum_{sen}_{st}')
        finfoerr.append(f'{datadir}/{config[eadj]}/Error/satinfo_err_{sen}_{st}')
        ferr.append(f'{datadir}/{config[eadj]}/Error/err_{sen}_{st}')
        fnpair.append(f'{datadir}/{config[eadj]}/Error/npair_{sen}_{st}')
        feigs.append(f'{datadir}/{config[eadj]}/Error/eigs_{sen}_{st}')

    """ figure setup """
    xticklabel=10
    plotobserr=True
    plotcorr=True
    plotobspair=True
    """ note that eigenvalues of raw maxtrix could be < 0.0 or a complex number """
    ploteigval=True
    ploteigvec=True
    save_figure=True

    for i, adj in enumerate(erradj):
        if i == 0:
            expstr=adj
        else:
            expstr += f'_{adj}'

    figdir=f'{fighome}/{stradj}/{strsen}/{strtyp}'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    figs = []; fignames = []
    chnum=[]; waven=[]; infoerr=[]; derr=[]; corr=[]; npair=[]
    eigs=[]; kreq=[]; infl=[]
    xl=[]; xl2=[]; labels=[]; titles=[]
    i = 0
    for eadj, sen, st in zip(erradj, sensor, stype): 
        """ read channel number """
        fileName=f'{fchnum[i]}'
        print (fileName)
        with open(fileName, mode='rb') as f:
            chnum.append(np.fromfile(f,dtype='>i4'))
        
        """ read wave number """
        fileName=f'{fwaven[i]}'
        print (fileName)
        with open(fileName, mode='rb') as f:
            waven.append(np.fromfile(f,dtype='>f4'))
            
        """ read satinfo err """
        fileName=f'{finfoerr[i]}'
        print (fileName)
        with open(fileName, mode='rb') as f:
            err = np.fromfile(f,dtype='>f4')
            infoerr.append(err)
    
        if plotobserr:
            """ read err """
            fileName=f'{ferr[i]}'
            print (fileName)
            with open(fileName, mode='rb') as f: 
                err = np.fromfile(f,dtype='>f4')
                derr.append(err)
    
        if plotcorr:
            """ read corr """
            fileName=f'{fcorr[i]}'
            print (fileName)
            with open(fileName,'rb') as f:
                corrb = np.fromfile(f,dtype='>f4')
                ni = int(np.sqrt(corrb.size))
                corr.append(corrb.reshape((ni,ni)))
                #print ('corr ', corr[i].shape)

        if plotobspair:
            """ read npair """
            fileName=f'{fnpair[i]}'
            print (fileName)
            if os.path.isfile(fileName):
                with open(fileName,'rb') as f:
                    npairb = np.fromfile(f,dtype='>f4')
                    print ('npairb ', npairb.shape, config[eadj])
                    if 'hl' in config[eadj] or 'HL' in config[eadj]:
                        ni = int(np.sqrt(npairb.size / 3))
                        npair.append(npairb.reshape((3,ni,ni)))
                    else:
                        print ('npairb.size ', npairb.size)
                        ni = int(np.sqrt(npairb.size))
                        npair.append(npairb.reshape((ni,ni)))
            else:
                npair.append(None)

        if ploteigval or ploteigvec:
            """ read eigenvalue """
            fileName=f'{feigs[i]}'
            print (fileName)
            if os.path.isfile(fileName):
                with open(fileName,'rb') as f:
                    eigsb = np.fromfile(f,dtype='>f4')
                    print (eigsb.shape)
                    kreq.append(eigsb[0])
                    infl.append(eigsb[1])
                    eigs.append(eigsb[2:])
            else:
                print (f'missing {fileName}')
                kreq.append(None)
                infl.append(None)
                eigs.append(None)

        xl=xl+waven[i].tolist()
        xl2=xl2+chnum[i].tolist()

        titles.append(f'{eadj}_{sensor[i]}_{stype[i]}')
        labels.append(f'{eadj}')

        i+=1
    wn=sorted(list(set(xl))) 
    chn=sorted(list(set(xl2)))
    #print (wn)
    #print (chn)
    xtick,xlabel,xlabel2 = get_xlabel(instr,np.array(wn),np.array(chn),xticklabel=xticklabel)
    #print (xtick)
    #print (xlabel)

    """ plot obs error """
    if plotobserr:
        print ('plot obs error')
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.subplots_adjust(bottom=0.2)

        samesensor=check(sensor)
        samestype=check(stype)
        
        i = 0
        for eadj, sen, st in zip(erradj, sensor, stype):
            x = np.arange(waven[i].size)
            if i == 0 or (i > 0 and not np.array_equal(infoerr[i],infoerr[0])):
                ax.plot(x,infoerr[i],label='old')
            i+=1
        ax.xaxis.set_major_locator(ticker.FixedLocator(xtick))
        ax.xaxis.set_ticklabels(xlabel, rotation=45,fontsize=8)
        ax.set_xlabel("Wavenumber $(cm^{-1})$")
        ax.set_ylabel("Error (K)")
        if samesensor and samestype:
            ax.set_title(f'{sensor[0]}_{stype[0]}')
            ax.title.set_size(12)

        ax2 = ax.twiny()
        i = 0
        for eadj, sen, st in zip(erradj, sensor, stype):
            x = np.arange(chnum[i].size)
            if samesensor and samestype:
                ax2.plot(x,derr[i],ls='--',label=labels[i])
            else:
                ax2.plot(x,derr[i],ls='--',label=titles[i])
            i+=1
        ax2.xaxis.set_major_locator(ticker.FixedLocator(xtick))
        ax2.xaxis.set_ticklabels(xlabel2,fontsize=8)
        ax2.set_xlabel("Channel number")
        ax2.legend(loc=2,frameon=False)

        figs.append(fig)
        if samesensor and samestype:
            fignames.append(f'{sensor[0]}_{stype[0]}_observation_error')
        else:
            fignames.append(f'{instr}_observation_error')

    """ plot cor matrix """
    if plotcorr:
        print ('plot cor matrix')
        i = 0
        for eadj, sen, st in zip(erradj, sensor, stype):
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.subplots_adjust(left=0.2,bottom=0.2)
            x = np.arange(waven[i].size)
            #ncolors = 8
            ncolors = 40
            im = sns.heatmap(corr[i],
                 cmap=sns.color_palette("coolwarm",n_colors=ncolors),
                 vmin=-1.0, vmax=1.0,
                 xticklabels=False, 
                 yticklabels=False,
                 cbar_kws={"shrink": 0.8},
                 square=True, ax=ax)
            ax.set(xticks=xtick, yticks=xtick)
            xtl = ax.set_xticklabels(xlabel, fontsize=10)
            ytl = ax.set_yticklabels(xlabel, fontsize=10)
            #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=10)
            ax.set_ylabel('Wavenumber ($cm^{-1}$)', fontsize=10)
            ax.set_title(titles[i])
            ax.title.set_size(10)
            plt.xticks(rotation=90)

            figs.append(fig)
            fignames.append(f'{eadj}_{sen}_{stype[i]}_corr_matrix')
            i+=1

    """ plot npair matrix """
    if plotobspair:
        print ('plot npair matrix')
        i = 0
        for eadj, sen, st in zip(erradj, sensor, stype):
            print (eadj, sen, st)
            if eadj == 'cloud1':
                print (npair[i])
            if npair[i] is not None: 
                x = np.arange(waven[i].size)
                if npair[i].ndim > 2:
                    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12, 8))
                    ax=axs[0]
                    im = ax.matshow(npair[i][0,:,:])
                else:
                    fig, ax = plt.subplots(figsize=(8, 8)) 
                    im = ax.matshow(npair[i])
                ax.set(xticks=xtick, yticks=xtick)
                xtl = ax.set_xticklabels(xlabel, rotation=90)
                ytl = ax.set_yticklabels(xlabel)
                ax.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=10)
                ax.set_ylabel('Wavenumber ($cm^{-1}$)', fontsize=10)
                plt.colorbar(im, location='bottom', pad=0.1, ax=ax)

                if npair[i].ndim > 2:       
                    ax=axs[1]
                    im = ax.matshow(npair[i][1,:,:])
                    ax.set(xticks=xtick, yticks=xtick)
                    xtl = ax.set_xticklabels(xlabel, rotation=90)
                    ytl = ax.set_yticklabels(xlabel)
                    ax.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=10)
                    plt.colorbar(im, location='bottom', pad=0.1, ax=ax)

                plt.suptitle(titles[i])
                figs.append(fig)
                fignames.append(f'{eadj}_{sen}_{st}_npair_matrix')
                i+=1
    
    if ploteigval:
        print ('plot eigenvalue')
        i = 0
        for eadj, sen, st in zip(erradj, sensor, stype):
            if kreq[i] is not None:
                if kreq[i] > 0.0:
                    eign=int(len(eigs[i])/3)
                else:
                    eign=int(len(eigs[i]))
                x=range(1,eign+1)
                fig, ax = plt.subplots(figsize=(8, 8))
                eigsor=eigs[i][:eign]
                ax.plot(x,np.sort(eigsor)[::-1],label='original')
                print ('eigenvalue org')
                print (np.sort(eigsor)[::-1])
                if kreq[i] > 0.0:
                    eigsrg=eigs[i][eign:2*eign] 
                    eigsfn=eigs[i][2*eign:]
                    ax.plot(x,np.sort(eigsrg)[::-1],label='recondition')
                    ax.plot(x,np.sort(eigsfn)[::-1],label='final')
                    print ('eigenvalue recondition')
                    print (np.sort(eigsrg)[::-1])
                    print ('eigenvalue change due to recondition')
                    eigdiff = np.sort(eigsrg)[::-1]-np.sort(eigsor)[::-1]
                    eignorm = eigdiff/np.sort(eigsrg)[::-1]
                    print (eadj)
                    print (eignorm)
                    print (eignorm[eignorm > 1.0e-03])
                    print (eignorm[eignorm > 1.0e-03].size)
          
                ax.legend(frameon=False)
                ax.set_xlabel('Eigenvector number')
                ax.set_ylabel('Eigenvalue') 
                ax.set_title(titles[i])
                figs.append(fig)
                fignames.append(f'{eadj}_{sen}_{stype[i]}_eigenvalue')
            i+=1

    if ploteigvec:
        print ('plot eigenvector')
        i = 0
        for eadj, sen, st in zip(erradj, sensor, stype):
            """ compute eigenvalue&eigenvector for correlation matrix 
                note: eigenvector is the column of the eigenvector matrix """
            eigenValues, eigenVectors = np.linalg.eig(corr[i])
            nev=int(eigenValues.size)
            """ sort eigenvalue and eigenvector """
            idx = eigenValues.argsort()[::-1]   
            eigenValues = eigenValues[idx]
            eigenVectors = eigenVectors[:,idx]
            #print (eigenValues)
            #print (eigenVectors)
            #print (eigenValues[0]*eigenVectors[:,0])
            #print (np.dot(corr[i],eigenVectors[:,0]))

            eigvlist=[0,1,2,nev-3,nev-2,nev-1]

            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2,3)
            for j, ev in enumerate(eigvlist):
                ax=plt.subplot(gs[j])
                x = np.arange(waven[i].size)
                ax.plot(x,eigenVectors[:,ev])
                ax.xaxis.set_major_locator(ticker.FixedLocator(xtick))
                ax.xaxis.set_ticklabels(xlabel, rotation=45,fontsize=7)
                if j > 2:
                    ax.set_xlabel("Wavenumber $(cm^{-1})$")
                if j == 0 or j == 3:
                    ax.set_ylabel('Eigenvector')
                plt.hlines(0.,x[0],x[-1],colors='k',linestyles='-',linewidth=1.5,label=None)
                print ('eigenValues ', eigenValues[ev])
                if eigenValues[ev] > 0.0:
                    tev=np.sqrt(eigenValues[ev])
                    eiginfo=f'eigenvector: {ev+1}, sqrt(ev)={str(np.round(tev,2))}'
                else:
                    tev=eigenValues[ev]
                    eiginfo=f'eigenvector: {ev+1}, ev={str(np.round(tev,2))}'
                ax.text(0.015, 0.95, eiginfo,
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes,
                    color='black', fontsize=7)
 
                if j < 3:
                    ax2 = ax.twiny()
                    x = np.arange(chnum[i].size)
                    ax2.plot(x,eigenVectors[:,ev],ls='--',label=labels[i])
                    ax2.xaxis.set_major_locator(ticker.FixedLocator(xtick))
                    ax2.xaxis.set_ticklabels(xlabel2, rotation=45, fontsize=7)
                    ax2.set_xlabel("Channel number")
    
            fig.suptitle(f'{titles[i]}')
            figs.append(fig)
            fignames.append(f'{eadj}_{sen}_{stype[i]}_eigenvector')
            i+=1

    if save_figure:
        for fig,figname in zip(figs,fignames):
            figname = figdir+'/%s' % figname
            print (figname)
            savefigure(fig,figname,format='png')
    else:
        plt.show()

    sys.exit(0)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print (type(e))
        sys.exit(2)

