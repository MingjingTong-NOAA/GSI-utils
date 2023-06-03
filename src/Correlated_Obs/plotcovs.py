#!/usr/bin/env python

import numpy as np
import os, sys
from matplotlib import pyplot as plt
from scipy.io import FortranFile
import matplotlib.ticker as ticker
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

def main():

    fighome = '/scratch2/GFDL/gfdlscr/Mingjing.Tong/GSI/GSI-utils/src/Correlated_Obs/figures'
    datadir = '/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/RadStat/shield_edmfdf'
    erradj = ['Error_noinf','Error_inf1p5']
    #instr = 'iasi'
    #sensors = [f'{instr}_metop-b',f'{instr}_metop-b',f'{instr}_metop-c',f'{instr}_metop-c']
    #stype = ['sea','land','sea','land']
    instr = 'cris'
    sensors = [f'{instr}-fsr_n20']
    stype = ['sea']
    if len(sensors) != len(stype):
        print ('sensors and stypes have different length')
        sys.exit()
    fcorr=[]; fwaven=[]; fchnum=[]; finfoerr=[]; ferr=[]
    for sen, st in zip(sensors,stype):
        fcorr.append(f'Rcorr_{sen}_{st}')
        fwaven.append(f'wave_{sen}_{st}')
        fchnum.append(f'chnum_{sen}_{st}')
        finfoerr.append(f'satinfo_err_{sen}_{st}')
        ferr.append(f'err_{sen}_{st}')

    """ figure setup """
    xticklabel=10
    save_figure=True

    if not os.path.exists(fighome):
        os.makedirs(fighome)

    figs = []; fignames = []
    chnum=[]; waven=[]; infoerr=[]; derr=[]; corr=[]
    xl=[]; xl2=[]
    i = 0
    for eadj in erradj: 
        j = 0
        for sen in sensors:
            """ read channel number """
            fileName=f'{datadir}/{eadj}/{fchnum[j]}'
            print (fileName)
            with open(fileName, mode='rb') as f:
                chnum.append(np.fromfile(f,dtype='>i4'))
        
            """ read wave number """
            fileName=f'{datadir}/{eadj}/{fwaven[j]}'
            print (fileName)
            with open(fileName, mode='rb') as f:
                waven.append(np.fromfile(f,dtype='>f4'))
            
            """ read satinfo err """
            fileName=f'{datadir}/{eadj}/{finfoerr[j]}'
            print (fileName)
            with open(fileName, mode='rb') as f:
                err = np.fromfile(f,dtype='>f4')
                infoerr.append(err)
    
            """ read err """
            fileName=f'{datadir}/{eadj}/{ferr[j]}'
            print (fileName)
            with open(fileName, mode='rb') as f: 
                err = np.fromfile(f,dtype='>f4')
                derr.append(err)
    
            """ read corr """
            fileName=f'{datadir}/{eadj}/{fcorr[j]}'
            print (fileName)
            with open(fileName,'rb') as f:
                corrb = np.fromfile(f,dtype='>f4')
                ni = int(np.sqrt(corrb.size))
                corr.append(corrb.reshape((ni,ni)))
                print ('corr ', corr[i].shape)
    
            xl=xl+waven[i].tolist()
            xl2=xl2+chnum[i].tolist()

            j+=1
            i+=1
    
    wn=sorted(list(set(xl))) 
    chn=sorted(list(set(xl2)))
    print (wn)
    print (chn)
    xtick,xlabel,xlabel2 = get_xlabel(instr,np.array(wn),np.array(chn),xticklabel=xticklabel)
    print (xtick)
    print (xlabel)

    """ plot obs error """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    i = 0
    for eadj in erradj:
        for sen in sensors:
            x = np.arange(waven[i].size)
            if i == 0 or (i > 0 and not np.array_equal(infoerr[i],infoerr[i-1])):
                ax.plot(x,infoerr[i])
            i+=1
    ax.xaxis.set_major_locator(ticker.FixedLocator(xtick))
    ax.xaxis.set_ticklabels(xlabel, rotation=45)
    ax.set_xlabel("Wavenumber $(cm^{-1})$")

    ax2 = ax.twiny()
    i = 0
    for eadj in erradj:
        j = 0
        for sen in sensors:
            print (eadj,sen,i)
            x = np.arange(chnum[i].size)
            ax2.plot(x,derr[i],ls='--',label=f'{sensors[j]}_{stype[j]}')
            j+=1
            i+=1
    ax2.xaxis.set_major_locator(ticker.FixedLocator(xtick))
    ax2.xaxis.set_ticklabels(xlabel2)
    ax2.set_xlabel("Channel number")
    ax2.legend()

    figs.append(fig)
    fignames.append(f'{instr}_observation_error')

    """ plot cor matrix """
    i = 0
    for eadj in erradj:
        j = 0
        for sen in sensors: 
            fig, ax = plt.subplots(figsize=(10, 10))
            x = np.arange(waven[i].size)
            xticks = waven[i][::10]
            yticks = waven[i][::10]
            sns.heatmap(corr[i],
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                xticklabels=False, 
                yticklabels=False,
            square=True, ax=ax)
            ax.set(xticks=x[::10], yticks=x[::10])
            xtl = ax.set_xticklabels(xticks)
            ytl = ax.set_yticklabels(yticks)
            #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=10)
            ax.set_ylabel('Wavenumber ($cm^{-1}$)', fontsize=10)
            ax.set_title(f'{sen}_{stype[j]}')
            plt.xticks(rotation=90)

            figs.append(fig)
            fignames.append(f'{eadj}_{sen}_{stype[j]}_corr_matrix')
            j+=1
            i+=1

    if save_figure:
        for fig,figname in zip(figs,fignames):
            figname = fighome+'/%s' % figname
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

