#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date: 2016-09-26 11:18:55 -0400 (Mon, 26 Sep 2016) $
# $Revision: 82099 $
# $Author: Michael.Lueken@noaa.gov $
# $Id: proc_gsistat.py 82099 2016-09-26 15:18:55Z Michael.Lueken@noaa.gov $
###############################################################

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as st
from datetime import datetime
from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib import gridspec as gspec
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from matplotlib.ticker import MaxNLocator
from glob import glob

sys.path.append('../lib')
import lib_GSI_mtong as lgsi

def _float10Power(value):
    if value == 0:
        return 0
    e = np.log10(abs(value))
    e = np.ceil(e) - 1. if e >= 0 else np.floor(e)
    return e

def get_data1(gsistat,varname,it=1,use='asm',typ='all'):
    df = []
    for i in gsistat:
        tmp = i.extract(varname)
        if tmp.empty:
            print ('missing %s %s'%(varname, i.analysis_date.strftime('%Y%m%d%H')))
        else:
            tmp = tmp.xs((it,use),level=['it','use'],drop_level=False)
            if typ != 'all':
                tmp2 = []
                if not isinstance(typ,list): typ = [typ]
                indx = tmp.index.get_level_values('typ') == ''
                for t in typ:
                    indx = np.ma.logical_or(indx,tmp.index.get_level_values('typ') == t)
                tmp2.append(tmp.iloc[indx])
                #unique_types = tmp.index.get_level_values('typ').unique()
                #for t in typ:
                #    if t in unique_types:
                #        tmp2.append(tmp.xs(t,level='typ',drop_level=False))
                #    else:
                #        print '%s is not present in %s' % (t, i.analysis_date.strftime('%Y%m%d%H'))
                tmp = pd.concat(tmp2)

            """ sum over data types """
            if varname in ['ps', 'sst', 'tcp']:
                tmp = tmp.groupby(level=['date','it','obs','use']).sum()
            elif varname in ['uv', 't', 'q', 'gps', 'amv']:
                tmp = tmp.groupby(level=['date','it','obs','use','stat']).sum()
            else:
                msg = 'get_data1: varname %s is not a valid variable\n' % varname
                msg += 'try: ps, uv, t, q'
                raise KeyError(msg)

            df.append(tmp)

    lendf = len(df)
    if lendf != 0:
        df = pd.concat(df)

    return df

def get_data2(gsistat,varname,select=None,level=None):
    df = []
    for i in gsistat:
        tmp = i.extract(varname)
        if select is not None:
            tmp = tmp.xs(select,level=level,drop_level=False)
        df.append(tmp)
    df = pd.concat(df)
    return df

def get_inst_data(gsistat,instname,select=None,level=None,plotanl=False,usedonly=True):
    df = []
    for n, i in enumerate(gsistat):
        tmp = i.extract_instrument('rad',instname,plotanl=plotanl,usedonly=usedonly)
        if tmp.empty or len(tmp) == 0:
            print ('file %s missing %s or no data available'%(str(n),instname))
        else: 
            if select is not None:
                tmp = tmp.xs(select,level=level,drop_level=False)
            df.append(tmp)
    
    if not tmp.empty:
        lendf = len(df)
        if lendf != 0:
            df = pd.concat(df)
    else:
        lendf=0
    
    return df, lendf

def plot_ps(ps,ps2=None,pname='ps'):

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(111,frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    cell_text = []; row_labels = []
    for e,expid in enumerate(labels):
        txt = ps[expid].mean()
        count,rms,bias = int(txt['count']),'%8.5f'%txt['rms'],'%8.5f'%txt['bias']
        cell_text.append([count,rms,bias])
        row_labels.append(labels[e]+' FG')
        if ps2 is not None:
            txt = ps2[expid].mean()
            count,rms,bias = int(txt['count']),'%8.5f'%txt['rms'],'%8.5f'%txt['bias']
            cell_text.append([count,rms,bias])
            row_labels.append(labels[e]+' ANL')

    col_labels = ['count','rms','bias']
    plt.table(cellText=cell_text,cellLoc='center',rowLabels=row_labels,colLabels=col_labels,loc='center')
    if pname == 'ps':
        plt.title('Surface Pressure\n%s'%title_substr,fontsize='x-large',fontweight='bold')
    else:
        plt.title('TcVitle MSP\n%s'%title_substr,fontsize='x-large',fontweight='bold')

    return fig

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return [m, h]

def plot_profile(uv, t, q, gps, amv, scrm, stat='rms', anl=False, normalize=False, 
                 pairdiff=False, diffsum=False, ylog=False, nlegend=False,
                 pltvar=['uv','t','q','gps','amv','scrm']):

    fontsize=12
    lw=2.0
    if len(pltvar) >=5:
        fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(top=0.875, hspace=0.3)
        gs = gspec.GridSpec(2, 3)
    else:
        if len(pltvar) == 1:
            fig = plt.figure(figsize=(4, 3))
            gs = gspec.GridSpec(1,1)
            plt.subplots_adjust(bottom=0.15, left=0.15)
            fontsize=8
            lw=1.0
        elif len(pltvar) == 3:
            fig = plt.figure(figsize=(11, 3))
            gs = gspec.GridSpec(1,3)
            plt.subplots_adjust(top=0.875, bottom=0.15, left=0.15)
            fontsize=8
        else:
            fig = plt.figure(figsize=(10, 8))
            plt.subplots_adjust(top=0.875, hspace=0.3)
            gs = gspec.GridSpec(2, 2)

    if ylog:
        lmin,lmax = 1100,50

    for v,var in enumerate(pltvar):
        if var == 'q':
            indexlevs = [1200, 1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200]
            levs = [ 1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200 ]
        else:
            indexlevs = [1200,1000, 900, 800, 600, 400, 300, 250, 200, 150, 100, 50]
            levs = [1000, 900, 800, 600, 400, 300, 250, 200, 150, 100, 50]
        ya = np.arange(len(levs))
        yb = np.arange(len(indexlevs))
        if not ylog:
            lmin = yb[0]
            lmax = yb[-1]
  
        data_dict = eval(var)

        xmin = 1.e10
        xmax = 0
        ax = plt.subplot(gs[v])

        #cntldf=data_dict[labels[0]].xs(stat, level='stat', drop_level=False)
        #count_cntl=data_dict[labels[0]].xs('count', level='stat', drop_level=False)
        cntldf=data_dict[labels[0]].xs(stat, level='stat')
        count_cntl=data_dict[labels[0]].xs('count', level='stat')
        #cntldf.where(count_cntl > 0, np.nan, inplace=True) 
        ncount = cntldf.shape[0]
        """ [:-1] remove whole column [0:2000] stats"""
        profilec=cntldf.mean()[:-1].values
        for e,expid in enumerate(labels):
            #expdf=data_dict[expid].xs(stat, level='stat', drop_level=False)
            expdf=data_dict[expid].xs(stat, level='stat')

            profile = expdf.mean()[:-1].values
            if diffsum:
                diff=expdf.sub(cntldf,axis=1) 
                profile0 = diff.sum()[:-1].values
                #print ('diffsum')
                #print diff
                #print (var, profile0)
            elif pairdiff:
                diff=expdf.sub(cntldf,axis=1)
                if ncount < 12:
                    profile0=diff.mean()[:-1].values
                else:
                    profile=diff.apply(mean_confidence_interval).values
            elif normalize:
                normdf=expdf.div(cntldf,axis=1)
                normdf.where(count_cntl > 0, np.nan, inplace=True)
                if ncount < 12:
                    profile0=normdf.mean()[:-1].values
                else:
                    profile=normdf.apply(mean_confidence_interval).values
            else:
                if ncount < 12:
                    profile0=expdf.mean()[:-1].values
                else:
                    profile=expdf.apply(mean_confidence_interval).values

            """if var == 't' and stat == 'rms':
                print 'profile'
                print profile """
            if ncount >= 12 and not diffsum:
                if profile.ndim == 1:
                    profile0=np.array([x[0] for x in profile[:-1] ])
                    CI_95=np.array([x[1] for x in profile[:-1] ])
                else:
                    profile0=np.array([x for x in profile[0,:-1] ])
                    CI_95=np.array([x for x in profile[1,:-1] ])

            if pairdiff:
                profilen = profile0/profilec
                if ncount >= 12:
                    CI_95n = CI_95/profilec
            elif normalize:
                profilen = profile0*100.
                if ncount >= 12:
                    CI_95n = CI_95*100.
    
            #countprofile=data_dict[expid].xs('count', level='stat', drop_level=False).mean()[:-1].values
            countprofile=data_dict[expid].xs('count', level='stat').mean()[:-1].values
            #if pairdiff:
            #    print 'countprofile'
            #    print countprofile
            profile0[countprofile == 0.0]=np.nan
            if ncount >= 12 and not diffsum:
                CI_95[countprofile == 0.0]=np.nan

            # Normalize counts by 10^exponent for clarity
            if stat == 'count' and not normalize and not pairdiff:
                exponent = _float10Power(profile.max())
                #profile = profile0 / np.power(10,exponent)
                profile = profile0

            elevs = np.array(levs) + e
            if ylog:
                if stat == 'rms' and ncount >= 12:
                    yindex = elevs
                else:
                    yindex = levs
            else:
                yindex = ya+0.5

            if (normalize or pairdiff) and len(labels) > 1:
                if e == 0:
                    if pairdiff:
                        plt.vlines(0.0,lmin,lmax,colors='k',linestyles='--',linewidth=lw,label=None)
                    else:
                        plt.vlines(100.,lmin,lmax,colors='k',linestyles='--',linewidth=lw,label=None)
                else:
                    if stat == 'rms':
                        if ncount >= 12:
                            tyidx=yindex+(e-1)*0.05
                            ax.errorbar(profilen, tyidx, xerr=CI_95n, label=labels[e], color=mc[e], 
                                        elinewidth=lw,linewidth=lw) 
                        else:
                            ax.plot(profilen, yindex, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e],
                                    linewidth=lw, alpha=alpha)
                    elif stat == 'count':
                        ax.plot(profilen, yindex, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e],
                                linewidth=lw, alpha=alpha)
            else:
                if stat == 'rms':
                    if ncount >= 12:
                        ax.errorbar(profile0, yindex, xerr=CI_95, label=labels[e], color=mc[e])
                    else:
                        ax.plot(profile0, yindex, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e],
                                linewidth=lw, alpha=alpha)
                elif stat == 'count':
                    ax.plot(profile, yindex, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e], linewidth=lw, alpha=alpha)
                else:
                    ax.plot(profile0, yindex, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e],
                            linewidth=lw, alpha=alpha)

            if e == 0 and stat == 'bias':
                plt.vlines(0.,lmin,lmax,colors='k',linestyles='--',linewidth=lw,label=None)
    
            if (normalize or pairdiff) and ncount >= 12:
                #print profilen.shape, profilen
                #print CI_95n.shape, CI_95n
                tmp1=profilen[~np.isnan(profilen)]
                if ncount >= 12:
                    tmp2=CI_95n[~np.isnan(CI_95n)]
                if stat == 'rms' and ncount >= 12:
                    xmin_,xmax_ = np.min(tmp1-tmp2),np.max(tmp1+tmp2) 
                else:
                    xmin_,xmax_ = np.min(tmp1),np.max(tmp1)
            #else:
            #    print profile0[~np.isnan(profile0)]
            #    xmin_,xmax_ = np.min(profile0[~np.isnan(profile0)]), np.max(profile0[~np.isnan(profile0)])
            #if ( xmin_ < xmin ): xmin = xmin_
            #if ( xmax_ > xmax ): xmax = xmax_
    
        if len(pltvar) >=5:
            if ( v in [0,3] ): plt.ylabel('Pressure (hPa)',fontsize=fontsize)
            if ( v in [5] ): plt.legend(loc=0,fontsize='small',numpoints=1,frameon=False)
        else:
            if len(pltvar) == 1:
                plt.ylabel('Pressure (hPa)',fontsize=fontsize)
            else:
                if ( v in [0,2] ): plt.ylabel('Pressure (hPa)',fontsize=fontsize)
            if ( v in [0] and not nlegend ): plt.legend(loc=0,fontsize='small',numpoints=1,frameon=False)

        if ( var == 'uv' ):
            var_unit = 'm/s'
            var_name = 'Vector Wind'
            plabel = '(a)'
        elif ( var == 't' ):
            var_unit = 'K'
            var_name = 'Temperature'
            plabel = '(b)'
        elif ( var == 'q' ):
            var_unit = 'frac'
            var_name = 'Sp. Humidity'
            plabel = '(c)'
        elif ( var == 'gps' ):
            var_unit = 'rad'
            var_name = 'GPS'
            plabel = '(d)'
        elif ( var == 'amv' ):
            var_unit = 'm/s'
            var_name = 'AMVs'
            plabel = '(e)'
        elif (var == 'scrm' ):
            var_unit = 'm/s'
            var_name = 'Scatterometeor'
            plabel = '(f)'
        
        if normalize or pairdiff:
            if pairdiff:
                plt.xlabel('normalized difference',fontsize=fontsize)
            else:
                if stat == 'rms':
                    plt.xlabel('normalized rms O-F (%)',fontsize=fontsize)
                else:
                    plt.xlabel('normalized data count (%)',fontsize=fontsize)
        else:
            if stat in ['rms','bias']:
                plt.xlabel('(%s)' % var_unit,fontsize=fontsize)
            else:
                plt.xlabel('count', fontsize=fontsize)

        if len(pltvar) > 1:
            if normalize or pairdiff:
                    if stat == 'rms':
                        if anl:
                            plt.suptitle('%s O-A\n%s' % (stat.upper(),title_substr),fontsize='x-large',fontweight='bold')
                        else:
                            plt.suptitle('%s O-F\n%s' % (stat.upper(),title_substr),fontsize='x-large',fontweight='bold')
                    elif stat == 'count':
                        plt.suptitle('Normalized Observation Counts\n%s'%title_substr,fontsize='x-large',fontweight='bold')                
            else:
                if stat in ['rms','bias']:
                    if anl:
                        plt.suptitle('%s O-A\n%s' % (stat.upper(),title_substr),fontsize='x-large',fontweight='bold')
                    else:
                        plt.suptitle('%s O-F\n%s' % (stat.upper(),title_substr),fontsize='x-large',fontweight='bold')
                else:
                    plt.suptitle('Observation Counts\n%s'%title_substr,fontsize='x-large',fontweight='bold')
    
        plt.title(var_name,fontsize=fontsize+2)
        if ylog:
            plt.ylim(lmin,lmax)
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.FixedLocator(indexlevs))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(plt.NullLocator())
            if v in [0,3]:
                #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0,subs=np.arange(1,10)))
                plt.yticks(fontsize=fontsize)
            else:
                ax.set_yticklabels([])
        else:
            plt.ylim(ya[0]+0.25,ya[-1]+0.5)
            ax.yaxis.set_ticks(yb[1:])
            ax.yaxis.set_ticklabels(indexlevs[1:])
            plt.yticks(fontsize=fontsize)
            
        #xmin = xmin - (xmax-xmin)*0.1
        #xmax = xmax + (xmax-xmin)*0.1
        if pairdiff:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f")) 
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        elif normalize:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))    
        else:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(8))
            xmin = 0 if stat in ['count'] else xmin - (xmax-xmin)*0.1
        plt.xticks(fontsize=fontsize)
        #plt.xlim(xmin,xmax)

        ax.text(0.015, 1.05, plabel,
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes,
                color='black', fontsize=fontsize+2)

    return fig

def plot_profile_FGvsANL(uv, t, q, gps, amv, scrm, uv_a, t_a, q_a, gps_a, amv_a, scrm_a,
                         expidcntl='CNTL', expid='exp',elabelc='CNTL',elabel='exp',stat='rms',
                         normalize=False, pairdiff=False, ylog=False,
                         pltvar=['uv','t','q','gps','amv','scrm']):

    if len(pltvar) >=5:
        fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(top=0.875, hspace=0.3)
        gs = gspec.GridSpec(2, 3)
    else:
        fig = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(top=0.875, hspace=0.3)
        gs = gspec.GridSpec(2, 2)

    lmin,lmax = 1200,50

    for v,var in enumerate(pltvar):
        if var == 'q':
            indexlevs = [1200, 1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200]
            levs = [ 1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200 ]
        else:
            indexlevs = [1200,1000, 900, 800, 600, 400, 300, 250, 200, 150, 100, 50]
            levs = [1000, 900, 800, 600, 400, 300, 250, 200, 150, 100, 50]
        ya = np.arange(len(levs))
        yb = np.arange(len(indexlevs))
        if not ylog:
            lmin = yb[0]
            lmax = yb[-1]

        xmin = 1.e10
        xmax = 0
        ax = plt.subplot(gs[v])

        data_dictf = eval(var)
        data_dicta = eval(var+'_a')

        #print (data_dictf)

        cntldf_f=data_dictf[expidcntl].xs(stat, level='stat', drop_level=False)
        profilec_f=cntldf_f.mean()[:-1].values
        ncount = profilec_f.shape[0]

        cntldf_a=data_dicta[expidcntl].xs(stat, level='stat', drop_level=False)
        profilec_a=cntldf_a.mean()[:-1].values

        expdf_f=data_dictf[expid].xs(stat, level='stat', drop_level=False)
        expdf_a=data_dicta[expid].xs(stat, level='stat', drop_level=False)

        if pairdiff:
            diff_f=expdf_f.sub(cntldf_f,axis=1)
            diff_a=expdf_a.sub(cntldf_a,axis=1)
            if ncount < 12:
                profile0_f=diff_f.mean()[:-1].values
                profile0_a=diff_a.mean()[:-1].values                
            else:
                profilef=diff_f.apply(mean_confidence_interval).values
                profilea=diff_a.apply(mean_confidence_interval).values
        elif normalize:
            normdf_f=expdf_f.div(cntldf_f,axis=1)
            normdf_a=expdf_a.div(cntldf_a,axis=1)
            if ncount < 12:
                profile0_f=normdf_f.mean()[:-1].values
                profile0_a=normdf_a.mean()[:-1].values
            else:
                profilef=normdf_f.apply(mean_confidence_interval).values
                profilea=normdf_a.apply(mean_confidence_interval).values
        else:
            if ncount < 12:
                profile0_f=expdf_f.mean()[:-1].values
                profile0_a=expdf_a.mean()[:-1].values
            else:
                profilef=expdf_f.apply(mean_confidence_interval).values
                profilea=expdf_a.apply(mean_confidence_interval).values

        if ncount >= 12:
            if profilef.shape[0] == 12:
                profile0_f=np.array([x[0] for x in profilef[:-1] ])
                CI_95_f=np.array([x[1] for x in profilef[:-1] ])
                profile0_a=np.array([x[0] for x in profilea[:-1] ])
                CI_95_a=np.array([x[1] for x in profilea[:-1] ])
            else:
                profile0_f=np.array([x for x in profilef[0,:-1] ])
                CI_95_f=np.array([x for x in profilef[1,:-1] ])
                profile0_a=np.array([x for x in profilea[0,:-1] ])
                CI_95_a=np.array([x for x in profilea[1,:-1] ])

        if pairdiff:
            profilen_f = profile0_f/profilec_f
            profilen_a = profile0_a/profilec_a
            if ncount >= 12:
                CI_95n_f = CI_95_f/profilec_f 
                CI_95n_a = CI_95_a/profilec_a
        elif normalize:
            profilen_f = profile0_f*100.
            profilen_a = profile0_a*100.
            if ncount >= 12:
                CI_95n_f = CI_95_f*100.
                CI_95n_a = CI_95_a*100.

        countprofile=data_dictf[expid].xs('count', level='stat', drop_level=False).mean()[:-1].values
        profile0_f[countprofile == 0.0]=np.nan
        countprofile=data_dicta[expid].xs('count', level='stat', drop_level=False).mean()[:-1].values
        profile0_a[countprofile == 0.0]=np.nan
        if ncount >= 12:
            CI_95_f[countprofile == 0.0]=np.nan
            CI_95_a[countprofile == 0.0]=np.nan

        if pairdiff:
            plt.vlines(0.0,lmin,lmax,colors='k',linestyles='--',linewidth=2.0,label=None)
        elif normalize:
            plt.vlines(100.,lmin,lmax,colors='k',linestyles='--',linewidth=2.0,label=None)

        if stat == 'bias':
            plt.vlines(0.,lmin,lmax,colors='k',linestyles='--',linewidth=2.0,label=None)

        elevs = np.array(levs) + 5
        if ylog:
            if stat == 'rms' and ncount >= 12:
                yindex = elevs
            else:
                yindex = levs
        else:
            yindex = ya+0.5

        if pairdiff or normalize:
            if ncount > 12:
                ax.errorbar(profilen_f, yindex, xerr=CI_95n_f, label='FG', color='b')
                ax.errorbar(profilen_a, yindex, xerr=CI_95n_a, label='Analysis', color='r') 
            else:
                ax.plot(profilen_f, yindex, marker='o', label='FG', color='b', mfc='b', mec='b',
                        linewidth=2.0, alpha=alpha)
                ax.plot(profilen_a, yindex, marker='o', label='Analysis', color='r', mfc='r', mec='r',
                        linewidth=2.0, alpha=alpha)


            tmp1=profilen_f[~np.isnan(profilen_f)]
            if ncount > 12:
                tmp2=CI_95n_f[~np.isnan(CI_95n_f)]
            if stat == 'rms' and ncount > 12:
                 xmin_,xmax_ = np.min(tmp1-tmp2),np.max(tmp1+tmp2)
            else:
                 xmin_,xmax_ = np.min(tmp1),np.max(tmp1)

            tmp1=profilen_a[~np.isnan(profilen_a)]
            if ncount > 12:
                tmp2=CI_95n_a[~np.isnan(CI_95n_a)]
            if stat == 'rms' and ncount > 12:
                xmin,xmax = np.min(tmp1-tmp2),np.max(tmp1+tmp2)
            else:
                xmin,xmax = np.min(tmp1),np.max(tmp1)
            if ( xmin_ < xmin ): xmin = xmin_
            if ( xmax_ > xmax ): xmax = xmax_
        else:
            xmin_,xmax_ = np.min(profile0_f[~np.isnan(profile0_f)]), np.max(profile0_f[~np.isnan(profile0_f)])
            xmin,xmax = np.min(profile0_a[~np.isnan(profile0_a)]), np.max(profile0_a[~np.isnan(profile0_a)])
            if ( xmin_ < xmin ): xmin = xmin_
            if ( xmax_ > xmax ): xmax = xmax_
            
            ax.plot(profile0_f, yindex, marker='o',color='k',mfc='k',mec='k',linewidth=2.0,label='FG',alpha=alpha)
            ax.plot(profile0_a, yindex, marker='o',color='r',mfc='r',mec='r',linewidth=2.0,label='ANL',alpha=alpha)

        if len(pltvar) >=5:
            if ( v in [0,3] ): plt.ylabel('pressure (hPa)',fontsize=12)
            if ( v in [5] ): plt.legend(loc=0,fontsize='small',numpoints=1,frameon=False)
        else:
            if ( v in [0,2] ): plt.ylabel('pressure (hPa)',fontsize=12)
            if ( v in [0] ): plt.legend(loc=0,fontsize='small',numpoints=1,frameon=False)

        if ( var == 'uv' ):
            var_unit = 'm/s'
            var_name = 'Winds'
        elif ( var == 't' ):
            var_unit = 'K'
            var_name = 'Temperature'
        elif ( var == 'q' ):
            var_unit = 'frac'
            var_name = 'Sp. Humidity'
        elif ( var == 'gps' ):
            var_unit = 'rad'
            var_name = 'GPS'
        elif ( var == 'amv' ):
            var_unit = 'm/s'
            var_name = 'AMV'
        elif (var == 'scrm' ):
            var_unit = 'm/s'
            var_name = 'Scatterometeor'

        if normalize or pairdiff:
            if pairdiff:
                plt.xlabel('normalized difference',fontsize=12)
                plt.suptitle('%s O-F/O-A %s-%s\n%s' % (stat.upper(),elabel,elabelc,title_substr),fontsize='x-large',fontweight='bold')
            else:
                plt.xlabel('normalized O-F rms (%)',fontsize=12)
                plt.suptitle('%s O-F/O-A %s/%s\n%s' % (stat.upper(),elabel,elabelc,title_substr),fontsize='x-large',fontweight='bold')
        else:
            plt.xlabel('(%s)' % var_unit,fontsize=12)
            plt.suptitle('%s O-F & O-A %s\n%s' % (stat.upper(),elabel,title_substr),fontsize='x-large',fontweight='bold')


        plt.title(var_name,fontsize='large')
        if ylog:
            plt.ylim(lmin,lmax)
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.FixedLocator(indexlevs))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(plt.NullLocator())
            if v in [0,3]:
                #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0,subs=np.arange(1,10)))
                plt.yticks(fontsize=10)
            else:
                ax.set_yticklabels([])
        else:
            plt.ylim(ya[0],ya[-1]+1.0)
            ax.yaxis.set_ticks(yb)
            ax.yaxis.set_ticklabels(indexlevs)

        xmin = xmin - (xmax-xmin)*0.1
        xmax = xmax + (xmax-xmin)*0.1
        if pairdiff:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        elif normalize:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(8))
            xmin = 0 if stat in ['count'] else xmin - (xmax-xmin)*0.1
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        plt.xticks(fontsize=10)

    return fig

def plot_cost(minim):

    # Collect all experiments into a single DataFrame
    tmpdf = []
    tmpdf2 = []
    for e,expid in enumerate(labels):
        tmp = minim[expid].groupby(level=['Outer','Inner']).mean().dropna()['J']
        tmp.name = labels[e]
        tmpdf.append(tmp)
        tmp = minim[expid]['J']
        tmp.name = labels[e]
        tmpdf2.append(tmp)

    # Scale the cost-function with 1e5
    exponent = 5
    df = pd.concat(tmpdf,axis=1) / np.power(10,exponent)
    df2 = pd.concat(tmpdf2,axis=1) / np.power(10,exponent)

    fig,ax = plt.subplots(figsize=(10,8))

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    # Plot the spaghetti of all dates, followed by the mean
    for idate in df2.index.get_level_values('date').unique():
        tmp = df2.xs(idate,level='date')
        tmp.plot(ax=ax,kind='line',linewidth=0.75,alpha=alpha/2.,color=lc,label=None,legend=False)
    df.plot(ax=ax,kind='line',linewidth=2.,alpha=alpha,color=lc)

    # This is needed to show the second+ outerloops with correct indices
    #xticks = ax.get_xticks()
    #xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #for i, (xtick,xticklabel) in enumerate(zip(xticks,xticklabels)):
    #    tmp = str(xticklabel)
    #    if tmp:
    #        if int(tmp[1]) > 1:
    #            xticks[i] = xticks[i] + 1
    #            it = int(tmp.split(',')[-1].strip().split(')')[0]) + 1
    #            xticklabels[i] = u'(2, %s)' % str(it)
    #ax.set_xticks(xticks[:-1])
    #ax.set_xticklabels(xticklabels[:-1])

    xindex = df.index.get_level_values('Inner')
    a = np.arange(len(xindex))
    it1idx = df.xs(1, level='Outer').index.get_level_values('Inner')
    iouter = len(it1idx)
    xticks = a[:iouter][::25].tolist()
    xlabels = xindex[:iouter][::25].tolist()
    it2idx = df.xs(2, level='Outer').index.get_level_values('Inner')
    xticks = xticks + a[iouter+25:][::25].tolist()
    xlabels = xlabels + xindex[iouter+25:][::25].tolist()
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xlabels)

    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Cost function (x $\mathregular{10^%d}$)' % exponent,fontsize=12)

    ymin,ymax = np.min(df2.min()),np.max(df2.max())
    dy = ymax - ymin
    ymin,ymax = ymin-0.1*dy,ymax+0.1*dy
    plt.ylim(ymin,ymax)
    plt.title('Cost function reduction\n%s'%title_substr,fontsize='x-large',fontweight='bold')

    return fig

def plot_gradient(minim):

    # Collect all experiments into a single DataFrame
    tmpdf = []
    tmpdf2 = []
    for e,expid in enumerate(labels):
        tmp = minim[expid].groupby(level=['Outer','Inner']).mean().dropna()['gJ']
        tmp.name = labels[e]
        tmpdf.append(tmp)
        tmp = minim[expid]['gJ']
        tmp.name = labels[e]
        tmpdf2.append(tmp)

    # Scale the cost-function with 1e5
    df = pd.concat(tmpdf,axis=1)
    df2 = pd.concat(tmpdf2,axis=1)

    fig,ax = plt.subplots(figsize=(10,8))

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    # Plot the spaghetti of all dates, followed by the mean
    for idate in df2.index.get_level_values('date').unique():
        tmp = df2.xs(idate,level='date')
        tmp.plot(ax=ax,kind='line',logy=True,linewidth=0.75,alpha=alpha/2.,color=lc,label=None,legend=False)
    df.plot(ax=ax,kind='line',logy=True,linewidth=2.,alpha=alpha,color=lc)

    # This is needed to show the second+ outerloops with correct indices
    #xticks = ax.get_xticks()
    #xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #for i, (xtick,xticklabel) in enumerate(zip(xticks,xticklabels)):
    #    tmp = str(xticklabel)
    #    if tmp:
    #        if int(tmp[1]) > 1:
    #            xticks[i] = xticks[i] + 1
    #            it = int(tmp.split(',')[-1].strip().split(')')[0]) + 1
    #            xticklabels[i] = u'(2, %s)' % str(it)
    #ax.set_xticks(xticks[:-1])
    #ax.set_xticklabels(xticklabels[:-1])

    xindex = df.index.get_level_values('Inner')
    a = np.arange(len(xindex))
    it1idx = df.xs(1, level='Outer').index.get_level_values('Inner')
    iouter = len(it1idx)
    xticks = a[:iouter][::25].tolist()
    xlabels = xindex[:iouter][::25].tolist()
    it2idx = df.xs(2, level='Outer').index.get_level_values('Inner')
    xticks = xticks + a[iouter+25:][::25].tolist()
    xlabels = xlabels + xindex[iouter+25:][::25].tolist()
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xlabels)

    ax.set_xlabel('Iteration',fontsize=12)
    ax.set_ylabel('Gradient of Cost function',fontsize=12)

    ymin,ymax = np.min(df2.min()),np.max(df2.max())
    dy = ymax - ymin
    ymin,ymax = ymin-0.1*dy,ymax+0.1*dy
    plt.ylim(top=ymax)
    plt.title('Gradient reduction\n%s'%title_substr,fontsize='x-large',fontweight='bold')

    return fig

def plot_cost_gradient(minim):

    fig = plt.figure(figsize=(6,8))
    gs = gspec.GridSpec(2, 1)

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    # Collect all experiments into a single DataFrame
    tmpdf = []
    tmpdf2 = []
    for e,expid in enumerate(labels):
        tmp = minim[expid].groupby(level=['Outer','Inner']).mean().dropna()['J']
        tmp.name = labels[e]
        tmpdf.append(tmp)
        tmp = minim[expid]['J']
        tmp.name = labels[e]
        tmpdf2.append(tmp)

    # Scale the cost-function with 1e5
    exponent = 5
    df = pd.concat(tmpdf,axis=1) / np.power(10,exponent)
    df2 = pd.concat(tmpdf2,axis=1) / np.power(10,exponent)

    ax = plt.subplot(gs[0])

    df.plot(ax=ax,kind='line',linewidth=2.,alpha=alpha,color=lc)
    xindex = df.index.get_level_values('Inner')
    a = np.arange(len(xindex))
    it1idx = df.xs(1, level='Outer').index.get_level_values('Inner')
    iouter = len(it1idx)
    xticks = a[:iouter][::25].tolist()
    xlabels = xindex[:iouter][::25].tolist()
    it2idx = df.xs(2, level='Outer').index.get_level_values('Inner')
    xticks = xticks + a[iouter+25:][::25].tolist()
    xlabels = xlabels + xindex[iouter+25:][::25].tolist()
    #print ('xtick', xticks)
    #print ('xlabels', xlabels)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xlabels)
    # This is needed to show the second+ outerloops with correct indices
    #xticks = ax.get_xticks()
    #xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #for i, (xtick,xticklabel) in enumerate(zip(xticks,xticklabels)):
    #    tmp = str(xticklabel)
    #    if tmp:
    #        if int(tmp[1]) > 1:
    #            xticks[i] = xticks[i] + 1
    #            it = int(tmp.split(',')[-1].strip().split(')')[0]) + 1
    #            xticklabels[i] = u'(2, %s)' % str(it)
    #ax.set_xticks(xticks[:-1])
    #ax.set_xticklabels(xticklabels[:-1])
    ax.set_xlabel('Iteration Number',fontsize=12)
    ax.set_ylabel('Cost Function (x $\mathregular{10^%d}$)' % exponent,fontsize=12)
    ax.legend(fontsize=12,frameon=False)
    ax.text(0.15, 0.85, '(a)',
            verticalalignment='bottom', horizontalalignment='center',
            transform=ax.transAxes,
            color='black', fontsize=12)

    ymin,ymax = np.min(df2.min()),np.max(df2.max())
    dy = ymax - ymin
    ymin,ymax = ymin-0.1*dy,ymax+0.1*dy
    plt.ylim(ymin,ymax)
    #plt.title('Cost Function',fontsize='large')

    # Collect all experiments into a single DataFrame
    tmpdf = []
    tmpdf2 = []
    for e,expid in enumerate(labels):
        tmp = minim[expid].groupby(level=['Outer','Inner']).mean().dropna()['gJ']
        tmp.name = labels[e]
        tmpdf.append(tmp)
        tmp = minim[expid]['gJ']
        tmp.name = labels[e]
        tmpdf2.append(tmp)

    # Scale the cost-function with 1e5
    df = pd.concat(tmpdf,axis=1)
    df2 = pd.concat(tmpdf2,axis=1)

    ax = plt.subplot(gs[1])

    df.plot(ax=ax,kind='line',logy=True,linewidth=2.,alpha=alpha,color=lc)

    # This is needed to show the second+ outerloops with correct indices
    #xticks = ax.get_xticks()
    #xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #for i, (xtick,xticklabel) in enumerate(zip(xticks,xticklabels)):
    #    tmp = str(xticklabel)
    #    if tmp:
    #        if int(tmp[1]) > 1:
    #            xticks[i] = xticks[i] + 1
    #            it = int(tmp.split(',')[-1].strip().split(')')[0]) + 1
    #            xticklabels[i] = u'(2, %s)' % str(it)
    #ax.set_xticks(xticks[:-1])
    #ax.set_xticklabels(xticklabels[:-1])
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xlabels)
    ax.set_xlabel('Iteration Number',fontsize=12)
    ax.set_ylabel('Norm of the Gradient',fontsize=12)
    ax.legend(fontsize=12,frameon=False)
    ax.text(0.15, 0.85, '(b)',
            verticalalignment='bottom', horizontalalignment='center',
            transform=ax.transAxes,
            color='black', fontsize=12)

    ymin,ymax = np.min(df2.min()),np.max(df2.max())
    dy = ymax - ymin
    ymin, ymax = max(1., ymin-0.1*dy), ymax+0.1*dy
    plt.ylim(top=ymax)
    #plt.title('Norm of the Gradient',fontsize='large')

    return fig

def plot_Jo(minim):

    fig,ax = plt.subplots(figsize=(8,6))

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    # Collect all experiments into a single DataFrame
    tmpdf = []
    tmpdf2 = []
    for e,expid in enumerate(labels):
        tmp = minim[expid].groupby(level=['Outer','Inner']).mean().dropna()['Jo']
        tmp.name = labels[e]
        tmpdf.append(tmp)
        tmp = minim[expid]['Jo']
        tmp.name = labels[e]
        tmpdf2.append(tmp)

    # Scale the cost-function with 1e5
    exponent = 5
    df = pd.concat(tmpdf,axis=1) / np.power(10,exponent)
    df2 = pd.concat(tmpdf2,axis=1) / np.power(10,exponent)

    #print (df)
    df.plot(ax=ax,kind='line',linewidth=2.,alpha=alpha,color=lc)

    # This is needed to show the second+ outerloops with correct indices
    xticks = ax.get_xticks()
    xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #print (xticklabels)
    for i, (xtick,xticklabel) in enumerate(zip(xticks,xticklabels)):
        tmp = str(xticklabel)
        if tmp:
            if int(tmp[1]) > 1:
                xticks[i] = xticks[i] + 1
                it = int(tmp.split(',')[-1].strip().split(')')[0]) + 1
                xticklabels[i] = u'(2, %s)' % str(it)
    ax.set_xticks(xticks[:-1])
    ax.set_xticklabels(xticklabels[:-1])
    ax.set_xlabel('Iteration Number',fontsize=12)
    ax.set_ylabel('Jo (x $\mathregular{10^%d}$)' % exponent,fontsize=12)
    ax.legend(fontsize=12,frameon=False)
    ax.text(0.15, 0.85, '(a)',
            verticalalignment='bottom', horizontalalignment='center',
            transform=ax.transAxes,
            color='black', fontsize=12)

    ymin,ymax = np.min(df2.min()),np.max(df2.max())
    dy = ymax - ymin
    ymin,ymax = ymin-0.1*dy,ymax+0.1*dy
    plt.ylim(ymin,ymax)
    #plt.title('Jo',fontsize='large')

    return fig

def get_yticklabels_new(ax):

    yticklabels = ax.get_yticklabels()
    yticklabels_new = []
    instp = None
    for l,lab in enumerate(yticklabels):
        lab = str(lab.get_text())
        inst,sat = lab.replace('(','').replace(')','').split(',')
        if inst == instp:
            new_label = sat
        else:
            instp = inst
            new_label = '%s, %s' % (inst,sat)
        yticklabels_new.append(new_label.upper())

    return yticklabels_new

def get_xticklabels_new(ax):

    xticklabels = ax.get_xticklabels()
    #print xticklabels
    xticklabels_new = []
    instp = None
    for l,lab in enumerate(xticklabels):
        lab = str(lab.get_text())
        #print 'lab=', lab
        inst,sat = lab.replace('(','').replace(')','').split(',')
        if inst == instp:
            new_label = sat
        else:
            instp = inst
            new_label = '%s, %s' % (inst,sat)
        xticklabels_new.append(new_label.upper())

    return xticklabels_new


def plot_sat(dfin,otype=''):

    # Collect all experiments into a single DataFrame
    read,keep,assim = [],[],[]
    for e,expid in enumerate(labels):
        #tmp = dfin[expid].mean(level=['instrument','satellite'])
        tmp = dfin[expid].groupby(level=['instrument','satellite']).mean()
        tmp[['read','keep','assim']] = tmp[['read','keep','assim']].astype(int)
        for stat in ['read','keep','assim']:
            tmp2 = tmp[stat]
            tmp2.name = labels[e]
            exec('%s.append(tmp2)'%stat)

    read = pd.concat(read,axis=1)
    keep = pd.concat(keep, axis=1)
    assim= pd.concat(assim,axis=1)

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    fig1,ax1 = plt.subplots(figsize=(10,8))
    read.plot(ax=ax1,kind='barh',logx=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    titlestr = 'Read : # of %s observations\n%s' % (otype,title_substr)
    ax1.set_title(titlestr,fontsize='x-large')
    yticklabels_new = get_yticklabels_new(ax1)
    ax1.set_yticklabels(yticklabels_new,fontsize=8)

    fig2,ax2 = plt.subplots(figsize=(10,8))
    assim.plot(ax=ax2,kind='barh',logx=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    titlestr = 'Assimilated: # of %s observations\n%s' % (otype,title_substr)
    ax2.set_title(titlestr,fontsize='x-large')
    yticklabels_new = get_yticklabels_new(ax2)
    ax2.set_yticklabels(yticklabels_new,fontsize=8)

    return [fig1,fig2]

def plot_sat_diff(dfin,otype=''):

    # Collect all experiments into a single DataFrame
    read,keep,assim = [],[],[]
    for e,expid in enumerate(labels):
        tmp = dfin[expid].groupby(level=['instrument','satellite']).mean()
        tmp[['read','keep','assim']] = tmp[['read','keep','assim']].astype(int)
        for stat in ['read','keep','assim']:
            tmp2 = tmp[stat]
            tmp2.name = labels[e]
            exec('%s.append(tmp2)'%stat)

    read = pd.concat(read,axis=1)
    keep = pd.concat(keep, axis=1)
    assim= pd.concat(assim,axis=1)
    #print ('oz assim')
    #print (assim)
    if assim.iloc[:,0].sum() > 0:
        tmp = assim.div(assim.iloc[:,0],axis='index') * 100.0 - 100.0
        nobsdiff = tmp.drop(tmp.columns[0],axis=1)
        nclm = len(nobsdiff.columns)
    
        lc = mc[0] if nclm == 1 else mc[:nclm]
    
        fig1,ax1 = plt.subplots(figsize=(10,8))
        nobsdiff.plot(ax=ax1,kind='barh',color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        #nobsdiff.plot(ax=ax1,kind='barh',width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        titlestr = 'Normalized number of assimilated %s observations \n%s' % (otype,title_substr)
        ax1.set_title(titlestr,fontsize='x-large')
        yticklabels_new = get_yticklabels_new(ax1)
        ax1.set_yticklabels(yticklabels_new,fontsize=8)
    else:
        fig1=None

    return fig1

def plot_channel(dfin,inst=''):

    # Collect all experiments into a single DataFrame
    assim = []
    for e,expid in enumerate(labels):
        tmp = dfin[expid].mean(level=['satellite','channel'])
        tmp[['nassim']] = tmp[['nassim']].astype(int)
        tmp2 = tmp['nassim']
        tmp2.name = labels[e]
        assim.append(tmp2)

    assim = pd.concat(assim,axis=1)

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    fig,ax = plt.subplots(figsize=(10,8))
    #assim.plot(ax=ax,kind='barh',logx=True,width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    assim.plot(ax=ax,kind='barh',logx=True,width=0.9,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    titlestr = 'Assimilated: # of %s observations\n%s' % (inst.upper(),title_substr)
    ax.set_title(titlestr,fontsize='x-large')
    yticklabels_new = get_yticklabels_new(ax)
    ax.set_yticklabels(yticklabels_new,fontsize=8)

    return fig

def plot_channel_nobsdiff(dfin,inst='',statslvl=['satellite','channel'],normdiff=True,wndic=None):

    # Collect all experiments into a single DataFrame
    assim = []
    for e,expid in enumerate(labels):
        tmp = dfin[expid].groupby(level=statslvl).mean()
        tmp[['nassim']] = tmp[['nassim']].astype(int)
        tmp2 = tmp['nassim']
        tmp2.name = labels[e]
        assim.append(tmp2)

    assim = pd.concat(assim,axis=1).dropna()

    if normdiff:
        tmp = assim.div(assim.iloc[:,0],axis='index') * 100.0 - 100.0
        obscount = tmp.drop(tmp.columns[0],axis=1)
    else:
        obscount = assim
    nclm = len(obscount.columns)

    lc = mc[0] if nclm == 1 else mc[:nclm]

    fig,ax = plt.subplots(figsize=(10,12))
    #obscount.plot(ax=ax,kind='barh',width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    obscount.plot(ax=ax,kind='barh',width=0.9,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    titlestr = 'Assimilated: # of %s observations\n%s' % (inst.upper(),title_substr)
    ax.set_title(titlestr,fontsize='x-large')
    if len(statslvl) == 1:
        yindex=obscount.index.get_level_values('channel')
        if wndic is not None:
            y2index=[]
            for ch in yindex:
                #print (ch, int(wndic[ch]))
                y2index.append(f'{ch}, {int(wndic[ch])}')
            yindex = y2index
            ax.set_ylabel('Channel, Wavenumber $(cm^{-1})$',fontsize=10)
        else:
            ax.set_ylabel('Channel',fontsize=10)
        if len(yindex) > 25:
            a = np.arange(len(yindex))
            ax.yaxis.set_ticks(a[::2])
            ax.yaxis.set_ticklabels(yindex[::2], fontsize=8)
        else:
            ax.yaxis.set_ticklabels(yindex, fontsize=8)    
        #plt.vlines(0.0,yindex[0],yindex[-1],colors='k',linestyles='-',linewidth=2.0,label=None)
        plt.legend(loc=0,numpoints=1,frameon=False)
        #plt.ylim(1,yindex[-1]+1)
        #plt.yticks(yindex)
    else:
        yticklabels_new = get_yticklabels_new(ax)
        if len(yticklabels_new) > 25:
            a = np.arange(len(yticklabels_new))
            ax.yaxis.set_ticks(a[::5])
            ax.yaxis.set_ticklabels(yticklabels_new[::5])
        else:
            ax.set_yticklabels(yticklabels_new,fontsize=8)
    if normdiff:
        plt.xlabel('Mean Assimilated Obs Count (%, normalized)')    
    else:
        plt.xlabel('Mean Assimilated Obs Count')

    return fig

def plot_channel_radfit(dfin,dflen,dfina=None,inst='',normalize=False,obsnum=False,
                        statslvl=['satellite','channel'],wndic=None):

    if obsnum:
        fig, axes = plt.subplots(1,2,figsize=(8, 6))
    else:
        fig, ax = plt.subplots(figsize=(11,8))

    # Collect all experiments into a single DataFrame
    if obsnum:
        ax = axes[0]
        assim = []
        for e,expid in enumerate(labels):
            print ('expid ', expid)
            print (dfin[expid])
            tmp = dfin[expid].groupby(level=statslvl).mean()
            tmp[['nassim']] = tmp[['nassim']].astype(int)
            tmp2 = tmp['nassim']
            tmp2.name = labels[e]
            assim.append(tmp2)

        assim = pd.concat(assim,axis=1).dropna()
        #print 'assim', assim
        tmp = assim.div(assim.iloc[:,0],axis='index') * 100.0
        #print 'tmp', tmp
        nobsdiff = tmp.drop(tmp.columns[0],axis=1)
        #print nobsdiff
        nclm = len(nobsdiff.columns)
    
        lc = mc[0] if nclm == 1 else mc[:nclm]
    
        #nobsdiff.plot(ax=ax,kind='barh',width=0.5,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        #yindex=nobsdiff.index.get_level_values('channel')
        yindex=nobsdiff.index.values.tolist()
        for e,expid in enumerate(labels):
            if e > 0:
                profile  = nobsdiff[labels[e]].values
                #print profile
                a = np.arange(len(yindex))
                ax.plot(profile, a, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e],
                        linewidth=2.0, alpha=alpha)
        titlestr = 'Assimilated: # of %s observations\n%s' % (inst.upper(),title_substr)
        #ax.set_title(titlestr,fontsize='x-large')
        if len(statslvl) == 1:
            ax.vlines(100.0,a[0],a[-1],colors='k',linestyles='-',linewidth=0.5,label=None)
            if len(labels) > 1:
                ax.legend(loc=0,numpoints=1,fontsize=10,frameon=False)
            #ax.xaxis.set_major_locator(plt.MaxNLocator(7))
            ax.xaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            if wndic is not None:
                y2index=[]
                for ch in yindex:
                    y2index.append(f'{ch}, {int(wndic[ch])}')
                ax.set_ylabel('Channel, Wavenumber $(cm^{-1})$',fontsize=10)
            else:
                ax.set_ylabel('Channel',fontsize=10)
                y2index = yindex
            if 'cris' in inst or inst == 'iasi' or inst == 'airs':
                ax.yaxis.set_ticks(a[::5])
                ax.yaxis.set_ticklabels(y2index[::5])
            else:
                ax.yaxis.set_ticks(a)
                ax.yaxis.set_ticklabels(y2index)

            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_xlabel('Assimilated Obs Count (%, normalized)',fontsize=10)
            ax.text(0.15, 0.9, '(a)',
                    verticalalignment='bottom', horizontalalignment='center',
                    transform=ax.transAxes,
                    color='black', fontsize=12)
        else:
            yticklabels_new = get_yticklabels_new(ax)
            ax.set_yticklabels(yticklabels_new,fontsize=8)
 

    omfstd = []; CI_95 = []
    cntldf=dfin[labels[0]][['OmFbc_std']].astype(float)
    print ('cntldf')
    print (cntldf)
    #cntldf=cntldf.dropna()
    if dfina is not None:
        cntldfa=dfina[labels[0]][['OmFbc_std']].astype(float)
    #omfstdwci=cntldf.groupby(level=['satellite','channel']).mean()
    omfstdwci=cntldf.groupby(level=['channel']).mean()
    print ('omfstdwci')
    print (omfstdwci)
    for e,expid in enumerate(labels):
        if normalize and e > 0:
            expdf = dfin[expid][['OmFbc_std']].astype(float)
            #expdf = expdf.dropna()
            if dfina is not None:
                expdfa = dfina[expid][['OmFbc_std']].astype(float)
            normdf=expdf.div(cntldf,axis=1)
            print ('normdf')
            print (normdf)
            #normdf=normdf.dropna()
            """with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print expid
                print normdf"""
            normdf['OmFbc_std']=normdf['OmFbc_std']*100.0
            #profile=normdf.groupby(expdf.index).apply(mean_confidence_interval)
            #profile=normdf.groupby(level=['satellite','channel']).apply(mean_confidence_interval)
            profile=normdf.groupby(level=['channel']).apply(mean_confidence_interval)
            print ('profile')
            print (profile)
            stdmean=np.array([x[0] for x in profile])
            column_values = stdmean
            column_name = '%s_omfstd'%(labels[e])
            omfstdwci[column_name]=column_values
            tmp=np.array([x[1] for x in profile])
            tmp2=np.array([x[0] for x in tmp]) 
            column_values = tmp2
            column_name = '%s_ci95'%(labels[e])
            omfstdwci[column_name]=column_values
            #print (omfstdwci)
        else:
            tmp = dfin[expid].groupby(level=['satellite','channel']).mean()
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
            tmp2.name = labels[e]
            omfstd.append(tmp2)
            #print ('omfstd')
            #print (omfstd)
    
    if not normalize:
        omfstd = pd.concat(omfstd,axis=1)
        tmp = omfstd.div(omfstd.iloc[:,0],axis='index')*100.0 - 100.0
        stddiff = tmp.drop(tmp.columns[0],axis=1)
        nclm = len(stddiff.columns)
        lc = mc[1] if nclm == 1 else mc[:nclm]
    
    if obsnum:
        ax = axes[1]
    if normalize:
        tomfstdwci=omfstdwci
        yindex0=tomfstdwci.index.values.tolist()
        mchan=len(yindex0)
        for e,expid in enumerate(labels):
            print ('exp ', expid)
            if e == 0:
                ax.vlines(100.0,a[0],a[-1],colors='k',linestyles='-',linewidth=0.5,label=None)
            else:
                column_std='%s_omfstd'%(labels[e])
                column_ci95='%s_ci95'%(labels[e])
                elevs = np.array(yindex) + e*0.1
                tomfstdwci = tomfstdwci.dropna(subset=[column_std])
                yindex=tomfstdwci.index.values.tolist()
                if len(yindex) < mchan:
                    misch=list(set(yindex0) - set(yindex))
                    print (f'WARNING: missing {inst} channel {misch} for experiment {expid}')
                a = np.arange(len(yindex)) 
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                if dflen < 10 or len(a) < 3:
                    ax.plot(tomfstdwci[[column_std]].values, a, marker='o', label=labels[e], color=mc[e], mfc=mc[e], mec=mc[e],
                            linewidth=2.0, alpha=alpha)
                else:
                    ax.errorbar(tomfstdwci[[column_std]].values[:,0], a, xerr=tomfstdwci[[column_ci95]].values[:,0],color=mc[e],
                                label=expid)
        if len(labels) > 1:
            ax.legend(loc=0,numpoints=1,fontsize=10,frameon=False)
        ax.xaxis.set_tick_params(labelsize=8)
        if inst == 'airs' or inst == 'iasi' or 'cris' in inst: 
            ax.yaxis.set_ticks(a[::5])
            ax.yaxis.set_ticklabels(yindex[::5]) 
        else:
            ax.yaxis.set_ticks(a)
            ax.yaxis.set_ticklabels(yindex)

        ax.yaxis.set_tick_params(labelsize=8)
        ax.set_xlabel('FG std. dev. (%, normalized)',fontsize=10)
        amin = tomfstdwci[[column_std]].values.min()
        amin = amin - (100.0- amin) * 0.1
        amax = tomfstdwci[[column_std]].values.max()
        amax = amax + (amax - 100.0) * 0.1
        #if amin > 99.99 or amax < 100.001:
        #    ax.set_xlim(left=99.98, right=100.02)
        if not obsnum: 
            ax.set_ylabel('Channel',fontsize=12)
        else:
            ax.text(0.15, 0.9, '(b)',
                    verticalalignment='bottom', horizontalalignment='center',
                    transform=ax.transAxes,
                    color='black', fontsize=12)
    else:
        #stddiff.plot(ax=ax,kind='barh',width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        stddiff.plot(ax=ax,kind='barh',width=0.9,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        yticklabels_new = get_yticklabels_new(ax)
        ax.set_yticklabels(yticklabels_new,fontsize=8)

    if not obsnum:
        titlestr = 'Normalized OmF std. dev of %s observations\n%s' % (inst.upper(),title_substr)
        ax.set_title(titlestr,fontsize='x-large')
    else:
        titlestr = 'Normalized Obs count & OmF std. dev of %s observations\n%s' % (inst.upper(),title_substr)
        fig.suptitle(titlestr, fontsize=12)

    return fig

def plot_channel_omfbc(dfin,inst='',statslvl=['satellite','channel'],wndic=None):

    # Collect all experiments into a single DataFrame
    omfbc = []
    for e,expid in enumerate(labels):
        tmp = dfin[expid].groupby(level=statslvl).mean()
        tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
        tmp2 = tmp['OmF_bc']
        tmp2.name = labels[e]
        omfbc.append(tmp2)

    omfbc = pd.concat(omfbc,axis=1)
    #omfbc = omfbc.div(omfbc.iloc[:,0],axis='index') - 1.0

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    fig,ax = plt.subplots(figsize=(10,8))
    #omfbc.plot(ax=ax,kind='barh',width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    omfbc.plot(ax=ax,kind='barh',width=0.9,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    titlestr = 'OmF bias w BC of %s observations\n%s' % (inst.upper(),title_substr)
    ax.set_title(titlestr,fontsize='x-large')

    yindex=omfbc.index.values.tolist()
    a = np.arange(len(yindex))
    if wndic is not None:
        y2index=[]
        for ch in yindex:
            y2index.append(f'{ch}, {int(wndic[ch])}')
        yindex = y2index
        ax.set_ylabel('Channel, Wavenumber $(cm^{-1})$',fontsize=10)
    else:
        ax.set_ylabel('Channel',fontsize=10)

    if inst == 'airs' or inst == 'iasi' or 'cris' in inst:
        ax.yaxis.set_ticks(a[::5])
        ax.yaxis.set_ticklabels(yindex[::5])
    else:
        ax.yaxis.set_ticks(a)
        ax.yaxis.set_ticklabels(yindex)
    ax.yaxis.set_tick_params(labelsize=8)

    if len(statslvl) == 1:
        #yindex=omfbc.index.get_level_values('channel')
        #plt.vlines(0.0,yindex[0],yindex[-1],colors='k',linestyles='--',linewidth=2.0,label=None)
        plt.legend(loc=0,numpoints=1,frameon=False)
        #plt.ylim(0,yindex[-1]+1)
        #plt.yticks(yindex)
    else:
        yticklabels_new = get_yticklabels_new(ax)
        ax.set_yticklabels(yticklabels_new,fontsize=8)
    plt.xlabel('O-F Mean Bias (K)')

    return fig

def plot_channel_omfwobc(dfin,inst='',statslvl=['satellite','channel'],wndic=None):

    # Collect all experiments into a single DataFrame
    omfwobc = []
    for e,expid in enumerate(labels):
        tmp = dfin[expid].groupby(level=statslvl).mean()
        tmp[['OmF_wobc']] = tmp[['OmF_wobc']].astype(float)
        tmp2 = tmp['OmF_wobc']
        tmp2.name = labels[e]
        omfwobc.append(tmp2)

    omfwobc = pd.concat(omfwobc,axis=1)
    #omfwobc = omfwobc.div(omfwobc.iloc[:,0],axis='index') - 1.0
    #print ('OmF bias')

    lc = mc[0] if len(labels) == 1 else mc[:len(labels)]

    fig,ax = plt.subplots(figsize=(10,8))
    #omfwobc.plot(ax=ax,kind='barh',width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    omfwobc.plot(ax=ax,kind='barh',width=0.9,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print (omfwobc)
    yindex=omfwobc.index.values.tolist()
    #print (len(yindex), yindex)
    a = np.arange(len(yindex))
    if wndic is not None:
        y2index=[]
        for ch in yindex:
            y2index.append(f'{ch}, {int(wndic[ch])}')
        yindex = y2index
        ax.set_ylabel('Channel, Wavenumber $(cm^{-1})$',fontsize=10)
    else:
        ax.set_ylabel('Channel',fontsize=10)

    if inst == 'airs' or inst == 'iasi' or 'cris' in inst:
        ax.yaxis.set_ticks(a[::5])
        ax.yaxis.set_ticklabels(yindex[::5])
    else:
        ax.yaxis.set_ticks(a)
        ax.yaxis.set_ticklabels(yindex)
    ax.yaxis.set_tick_params(labelsize=8)

    titlestr = 'OmF bias w/o BC of %s observations\n%s' % (inst.upper(),title_substr)
    ax.set_title(titlestr,fontsize='x-large')
    if len(statslvl) == 1:
        #yindex=omfwobc.index.get_level_values('channel')
        #plt.vlines(0.0,yindex[0],yindex[-1],colors='k',linestyles='--',linewidth=2.0,label=None)
        plt.legend(loc=0,numpoints=1,frameon=False)
        #plt.ylim(0,yindex[-1]+1)
        #plt.yticks(yindex)
        #plt.ylabel('Channel',fontsize=12)
    else:
        yticklabels_new = get_yticklabels_new(ax)
        ax.set_yticklabels(yticklabels_new,fontsize=8)
    plt.xlabel('O-F Mean Bias w/o BC (K)')

    return fig

def plot_channel_omfbias(dfin,inst='',wndic=None):

    # Collect all experiments into a single DataFrame
    figs = []; fignames = []
    for e,expid in enumerate(labels):
        tmp = dfin[expid].groupby(level=['satellite','channel']).mean()
        tmp[['OmF_wobc','OmF_bc']] = tmp[['OmF_wobc','OmF_bc']].astype(float)
        omfbc = tmp[['OmF_wobc','OmF_bc']]
        omfbc.columns = ['bfbc','aftbc']
        #omfbc.rename(columns={'OmF_wobc': labels[e]+'bfbc', 'OmF_bc': labels[e]+'afbc'}, inplace=True)
        lc = mc[:2]
        fig,ax = plt.subplots(figsize=(10,8))
        #omfbc.plot(ax=ax,kind='barh',width=0.9,sort_columns=True,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        omfbc.plot(ax=ax,kind='barh',width=0.9,color=lc,alpha=alpha,fontsize=12,edgecolor='k',linewidth=0.0)
        yindex=omfbc.index.values.tolist()
        if wndic is not None:
            y2index=[]
            for ch in yindex:
                y2index.append((ch[0], f'{ch[1]}/{int(wndic[ch[1]])}'))
                #y2index.append((ch[0], (ch[1],int(wndic[ch[1]]))))
            yindex = y2index
            ax.set_ylabel('Instrument, Channel/Wavenumber $(cm^{-1})$',fontsize=10)
        else:
            ax.set_ylabel('Instrument, Channel',fontsize=10)

        if len(yindex) > 50:
            a = np.arange(len(yindex))
            ax.yaxis.set_ticks(a[::5])
            ax.yaxis.set_ticklabels(yindex[::5])
        elif len(yindex) > 20:
            a = np.arange(len(yindex))
            ax.yaxis.set_ticks(a[::2])
            ax.yaxis.set_ticklabels(yindex[::2])

        titlestr = '%s OmF bias of %s observations\n%s' % (labels[e],inst.upper(),title_substr)
        ax.set_title(titlestr,fontsize='x-large')
        yticklabels_new = get_yticklabels_new(ax)
        ax.set_yticklabels(yticklabels_new,fontsize=8)
        figs.append(fig)
        fignames.append(inst+'omfbias'+labels[e])

    return figs, fignames

def plot_channel_omf_FGvsANL(dfin,dfin2,dfin3,stats='std',inst='',wndic=None):

    # Collect all experiments into a single DataFrame

    fig = plt.figure(figsize=(12, 8))
    if len(labels) <= 3:
        gs = gspec.GridSpec(1, len(labels))
    else:
        gs = gspec.GridSpec(2, 3)

    amin=0.0; amax=0.0
    for e,expid in enumerate(labels):
        omfstats = []
        tmp = dfin[expid].groupby(level=['channel']).mean()
        if stats == 'std':
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
        elif stats == 'bias':
            tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
            tmp2 = tmp['OmF_bc']
        else:
            tmp[['nassim']] = tmp[['nassim']].astype(float)
            tmp2 = tmp['nassim']

        tmp2.name = 'oloop1'
        omfstats.append(tmp2)
        tmp = dfin2[expid].groupby(level=['channel']).mean()
        if stats == 'std':
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
        elif stats == 'bias':
            tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
            tmp2 = tmp['OmF_bc']
        else:
            tmp[['nassim']] = tmp[['nassim']].astype(float)
            tmp2 = tmp['nassim']

        tmp2.name = 'oloop2'
        omfstats.append(tmp2)
        tmp = dfin3[expid].groupby(level=['channel']).mean()
        if stats == 'std':
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
        elif stats == 'bias':
            tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
            tmp2 = tmp['OmF_bc']
        else:
            tmp[['nassim']] = tmp[['nassim']].astype(float)
            tmp2 = tmp['nassim']

        tmp2.name = 'oloop3'
        omfstats.append(tmp2)

        loopstat = pd.concat(omfstats,axis=1)
        nclm = len(loopstat.columns)

        lc = ['k','b','r']

        yindex=loopstat.index.values.tolist()
        if wndic is not None:
            y2index=[]
            for ch in yindex:
                y2index.append(f'{ch}, {int(wndic[ch])}')
            yindex = y2index
            yaxislabel = 'Channel, Wavenumber $(cm^{-1})$'
        else:
            yaxislabel = 'Channel'

        ax = plt.subplot(gs[e])

        for l,lpn in enumerate(['oloop1','oloop2','oloop3']):
            profile  = loopstat[lpn].values
            amax = max(profile.max(), amax)
            amin = min(profile.min(), amin)
            a = np.arange(len(yindex))
            ax.plot(profile, a, marker='o', label=lpn, color=lc[l], mfc=lc[l], mec=lc[l],
                    linewidth=2.0, alpha=alpha)

        if inst == 'airs' or inst == 'iasi' or inst == 'ssmis' or 'cris' in inst:
            ax.yaxis.set_ticks(a[::5])
            ax.yaxis.set_ticklabels(yindex[::5])
        else:
            ax.yaxis.set_ticks(a)
            ax.yaxis.set_ticklabels(yindex)
        ax.yaxis.set_tick_params(labelsize=8)
        if e == 0:
            ax.legend(loc=0,numpoints=1,fontsize=10,frameon=False)
        if e == 0 or e == 3:
            ax.set_ylabel(yaxislabel,fontsize=10)
        if stats == 'std':
            ax.set_xlabel('standard deviation')
        elif stats == 'bias':
            ax.vlines(0.0,a[0],a[-1],colors='k',linestyles='-',linewidth=1.0,label=None)
            dx = (amax-amin)*1.1
            plt.xlim(amin-dx,amax+dx)
            ax.set_xlabel('bias')
        else:
            ax.set_xlabel('assimilated obs count')
        ax.set_title(labels[e])

    for e,expid in enumerate(labels):
        ax = plt.subplot(gs[e])
        plt.xlim(amin,amax)

    if stats == 'std' or stats == 'bias':
        titlestr = 'OmF %s of %s observations\n%s' % (stats,inst.upper(),title_substr)
    else:
        titlestr = 'Assimilated %s observations\n%s' % (inst.upper(),title_substr)
         

    fig.suptitle(titlestr, fontsize=12)

    return fig


def plot_channel_omf_FGANL(dfin,dfin2,dfin3,stats='std',inst=''):

    # Collect all experiments into a single DataFrame
    figs = []; fignames = []
    
    for e,expid in enumerate(labels):
        omfstats = []
        tmp = dfin[expid].groupby(level=['satellite','channel']).mean()
        if stats == 'std':
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
        else:
            tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
            tmp2 = tmp['OmF_bc']
        tmp2.name = 'oloop1'
        omfstats.append(tmp2)
        tmp = dfin2[expid].groupby(level=['satellite','channel']).mean()
        if stats == 'std':
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
        else:
            tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
            tmp2 = tmp['OmF_bc']
        tmp2.name = 'oloop2'
        omfstats.append(tmp2)
        tmp = dfin3[expid].groupby(level=['satellite','channel']).mean()
        if stats == 'std':
            tmp[['OmFbc_std']] = tmp[['OmFbc_std']].astype(float)
            tmp2 = tmp['OmFbc_std']
        else:
            tmp[['OmF_bc']] = tmp[['OmF_bc']].astype(float)
            tmp2 = tmp['OmF_bc']
        tmp2.name = 'oloop3'
        omfstats.append(tmp2)
    
        loopstats = pd.concat(omfstats,axis=1)

        fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(top=0.875, hspace=0.3)
        titlestr = '%s OmF %s of %s observations\n%s' % (labels[e],stats,inst.upper(),title_substr) 
        lc = ['k','b','r']
     
        satlist = loopstats.index.get_level_values('satellite').unique()
        if len(satlist) <= 3:
            gs = gspec.GridSpec(len(satlist), 1)
        elif len(satlist) == 4:
            gs = gspec.GridSpec(2, 2)
        else:
            gs = gspec.GridSpec(2, 3)

        for s, sat in enumerate(satlist):
            ax = plt.subplot(gs[s])
            #loopstats.xs(sat).plot(ax=ax,kind='line',sort_columns=True,color=lc,alpha=alpha,fontsize=12,linewidth=2.0)
            if s == 0:
                loopstats.xs(sat).plot(ax=ax,kind='bar',alpha=alpha,linewidth=0.0,legend=True,fontsize=6)
            else:
                loopstats.xs(sat).plot(ax=ax,kind='bar',alpha=alpha,linewidth=0.0,legend=False)
            #plt.legend(fontsize='small')
            ax.set_title(sat)
            if stats == 'std':
                ax.set_ylabel('standard deviation')
            else:
                ax.set_ylabel('bias')

        plt.tight_layout()

        figs.append(fig)
        fignames.append(inst+'omf'+stats+'_'+labels[e])

    return figs, fignames

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

if __name__ == '__main__':

    global expids,labels,save_figure
    global title_substr
    global mc,alpha

    parser = ArgumentParser(description = 'Process gsistat.gdas.YYYYMMDDHH file',formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-x','--expid',help='experiment ID',type=str,nargs='+',required=True)
    parser.add_argument('-b','--begin_date',help='beginning date',type=str,metavar='YYYYMMDDHH',required=True)
    parser.add_argument('-d','--dump',help='forecast type gfs or gdas',type=str,nargs='+',required=False,default='gdas')
    parser.add_argument('-e','--end_date',help='ending date',type=str,metavar='YYYYMMDDHH',default=None,required=False)
    parser.add_argument('-a','--archive_dir',help='archive directory',type=str,nargs='+',required=False,default=['/da/noscrub/%s/archive'%os.environ['USER']])
    parser.add_argument('-l','--label',help='list of labels for experiment IDs',nargs='+',required=False)
    parser.add_argument('-m','--mode',help='gsi or enkf',type=str,nargs='+',required=False,default='gsi',)
    parser.add_argument('-s','--save_figure',help='save figures as png and pdf',action='store_true',required=False)
    parser.add_argument('-i','--instruments',help='list of instruments to show',nargs='+',required=False, default=None)
    parser.add_argument('-u','--plot_used',help='plot used radiance channels',action='store_true',required=False)
    parser.add_argument('-c','--plot_conv',help='plot stats for conventional data',action='store_true',required=False)
    parser.add_argument('-cnvall','--plot_cnvall',help='plot all conventional uv, t, q',action='store_true',required=False)
    parser.add_argument('-cg','--plot_costg',help='plot cost function',action='store_true',required=False)
    parser.add_argument('-cgJo','--plot_costgJo',help='plot cost function and Jo',action='store_true',required=False)
    parser.add_argument('-oz','--plot_oz',help='plot satellite data',action='store_true',required=False)
    parser.add_argument('-rad','--plot_rad',help='plot satellite data',action='store_true',required=False)
    parser.add_argument('-anl','--plot_analysis',help='plot stats for analysis',action='store_true',required=False)
    parser.add_argument('-fganl','--plot_fganl',help='plot stats for FGvsANL',action='store_true',required=False)
    parser.add_argument('-scyc','--singe_cycle',help='single cycle from run directory',action='store_true',required=False)
    parser.add_argument('-randomcyc','--random_cycle',help='random cycles',action='store_true',required=False)
    parser.add_argument('-cycfreq','--cycle_freq',help='cycle frequency',type=str,required=False,default='6H')
    parser.add_argument('-pltvar','--pltvar',help='variables to plot',nargs='+',required=False, default=['uv','t','q','gps','amv','scrm'])
    parser.add_argument('-ylog','--ylog',help='log for y axis',action='store_true',required=False)
    parser.add_argument('-lclr','--linecolors',help='line colors',nargs='+',required=False, default=['k', 'b', 'r', 'g', 'm','c','y'])
    parser.add_argument('-style','--panelstyle',help='panel style',type=str,required=False,default='nappend')
    parser.add_argument('-nlegend','--no_legend',help='plot legend',action='store_true',required=False)

    args = parser.parse_args()

    expids = args.expid
    bdate = datetime.strptime(args.begin_date,'%Y%m%d%H')
    edate = bdate if args.end_date is None else datetime.strptime(args.end_date,'%Y%m%d%H')
    archdirs = args.archive_dir
    if not type(archdirs) is list:
        archdirs = [archdirs]
    save_figure = args.save_figure
    plot_conv = args.plot_conv
    plot_cnvall = args.plot_cnvall
    plot_anl = args.plot_analysis
    plot_fganl = args.plot_fganl
    labels = expids if args.label is None else expids if len(args.label) != len(expids) else args.label
    modes = args.mode
    if not type(modes) is list:
        modes= [ modes ]
    instruments = args.instruments
    plot_used = args.plot_used
    single_cycle = args.singe_cycle
    random_cycle = args.random_cycle
    cycle_freq = args.cycle_freq
    cdumps = args.dump
    if not type(cdumps) is list:
        cdumps = [cdumps]
    plot_costg = args.plot_costg
    plot_costgJo = args.plot_costgJo
    plot_oz = args.plot_oz
    plot_rad = args.plot_rad
    pltvar = args.pltvar
    style = args.panelstyle
    nlegend = args.no_legend
    ylog = args.ylog
    mc = args.linecolors

    if ( edate < bdate ):
        print ('start date cannot be after end date, switching!')
        bdate,edate = edate,bdate

    if len(expids) > 1 and len(archdirs) == 1:
        archdirs = archdirs * len(expids)

    if len(expids) > 1 and len(cdumps) == 1:
        cdumps = cdumps * len(expids)

    if len(expids) > 1 and len(modes) == 1:
        modes = modes * len(expids)

    # Collect all the objects for all expids and all dates
    gsistat, ps, tcp, sst = {}, {}, {}, {}
    uv, t, q, gps, amv, scrm = {}, {}, {}, {}, {}, {}
    minim, oz, rad = {}, {}, {}

    ps_a, tcp_a, uv_a, t_a, q_a, sst_a, gps_a, amv_a, scrm_a =  {}, {}, {}, {}, {}, {}, {}, {}, {}

    pstyp = [120,180,181,187]
    #uvtyp = [220,221,222,223,224,227,228,229,230,231,232,233,234,235,
    #         270,271,280,281,282,283,284,285,286,287,288,291,292,293,294,295]
    #uvtyp = [220,221,230,231,232,233,280,282]
    if plot_cnvall:
        uvtyp = [220,221,223,229,230,231,232,233,234,280,282,289,290]
    else:
        #uvtyp = [230,231,232,233,234,235,236]
        uvtyp = [220]
        #uvtyp = [221,229]
        #uvtyp = [220,221,229,223,229,280,282,289,290]
        #ttyp = [120,180,182]
        ttyp = [120]
        #ttyp = [130,131,132,133,134,135,136]
        #qtyp = [120,131,133,134,180,182]
        #qtyp = [120,180,182]
        qtyp = [120]
    #qtyp = [130,131,132,133,134,135,136]
    #gpstyp = [004,722,745,042,043,003]
    #amvtyp = [240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260]
    amvtyp = [242, 243, 244, 245, 246, 247, 250, 252, 253, 254, 257, 258, 259, 260]
    scrmtyp = [280, 282, 289, 290]

    for expid,label,mode,cdump,archdir in zip(expids,labels,modes,cdumps,archdirs):

        print ('reading in data for experiment ... %s' % expid)
        gsistat[label] = []
        nfile=0
        if mode == 'gsi':
            suffix=""
        else:
            suffix=".ensmean"
        if single_cycle:
            fname = os.path.join(archdir,expid,'%s.t%sz.gsistat%s'%(cdump,bdate.strftime('%Y%m%d%H')[-2:],suffix))
            print (fname)
            if not os.path.exists(fname):
                print ('\033[1;31m' + '%s does not exist' % fname + '\033[1;m')
                continue
            gsistat[label].append(lgsi.GSIstat(fname,bdate.to_datetime()))
            nfile =+ 1
        elif random_cycle:
            print (archdir, expid, label)
            for fname in glob(os.path.join(archdir,expid,'gsistat.%s.*'%(cdump))):
                print (fname)
                adate=os.path.basename(fname)[-10:]
                gsistat[label].append(lgsi.GSIstat(fname,pd.to_datetime(adate,format='%Y%m%d%H')))
                nfile =+ 1
        else:
            for adate in pd.date_range(bdate,edate,freq=cycle_freq):
                fname = os.path.join(archdir,expid,'gsistat.%s.%s%s'%(cdump,adate.strftime('%Y%m%d%H'),suffix))
                print (fname)
                if not os.path.exists(fname):
                    print ('\033[1;31m' + '%s does not exist' % fname + '\033[1;m')
                    continue
                gsistat[label].append(lgsi.GSIstat(fname,pd.to_datetime(adate)))
                nfile =+ 1
    
        if plot_conv:
            #ps[label] = get_data1(gsistat[label],'ps',it=1,use='asm',typ=pstyp)
            ps[label] = get_data1(gsistat[label],'ps',it=1,use='asm',typ='all')
            """ only turn on sst when NSST is turned on """
            #sst[label] = get_data1(gsistat[label],'sst',it=1,use='asm',typ='all')
            tcp[label] = get_data1(gsistat[label],'tcp',it=1,use='asm',typ='all')
            uv[label] = get_data1(gsistat[label],'uv',it=1,use='asm',typ=uvtyp)
            if plot_cnvall:
                t[label] = get_data1(gsistat[label],'t',it=1,use='asm',typ='all')
                q[label] = get_data1(gsistat[label],'q',it=1,use='asm',typ='all')
            else:
                t[label] = get_data1(gsistat[label],'t',it=1,use='asm',typ=ttyp)
                q[label] = get_data1(gsistat[label],'q',it=1,use='asm',typ=qtyp)
            gps[label] = get_data1(gsistat[label],'gps',it=1,use='asm',typ='all')
            amv[label] = get_data1(gsistat[label],'uv',it=1,use='asm',typ=amvtyp)
            scrm[label] = get_data1(gsistat[label],'uv',it=1,use='asm',typ=scrmtyp)

            if plot_anl or plot_fganl:
                ps_a[label] = get_data1(gsistat[label],'ps',it=3,use='asm',typ='all')
                tcp_a[label] = get_data1(gsistat[label],'tcp',it=3,use='asm',typ='all')
                uv_a[label] = get_data1(gsistat[label],'uv',it=3,use='asm',typ=uvtyp)
                t_a[label] = get_data1(gsistat[label],'t',it=3,use='asm',typ='all')
                q_a[label] = get_data1(gsistat[label],'q',it=3,use='asm',typ='all')
                #t_a[label] = get_data1(gsistat[label],'t',it=3,use='asm',typ=ttyp)
                #q_a[label] = get_data1(gsistat[label],'q',it=3,use='asm',typ=qtyp)
                gps_a[label] = get_data1(gsistat[label],'gps',it=3,use='asm',typ='all')
                amv_a[label] = get_data1(gsistat[label],'uv',it=3,use='asm',typ=amvtyp)
                scrm_a[label] = get_data1(gsistat[label],'uv',it=3,use='asm',typ=scrmtyp)

        if plot_costg or plot_costgJo:
            minim[label] = get_data2(gsistat[label],'cost')

        if plot_oz:
            oz[label] = get_data2(gsistat[label],'oz',select=1,level='it')
        if plot_rad:
            rad[label] = get_data2(gsistat[label],'rad',select=1,level='it')


    # If instruments are desired, get them too
    if instruments is not None:
        insts, insts2, insts3 = {}, {}, {}
        for inst in instruments:
            insts[inst], insts2[inst], insts3[inst]  = {}, {}, {}
            tmp, tmp2, tmp3 = {}, {}, {}
            for expid in labels:
                expid_gsistat = gsistat[expid]
                expid_inst, lendf = get_inst_data(expid_gsistat,inst,select=1,level='it',plotanl=plot_anl,usedonly=plot_used)
                if lendf > 0:
                    tmp[expid] = expid_inst
                    insts[inst] = tmp
                if plot_anl:
                   expid_inst, lendf2 = get_inst_data(expid_gsistat,inst,select=2,level='it',plotanl=True,usedonly=plot_used)
                   if lendf > 0 and lendf2 > 0:
                       tmp2[expid] = expid_inst
                       insts2[inst] = tmp2
                   expid_inst, lendf3 = get_inst_data(expid_gsistat,inst,select=3,level='it',plotanl=True,usedonly=plot_used)
                   if lendf > 0 and lendf3 > 0:
                       tmp3[expid] = expid_inst
                       insts3[inst] = tmp3

    # Start plotting

    #mc = ['k', 'r', 'g', 'm','c','y']
    alpha = 0.8

    if random_cycle:
        title_substr = 'random cycles between %s and %s' % (bdate.strftime('%Y%m%d%H'),edate.strftime('%Y%m%d%H'))
    else:
        if bdate == edate:
            title_substr = '%s' % bdate.strftime('%Y%m%d%H')
        else:
            title_substr = '%s-%s' % (bdate.strftime('%Y%m%d%H'),edate.strftime('%Y%m%d%H'))

    figs = []; fignames = []

    if plot_conv:

        if plot_anl:
            fig = plot_ps(ps,ps2=ps_a,pname='ps') ; figs.append(fig) ; fignames.append('ps')
            #fig = plot_ps(tcp,ps2=tcp_a,pname='tcp'); figs.append(fig) ; fignames.append('tcp')
        else:
            fig = plot_ps(ps,pname='ps') ; figs.append(fig) ; fignames.append('ps')
            #fig = plot_ps(tcp,pname='tcp') ; figs.append(fig) ; fignames.append('tcp')
      
        fig = plot_profile(uv,t,q,gps,amv,scrm,stat='bias',pltvar=pltvar,ylog=ylog,nlegend=nlegend) 
        figs.append(fig) ; fignames.append('bias')
        if plot_anl:
            fig = plot_profile(uv_a,t_a,q_a,gps_a,amv_a,scrm_a,stat='bias',anl=True,pltvar=pltvar,ylog=ylog,
                               nlegend=nlegend) 
            figs.append(fig) ; fignames.append('bias_anl')
    
        if len(expids) > 1 and nfile >= 1:
            fig = plot_profile(uv,t,q,gps,amv,scrm,stat='count',normalize=True,pltvar=pltvar,ylog=ylog,
                               nlegend=nlegend)
            figs.append(fig) ; fignames.append('countn')	
            if style == 'single':
                for var in pltvar:
                    fig = plot_profile(uv,t,q,gps,amv,scrm,stat='rms',normalize=True,pltvar=[var],ylog=ylog,
                                       nlegend=nlegend)
                    figs.append(fig) ; fignames.append(f'rmsn_{var}')
            else:
                fig = plot_profile(uv,t,q,gps,amv,scrm,stat='rms',normalize=True,pltvar=pltvar,ylog=ylog,
                                   nlegend=nlegend)
                figs.append(fig) ; fignames.append('rmsn')
            if plot_anl:
                fig = plot_profile(uv_a,t_a,q_a,gps_a,amv_a,scrm_a,stat='rms',anl=True,normalize=True,
                                   pltvar=pltvar,ylog=ylog,nlegend=nlegend)
                figs.append(fig) ; fignames.append('rmsn_anl')
        else:
            fig = plot_profile(uv,t,q,gps,amv,scrm,stat='rms',pltvar=pltvar,ylog=ylog)
            figs.append(fig) ; fignames.append('rms')
            if plot_anl:
                fig = plot_profile(uv_a,t_a,q_a,gps_a,amv_a,scrm_a,stat='rms',anl=True,pltvar=pltvar,
                                   ylog=ylog,nlegend=nlegend) 
                figs.append(fig) ; fignames.append('rms_anl')


    # plot cost function and gradient together
    if plot_costg or plot_costgJo: 
        fig = plot_cost_gradient(minim) ; figs.append(fig) ; fignames.append('cost_gradient')
        fig = plot_cost(minim) ; figs.append(fig) ; fignames.append('cost')
        if plot_costgJo:
            fig = plot_Jo(minim); figs.append(fig) ; fignames.append('Jo')
        fig = plot_gradient(minim) ; figs.append(fig) ; fignames.append('gradient')

    if plot_oz:
        figr, figa = plot_sat(oz,otype='ozone') ; figs += [figr, figa] ; fignames += ['oz_read','oz_assim']
        if len(expids) > 1:
            fig = plot_sat_diff(oz,otype='ozone') ; figs.append(fig); fignames.append('oz_assim_diff')

    if plot_conv and plot_fganl:
        for e, expid in enumerate(labels):
            fig = plot_profile_FGvsANL(uv, t, q, gps, amv, scrm, uv_a, t_a, q_a, gps_a, amv_a, scrm_a,
                                       expidcntl=labels[0],expid=expid,elabelc=labels[0],elabel=labels[e],
                                       stat='rms',normalize=False,pltvar=pltvar,ylog=ylog)
            figs.append(fig)
            fignames.append('rms_FGvANL_%s'%(expid))
            if e > 0:
                fig = plot_profile_FGvsANL(uv, t, q, gps, amv, scrm, uv_a, t_a, q_a, gps_a, amv_a, scrm_a,
                                           expidcntl=labels[0],expid=expid,elabelc=labels[0],elabel=labels[e],
                                           stat='rms',normalize=True,pltvar=pltvar,ylog=ylog)
                figs.append(fig)
                fignames.append('rms_FGvANL_norm_%s'%(expid))
    
            fig = plot_profile_FGvsANL(uv, t, q, gps, amv, scrm, uv_a, t_a, q_a, gps_a, amv_a, scrm_a,
                                       expidcntl=labels[0],expid=expid,elabelc=labels[0],elabel=labels[e],
                                       stat='bias',pltvar=pltvar,ylog=ylog)
            figs.append(fig)
            fignames.append('bias_FGvANL_%s'%(expid))
        
    if instruments is None and plot_rad:
        #figr, figa = plot_sat(rad,otype='radiance') ; figs += [figr, figa]; fignames += ['rad_read','rad_assim']
        if len(expids) > 1:
            fig = plot_sat_diff(rad,otype='radiance')
            if fig is not None:
                figs.append(fig); fignames.append('rad_assim_diff')

    if instruments is not None:
        csvdir='/scratch2/GFDL/gfdlscr/Mingjing.Tong/gsidiag/ush/gsistat/data'
        for inst in instruments:
            if 'cris' in inst or inst == 'iasi' or inst == 'airs':
                instid = inst
                if 'cris' in inst:
                    instid = 'cris'
                wndic=pd.read_csv(f'{csvdir}/wavenumber_{instid}.csv',header=None,index_col=0).squeeze("columns").to_dict()
            else:
                wndic=None
            if len(insts[inst]) != 0:
                if len(expids) > 1:
                    # plot difference of assimilated observation
                    #fig = plot_channel_nobsdiff(insts[inst],inst=inst,statslvl=['channel'],wndic=wndic)
                    #figs.append(fig); fignames.append(f'obscount_norm_{inst}')
                    fig = plot_channel_nobsdiff(insts[inst],inst=inst,statslvl=['channel'],normdiff=False,wndic=wndic)
                    figs.append(fig); fignames.append(f'obscount_{inst}')
                    # plot OmF_std difference grouped by satellite and channel in bars
                    #fig = plot_channel_radfit(insts[inst],lendf,inst=inst,wndic=wndic)
                    #figs.append(fig); fignames.append(inst+'fitbar')
                    fig = plot_channel_radfit(insts[inst],lendf,inst=inst,normalize=True,obsnum=True,
                                              statslvl=['channel'],wndic=wndic)
                    figs.append(fig); fignames.append(inst+'fit')
                    if plot_anl:
                        fig = plot_channel_radfit(insts3[inst],lendf,inst=inst,normalize=True,obsnum=True,
                                                  statslvl=['channel'],wndic=wndic)
                        figs.append(fig); fignames.append(inst+'fit_anl')
                        fig = plot_channel_omf_FGvsANL(insts[inst],insts2[inst],insts3[inst],
                                                       stats='std',inst=inst,wndic=wndic)
                        figs.append(fig); fignames.append(inst+'std_fganl')
                        fig = plot_channel_omf_FGvsANL(insts[inst],insts2[inst],insts3[inst],
                                                       stats='bias',inst=inst,wndic=wndic)
                        figs.append(fig); fignames.append(inst+'bias_fganl')
                        fig = plot_channel_omf_FGvsANL(insts[inst],insts2[inst],insts3[inst],
                                                       stats='count',inst=inst,wndic=wndic)
                        figs.append(fig); fignames.append(inst+'count_fganl')
                    
                fig = plot_channel_omfbc(insts[inst],inst=inst,statslvl=['channel'],wndic=wndic)
                figs.append(fig) ; fignames.append(inst+'omf')
                fig = plot_channel_omfwobc(insts[inst],inst=inst,statslvl=['channel'],wndic=wndic) 
                figs.append(fig) ; fignames.append(inst+'omfwobc')

                bfigs,bfignames = plot_channel_omfbias(insts[inst],inst=inst,wndic=wndic)
                if save_figure:
                    for fig,figname in zip(bfigs,bfignames):
                        figname = './gsistat_%s' % figname
                        savefigure(fig,figname,format='png')


                """bfigs,bfignames = plot_channel_omf_FGANL(insts[inst],insts2[inst],insts3[inst],stats='std',inst=inst)
                if save_figure:
                    for fig,figname in zip(bfigs,bfignames):
                        figname = './gsistat_%s' % figname
                        savefigure(fig,figname,format='png')
    
                bfigs,bfignames = plot_channel_omf_FGANL(insts[inst],insts2[inst],insts3[inst],stats='bias',inst=inst)
                if save_figure:
                    for fig,figname in zip(bfigs,bfignames):
                        figname = './gsistat_%s' % figname
                        savefigure(fig,figname,format='png')"""

    if save_figure:
        for fig,figname in zip(figs,fignames):
            figname = './gsistat_%s' % figname
            savefigure(fig,figname,format='png')
    else:
        plt.show()

    sys.exit(0)
