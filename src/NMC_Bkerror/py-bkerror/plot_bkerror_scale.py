#!/usr/bin/env python

import sys, os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from scipy import stats
import numpy as np
from bkerror import bkerror
from splat import splat
import re

sys.path.append('/scratch2/GFDL/gfdlscr/Mingjing.Tong/ModelDiag/lib')
import model_levels as mllib
import general_lib as glib

try:
    import lib_mapping as lmapping
    plot_map = True
except:
    print ('lib_mapping module is not in your path')
    print ('No maps will be produced')
    plot_map = False


class GSIbkgerr(object):
    '''
    Object containing GSI static background error information
    '''
    def __init__(self,filename):
        '''
        Read and store GSI background error file.
        '''
        nsig,nlat,nlon = bkerror.get_header(filename)
        ivar,agvin,bgvin,wgvin,corzin,hscalesin,vscalesin,corq2in,corsstin,hsstin,corpin,hscalespin = bkerror.get_bkerror(filename,nsig,nlat,nlon)
        var = (ivar.tostring()).replace('\x00','')[:-1].split('|')

        self.filename = filename

        self.nsig = nsig
        self.nlat = nlat
        self.nlon = nlon

        self.ivar = ivar
        self.var = var

        self.agvin = agvin
        self.bgvin = bgvin
        self.wgvin = wgvin
        self.corzin = corzin
        self.hscalesin = hscalesin
        self.vscalesin = vscalesin
        self.corq2in = corq2in
        self.corsstin = corsstin
        self.hsstin = hsstin
        self.corpin = corpin
        self.hscalespin = hscalespin

        return


    def print_summary(self):
        '''
        Print a summary of the GSI background error file
        '''

        print
        print 'file = %s' % self.filename
        print 'nsig = %d, nlat = %d, nlon = %d, nvar = %d' % (self.nsig,self.nlat,self.nlon,len(self.var))
        print 'variables = %s' % ', '.join(self.var)
        print 'agv.shape: ', self.agvin.shape
        print 'bgv.shape: ', self.bgvin.shape
        print 'wgv.shape: ', self.wgvin.shape
        print 'corz.shape: ', self.corzin.shape
        print 'hscales.shape: ', self.hscalesin.shape
        print 'vscales.shape: ', self.vscalesin.shape
        print 'corq2.shape: ', self.corq2in.shape
        print 'corsst.shape: ', self.corsstin.shape
        print 'hsst.shape: ', self.hsstin.shape
        print 'corp.shape: ', self.corpin.shape
        print 'hscalesp.shape: ', self.hscalespin.shape
        print

        return

def plotZM(fig,ax1,data, x, y, plotOpt=None, modelLevels=None, surfacePressure=None):
    """Create a zonal mean contour plot of one variable
    plotOpt is a dictionary with plotting options:
      'scale_factor': multiply values with this factor before plotting
      'units': a units label for the colorbar
      'levels': use list of values as contour intervals
      'title': a title for the plot
    modelLevels: a list of pressure values indicating the model vertical resolution. If present,
        a small side panel will be drawn with lines for each model level
    surfacePressure: a list (dimension len(x)) of surface pressure values. If present, these will
        be used to mask out regions below the surface
    """
    # explanation of axes:
    #   ax1: primary coordinate system latitude vs. pressure (left ticks on y axis)
    #   ax2: twinned axes for altitude coordinates on right y axis
    #   axm: small side panel with shared y axis from ax2 for display of model levels
    # right y ticks and y label will be drawn on axr if modelLevels are given, else on ax2
    #   axr: pointer to "right axis", either ax2 or axm

    if plotOpt is None: plotOpt = {}
    labelFontSize = "small"
    # scale data if requested
    scale_factor = plotOpt.get('scale_factor', 1.0)
    pdata = data * scale_factor
    # determine contour levels to be used; default: linear spacing, 20 levels
    if data.min() < 0.0:
        abmax=max(abs(data.min()),abs(data.max()))
        clevs = plotOpt.get('levels', np.linspace(-abmax, abmax, 21))
    else:
        clevs = plotOpt.get('levels', np.linspace(data.min(), data.max(), 20))
    # map contour values to colors
    norm=colors.BoundaryNorm(clevs, ncolors=256, clip=False)
    # set minimum value
    vmin = plotOpt.get('vmin', None)
    #print 'vmin', vmin
    # draw the (filled) contours
    #contour = ax1.contourf(x, y, pdata, vmin=vmin, levels=clevs, norm=norm) 
    cmap = plotOpt.get('cmap','Spectral_r')
    #if clevs is None: 
    #    #contour0 = ax1.contourf(x, y, pdata, 21, vmin=vmin, cmap=cmap, extend='both')
    contour = ax1.contourf(x, y, pdata, vmin=vmin, levels=clevs, cmap=cmap, extend='both') 
    # mask out surface pressure if given
    if not surfacePressure is None: 
        ax1.fill_between(x, surfacePressure, surfacePressure.max(), color="white")    
    # add a title
    title = plotOpt.get('title', 'Vertical cross section')
    ax1.invert_yaxis()
    ax1.set_title(title,fontsize=10)
    #print title, contour.levels
    # add colorbar
    # Note: use of the ticks keyword forces colorbar to draw all labels
    #fmt = ticker.FormatStrFormatter("%g")
    #cbar = fig.colorbar(contour, ax=ax1, orientation='vertical', shrink=0.8,
    #                    ticks=clevs, format=fmt)
    #cbar.set_label(plotOpt.get('units', ''))
    #for t in cbar.ax.get_xticklabels():
    #    t.set_fontsize(labelFontSize)
    fig.colorbar(contour, ax=ax1, pad=0.1)
    # set up y axes: log pressure labels on the left y axis, altitude labels
    # according to model levels on the right y axis
    ax1.set_ylabel("Pressure [hPa]",fontsize=10)
    ylog = plotOpt.get('ylog', True)
    if ylog:
        ax1.set_yscale('log')
        #ax1.set_ylim(10.*np.ceil(y.max()/10.), y.min()) # avoid truncation of 1000 hPa
        subs = [1,2,5]
        if y.max()/y.min() < 30.:
            subs = [1,2,3,4,5,6,7,8,9]
        y1loc = ticker.LogLocator(base=10., subs=subs)
        ax1.yaxis.set_major_locator(y1loc)
        fmt = ticker.FormatStrFormatter("%g")
        ax1.yaxis.set_major_formatter(fmt)
        for t in ax1.get_yticklabels():
            t.set_fontsize(labelFontSize)
    ylim=plotOpt.get('ylim', 0.4)
    ax1.set_ylim(10.*np.ceil(y.max()/10.), ylim)
    # calculate altitudes from pressure values (use fixed scale height)
    z0 = 8.400    # scale height for pressure_to_altitude conversion [km]
    altitude = z0 * np.log(1015.23/y)
    # change values and font size of x labels
    x_label = plotOpt.get('x_label', True)
    if x_label:
        ax1.set_xlabel('Latitude [degrees]',fontsize=10)
    xloc = ticker.FixedLocator(np.arange(-90.,91.,30.))
    ax1.xaxis.set_major_locator(xloc)
    for t in ax1.get_xticklabels():
        t.set_fontsize(labelFontSize)
    # draw horizontal lines to the right to indicate model levels
    # add second y axis for altitude scale
    """ax2 = ax1.twinx()
    if not modelLevels is None:
        pos = ax1.get_position()
        axm = fig.add_axes([pos.x1,pos.y0,0.02,pos.height], sharey=ax2)
        axm.set_xlim(0., 1.)
        axm.xaxis.set_visible(False)
        modelLev = axm.hlines(altitude, 0., 1., color='0.5')
        axr = axm     # specify y axis for right tick marks and labels
        #turn off tick labels of ax2
        for t in ax2.get_yticklabels():
            t.set_visible(False)
        label_xcoor = 3.7
    else:
        axr = ax2
        label_xcoor = 1.05
    axr.set_ylabel("Altitude [km]",fontsize=10)
    axr.yaxis.set_label_coords(label_xcoor, 0.5)
    axr.set_ylim(altitude.min(), altitude.max())
    yrloc = ticker.MaxNLocator(steps=[1,2,5,10])
    axr.yaxis.set_major_locator(yrloc)
    axr.yaxis.tick_right()
    for t in axr.yaxis.get_majorticklines():
        t.set_visible(False)
    for t in axr.get_yticklabels():
        t.set_fontsize(labelFontSize) """

    return contour

def get_hybens_info(hybens_info):

    if os.path.isfile(hybens_info):
        file1 = open(hybens_info, 'r')
        Lines = file1.readlines()
        file1.close()

        hloc = []; vloc = []; betaS = []
        pattern = '\s+\S+\s+\S+\s+\S+\s+\S+\s+'
        for line in Lines:
            if re.match(pattern, line):
                tst = line.strip().split()
                hloc.append(tst[0])
                vloc.append(tst[1])
                betaS.append(tst[2])

        hl=np.array(hloc).astype(float)
        hl_rv=hl[::-1]
        vl=np.array(vloc).astype(float)
        vl_rv=vl[::-1]
        beta=np.array(betaS).astype(float)
        betaS_rv=beta[::-1]
        betaE_rv=1-betaS_rv

    else:
        raise Exception('hybens_info file not exists')

    return hl_rv, vl_rv, betaS_rv, betaE_rv

# bkerror file to read; e.g. global_berror.l64y258.f77
parser = ArgumentParser(description='read background error file and plot',formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--filenames',help='background error files to read and plot',type=str,nargs='+',required=True)
parser.add_argument('-f0','--filename0',help='background error file to compare',type=str,required=True)
parser.add_argument('-a','--archive',help='archive directory',type=str,required=False,default='../../../fix/Big_Endian')
parser.add_argument('-s','--scale',help='scale variance',action='store_true',required=False)
parser.add_argument('-p','--meanprofile',help='plot mean profile',action='store_true',required=False)
args = parser.parse_args()
fnames=[args.filename0]
fixdir=args.archive
apply_scale=args.scale
plot_meanprofile=args.meanprofile


if not type(args.filenames) is list:
    filenames = [args.filenames]
else: 
    filenames = args.filenames

fnames = fnames + filenames
print 'fnames ', fnames

figdir = './figures'
if not os.path.exists(figdir):
    os.makedirs(figdir)

clevels={}; hlevels={}; vlevels={}; glev={}; plevs={}
vstdmean={}; hlmean={}; vlmean={}
vmin=np.zeros(6)+99999.; vmax=np.zeros(6)
scale={}
if apply_scale:
    scale[64]=np.array([0.6,0.6,0.75,0.75,0.75,1.0,0.75])
    scale[127]=np.array([1.0,1.0,1.5,1.5,2.0,1.0,1.2])
    #scale[91]=np.array([0.6,0.6,0.7,0.7,0.9,1.0,0.7])
    scale[91]=np.array([0.6,0.6,0.8,0.8,0.9,1.0,0.7])
    #scale[91]=np.ones(7)
else:
    scale[64]=np.ones(7)
    scale[127]=np.ones(7)
    scale[91]=np.ones(7)


for iff, ff in enumerate(fnames):
    fn=os.path.join(fixdir,ff)
    tgsi = GSIbkgerr(fn)
    #tgsi.print_summary()
    case=os.path.basename(fn)

    idrt = 4
    slat,wlat = splat(idrt,tgsi.nlat)
    glat = 180. / np.arccos(-1.) * np.arcsin(slat[::-1])
    glon = np.linspace(0.,360.,tgsi.nlon,endpoint=False)
    glev[iff] = np.arange(1,tgsi.nsig+1)
    nlev=len(glev[iff])

    cmapdiv = 'Spectral_r'
    cmappos = 'Spectral_r'
    
    datadir='/scratch2/GFDL/gfdlscr/Mingjing.Tong/ModelDiag/data/akbk'
    akfile = os.path.join(datadir,'L%s.fv_core.res.nc'%(glev[iff].size))

    pe,ze,pm,zm,dp,dz = mllib.get_p_z(akfile)
    plevs[iff]=pm[::-1]
    aplevs=np.around(plevs[iff],decimals=2)

    bglevs = [ 1, 25, 45, 60 ]
    bgplevs = [ 1000, 850, 700, 500, 200, 100 ]
    tbglevs=[]
    for plev in bgplevs:
        tbglevs.append((np.abs(aplevs - plev)).argmin())

    zg,xg = np.meshgrid(aplevs,glat)

    #print 'aplevs', aplevs
    #print 'xg', xg.shape, zg.shape

    labelFontSize = "small"

    for l, lev in enumerate(tbglevs):
        print 'plotting agv at level = %d'  % lev
        fig=plt.figure()
        ax = plt.subplot(111)
        z = tgsi.agvin[:,:,lev]
        atitle='agv at level = %d (~%s hPa) max=%s, min=%s' % (lev,bgplevs[l],z.max(),z.min())
        if iff == 0 and l == 0:
            plotOpt = { 'vmin': -z.max(), 'title': atitle, 'extend': 'both' }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
            #cs=plotZM(fig,ax,z, xg, zg, y2=glev[iff], plotOpt=plotOpt)
            #cs=plt.contourf(xg,zg,z,21,vmin=-z.max(),cmap=cmapdiv,extend='both')
            alevels=cs.levels
        else:
            plotOpt = { 'levels': alevels, 'title': atitle, 'extend': 'both' }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
            #cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt)
            #cs=plt.contourf(xg,zg,z,alevels,cmap=cmapdiv,extend='both') 
 
        figname=os.path.join(figdir,'%s_agvl%02d.png'%(case,lev))
        plt.savefig(figname)

    for l, lev in enumerate(tbglevs):
        print 'plotting agv at level = %d'  % lev
        fig=plt.figure()
        ax = plt.subplot(111)
        z = tgsi.agvin[:,lev,:]
        atitle='agv at level = %d (~%s hPa) max=%s, min=%s' % (lev,bgplevs[l],z.max(),z.min())
        plotOpt = { 'levels': alevels, 'title': atitle, 'extend': 'both' }
        cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
        #cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt)
        #cs=plt.contourf(xg,zg,z,alevels,cmap=cmapdiv,extend='both')

        figname=os.path.join(figdir,'%s_agvMl%02d.png'%(case,lev))
        plt.savefig(figname)
    
    print 'plotting bgv and wgv'
    fig=plt.figure()
    ax=plt.subplot(2,1,1)
    atitle='bgv max=%s, min=%s'%(tgsi.bgvin.max(),tgsi.bgvin.min())
    if iff == 0:
        plotOpt = { 'vmin': -tgsi.bgvin.max(), 'title': atitle, 'extend': 'both', 'x_label': False }
        cs=plotZM(fig,ax,tgsi.bgvin, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
        blevels = cs.levels
    else:
        plotOpt = { 'levels': blevels, 'title': atitle, 'extend': 'both', 'x_label': False }
        cs=plotZM(fig,ax,tgsi.bgvin, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
    ax=plt.subplot(2,1,2)
    atitle='wgv max=%s, min=%s'%(tgsi.wgvin.max(),tgsi.wgvin.min())
    if iff == 0:
        plotOpt = { 'vmin': -tgsi.wgvin.max(), 'title': atitle, 'extend': 'both', 'ylim': 950.0, 'ylog': False }
        cs=plotZM(fig,ax,tgsi.wgvin, xg, zg, plotOpt=plotOpt)
        wlevels = cs.levels
    else:
        plotOpt = { 'levels': wlevels, 'title': atitle, 'extend': 'both', 'ylim': 950.0, 'ylog': False }
        cs=plotZM(fig,ax,tgsi.wgvin, xg, zg, plotOpt=plotOpt)

    figname=os.path.join(figdir,'%s_bgvwgv.png'%(case))
    plt.savefig(figname)
    
    vvstdmean={}; vhlmean={}; vvlmean={}; varname={}
    for i in range(6):
    
        varname[i] = tgsi.var[i].strip()
    
        print 'plotting %s scale %s'  %(varname[i],scale[nlev][i])
    
        fig=plt.figure(figsize=(8,12))
        ax=plt.subplot(3,1,1)
       
        z = tgsi.corzin[:,:,i]
        if not 'f06' in case:
            z = z*scale[nlev][i]
        vvstdmean[i] = np.mean(z,axis=0)
        atitle='correlation max=%s min=%s'%(z.max(),z.min())
        if iff == 0:
            plotOpt = { 'title': atitle, 'extend': 'both', 'x_label': False  } 
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
            clevels[i]=cs.levels
        else:
            plotOpt = { 'levels': clevels[i], 'title': atitle, 'extend': 'both', 'x_label': False  }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
    
        ax=plt.subplot(3,1,2)
        z = tgsi.hscalesin[:,:,i]/1000.
        vhlmean[i] = np.mean(z,axis=0) 
        #vhlmean[i] = z.max(axis=0)
        #vhlmean[i] = stats.mode(z,axis=0)[0][0]
        #print 'vhlmean ', vhlmean[i].shape
        vmin[i]=min(vmin[i],vhlmean[i].min())
        vmax[i]=max(vmax[i],vhlmean[i].max())
        atitle='horizontal scales (km) max=%s min=%s'%(z.max(),z.min())
        if iff == 0:
            plotOpt = { 'title': atitle, 'extend': 'both', 'x_label': False  }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
            hlevels[i]=cs.levels
        else:
            plotOpt = { 'levels': hlevels[i], 'title': atitle, 'extend': 'both', 'x_label': False  }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
    
        ax=plt.subplot(3,1,3)
        z = 1./tgsi.vscalesin[:,:,i]
        vvlmean[i] = np.mean(z,axis=0)
        atitle='vertical scales max=%s, min=%s'%(z.max(),z.min())
        if iff == 0:
            plotOpt = { 'title': atitle, 'extend': 'both' }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
            vlevels[i]=cs.levels
        else:
            plotOpt = { 'levels': vlevels[i], 'title': atitle, 'extend': 'both' }
            cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
    
        plt.suptitle('variable = %s' % varname[i],fontsize=14,fontweight='bold')
        figname=os.path.join(figdir,'%s_%s.png'%(case,varname[i]))
        plt.savefig(figname)

    vstdmean[iff]=vvstdmean
    hlmean[iff]=vhlmean
    vlmean[iff]=vvlmean
    
    print 'plotting corq2 scale %s'%(scale[nlev][3])
    plt.figure()
    ax=plt.subplot(1,1,1)
    z = tgsi.corq2in * scale[nlev][3]
    atitle='corq2 max=%s min=%s'%(z.max(),z.min())
    if iff == 0:
        plotOpt = { 'title': atitle } 
        cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
        corq2levels=cs.levels
    else:
        plotOpt = { 'levels': corq2levels, 'title': atitle}
        cs=plotZM(fig,ax,z, xg, zg, plotOpt=plotOpt, modelLevels = glev[iff])
    figname=os.path.join(figdir,'%s_corq2.png'%(case))
    plt.savefig(figname)
    
    print 'plotting surface pressure'
    plt.figure(figsize=(8,10))
    plt.subplot(1,2,1)
    z=tgsi.corpin
    if not 'f06' in case:
        z=z*scale[nlev][-1]
    plt.plot(glat,z,'b.')
    plt.plot(glat,z,'b-')
    if iff == 0:
        xmin, xmax, cymin, cymax = plt.axis()
    else:
        plt.ylim(cymin,cymax)
    plt.xlabel('latitude')
    plt.xlim(-90,90)
    plt.ylim(0.0,0.12)
    plt.ylabel('std max=%s min=%s'%(tgsi.corpin.max(),tgsi.corpin.min()),fontsize=12,fontweight='normal')
    plt.title('correlation',fontsize=12,fontweight='normal')
    plt.subplot(1,2,2)
    z=tgsi.hscalespin/1000.
    plt.plot(glat,z,'r-')
    plt.plot(glat,z,'r.')
    plt.ylim(50.0,180.0)
    plt.xlabel('latitude')
    plt.xlim(-90,90)
    plt.ylabel('horizontal scales (km)',fontsize=12,fontweight='normal')
    plt.title('horizontal scales (km) max=%s min=%s'%(z.max(),z.min()),fontsize=12,fontweight='normal')
    
    plt.suptitle('variable = ps',fontsize=14,fontweight='bold')
    figname=os.path.join(figdir,'%s_ps.png'%(case))
    plt.savefig(figname)
    
    if plot_map:
        proj = lmapping.Projection('mill',resolution='c',llcrnrlat=-80.,urcrnrlat=80.)
        bmap = lmapping.createMap(proj)
        gglon,gglat = np.meshgrid(glon,glat)
        xm,ym = bmap(gglon,gglat)
    
        print 'plotting sst'
        plt.figure()
        plt.subplot(2,1,1)
        lmapping.drawMap(bmap,proj)
        z = tgsi.corsstin
        c = bmap.contourf(xm,ym,z,21,cmap=cmappos,extend='both')
        bmap.colorbar(c,'right',size='5%',pad='2%')
        plt.title('correlation',fontsize=12,fontweight='normal')
    
        plt.subplot(2,1,2)
        lmapping.drawMap(bmap,proj)
        z = tgsi.hsstin
        c = bmap.contourf(xm,ym,z,21,cmap=cmappos,extend='both')
        bmap.colorbar(c,'right',size='5%',pad='2%')
        plt.title('horizontal scales (km)',fontsize=12,fontweight='normal')
    
        plt.suptitle('variable = sst',fontsize=14,fontweight='bold')
        figname=os.path.join(figdir,'%s_sst.png'%(case))
        plt.savefig('sst.png')

if plot_meanprofile:

    hl_info={}
    hybens_info_dir='/scratch2/GFDL/gfdlscr/Mingjing.Tong/global_workflow/shield_develop/sorc/gsi.fd/fix'
    hybens_info='%s/global_hybens_info.l91.txt'%(hybens_info_dir)
    hl_info[91], vl, betaS, betaE = get_hybens_info(hybens_info)
    hybens_info='%s/global_hybens_info.l64.txt'%(hybens_info_dir)
    hl_info[64], vl, betaS, betaE = get_hybens_info(hybens_info)
    hybens_info='%s/global_hybens_info.l127.txt'%(hybens_info_dir)
    hl_info[127], vl, betaS, betaE = get_hybens_info(hybens_info)


    mcolor = ['r', 'g', 'b', 'm','c','y']
    for i in range(6):
        """ mean horizontal scale """
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        for iff, ff in enumerate(fnames):
            case=os.path.basename(ff)
            nlev=len(glev[iff])
            if nlev == 91:
                label='%s L%s'%(case.split('.')[-1],nlev)
            else:
                label='gfs L%s'%(nlev)
            ax.plot(hlmean[iff][i],plevs[iff],label=label,marker='o',color=mcolor[iff],mfc=mcolor[iff],
                    mec=mcolor[iff],linewidth=0.5,markersize=4,alpha=0.8)
            if i == 0:
                if nlev == 91:
                    ax.plot(hl_info[91][::-1],plevs[iff],label='horLocLen 91',marker='o',color='k',mfc='k',
                            mec='k',linewidth=0.5,markersize=4,alpha=0.8)
                elif nlev == 64:
                    ax.plot(hl_info[64][::-1],plevs[iff],label='horLocLen 64',marker='.',color='k',mfc='k',
                            mec='k',linewidth=0.5,markersize=4,alpha=0.8)
                else:
                    ax.plot(hl_info[127][::-1],plevs[iff],label='horLocLen 127',marker='*',color='k',mfc='k',
                            mec='k',linewidth=0.5,markersize=4,alpha=0.8)
    
        ax.invert_yaxis()
        ax.set_yscale('log')
        ax.legend(frameon=False,fontsize='small',numpoints=1,
                  handletextpad=0.0,borderpad=0.0)
        subs = [1,2,3,4,5]
        y1loc = ticker.LogLocator(base=10., subs=subs)
        ax.yaxis.set_major_locator(y1loc)
        fmt = ticker.FormatStrFormatter("%g")
        ax.yaxis.set_major_formatter(fmt)
        for t in ax.get_yticklabels():
            t.set_fontsize(8)
        #ax.set_xlim(-2.0, 37.0)
        #ax.set_ylim(400., 0.01)
        ax.set_ylabel('pressure (hPa)', fontsize=12)
        ax.set_xlabel('hv (km)', fontsize=12)
        figname=os.path.join(figdir,'mean_hscale_plev_%s.png'%(varname[i]))
        plt.savefig(figname)
    
        """ mean variance """
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        for iff, ff in enumerate(fnames):
            case=os.path.basename(ff)
            nlev=len(glev[iff])
            if nlev == 91:
                label='%s L%s'%(case.split('.')[-1],nlev)
            else:
                label='gfs L%s'%(nlev)
            if iff == 0:
                exponent = glib.float10Power(np.abs(vstdmean[iff][i]).max())
            var=vstdmean[iff][i]/np.power(10,exponent)
            ax.plot(var,plevs[iff],label=label,marker='o',color=mcolor[iff],mfc=mcolor[iff],
                    mec=mcolor[iff],linewidth=0.5,markersize=4,alpha=0.8)
    
        ax.invert_yaxis()
        ax.set_yscale('log')
        ax.legend(frameon=False,fontsize='small',numpoints=1,
                  handletextpad=0.0,borderpad=0.0)
        subs = [1,2,3,4,5]
        y1loc = ticker.LogLocator(base=10., subs=subs)
        ax.yaxis.set_major_locator(y1loc)
        fmt = ticker.FormatStrFormatter("%g")
        ax.yaxis.set_major_formatter(fmt)
        for t in ax.get_yticklabels():
            t.set_fontsize(8)
        #ax.set_xlim(-2.0, 37.0)
        #ax.set_ylim(400., 0.01)
        ax.set_ylabel('pressure (hPa)', fontsize=12)
        ax.set_xlabel('%s variance x$\mathregular{10^{%d}}$'%(varname[i], exponent), fontsize=12)
        figname=os.path.join(figdir,'mean_variance_plev_%s.png'%(varname[i]))
        plt.savefig(figname)
    
        """ mean vertical length scale """
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        for iff, ff in enumerate(fnames):
            case=os.path.basename(ff)
            nlev=len(glev[iff])
            if nlev == 91:
                label='%s L%s'%(case.split('.')[-1],nlev)
            else:
                label='gfs L%s'%(nlev)
            ax.plot(vlmean[iff][i],plevs[iff],label=label,marker='o',color=mcolor[iff],mfc=mcolor[iff],
                    mec=mcolor[iff],linewidth=0.5,markersize=4,alpha=0.8)
    
        ax.invert_yaxis()
        ax.set_yscale('log')
        ax.legend(frameon=False,fontsize='small',numpoints=1,
                  handletextpad=0.0,borderpad=0.0)
        subs = [1,2,3,4,5]
        y1loc = ticker.LogLocator(base=10., subs=subs)
        ax.yaxis.set_major_locator(y1loc)
        fmt = ticker.FormatStrFormatter("%g")
        ax.yaxis.set_major_formatter(fmt)
        for t in ax.get_yticklabels():
            t.set_fontsize(8)
        #ax.set_xlim(-2.0, 37.0)
        #ax.set_ylim(400., 0.01)
        ax.set_ylabel('pressure (hPa)', fontsize=12)
        ax.set_xlabel('vertical length scale (# grid)', fontsize=12)
        figname=os.path.join(figdir,'mean_vscale_plev_%s.png'%(varname[i]))
        plt.savefig(figname)
    
        """ mean variance difference """
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        ii=0
        for iff, ff in enumerate(fnames):
            nlev=len(glev[iff])
            if nlev == 91:
                ii=ii+1
                if ii == 1: 
                    var0=vstdmean[iff][i]            
                else:
                    var=np.divide(vstdmean[iff][i], var0, out=np.ones_like(vstdmean[iff][i]), where=var0!=0)
                    ax.plot(var,glev[iff],marker='o',color=mcolor[iff],mfc=mcolor[iff],
                            mec=mcolor[iff],linewidth=0.5,markersize=4,alpha=0.8)
    
        plt.xlabel('%s variance'%(varname[i]),fontsize=12)
        plt.ylabel('model level',fontsize=12)
        figname=os.path.join(figdir,'variance_diff_%s.png'%(varname[i]))
        plt.savefig(figname)
    
    
    for iff, ff in enumerate(fnames):
        case=os.path.basename(ff)
        nlev=len(glev[iff])
        for i in range(6):
            fig,ax = plt.subplots(1,1,figsize=(6,6))
            ax.plot(hlmean[iff][i],glev[iff],label=varname[i],linewidth=2.0)
            if i == 0:
                hlorg = hl_info[nlev][::-1]
                hlnew = (hlorg[0] - hlmean[iff][i][0]) + hlmean[iff][i]
                file1 = open('hybens_info_new_%s'%(case), 'w')
                for a in hlnew:
                    Lines = file1.write('  %s \n'%(a))
                file1.close()
                ax.plot(hlorg,glev[iff],label='horLocLen',linewidth=2.0)
                ax.plot(hlnew,glev[iff],label='horLocnew',linewidth=1.0,color='r')
            plt.legend(loc=0,fontsize='small',numpoints=1)
            plt.xlabel('%s horizontal scale'%(varname[i]),fontsize=12)
            plt.ylabel('model level',fontsize=12)
            plt.xlim(vmin[i],vmax[i])          
            figname=os.path.join(figdir,'L%s_%s_mean_hscale_%s_mlev.png'%(case.split('.')[-1],len(glev[iff]),varname[i]))
            plt.savefig(figname)
        
        #plt.show()
sys.exit(0)
