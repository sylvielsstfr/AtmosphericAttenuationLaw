#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 19:43:37 2018

@author: dagoret
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
import os,sys
from matplotlib.backends.backend_pdf import PdfPages 
from scipy import interpolate
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit





#------------------------------------------------------------------------
# Definition of data format for the atmospheric grid
#-----------------------------------------------------------------------------
WLMIN=300.  # Minimum wavelength : PySynPhot works with Angstrom
WLMAX=1100. # Minimum wavelength : PySynPhot works with Angstrom
WL=np.arange(WLMIN,WLMAX,1) # Array of wavelength in Angstrom
NBWL=len(WL)


#----------------------------------------------------------------------------------------
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: 
        return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    if len(x)%2==0: # even case
        return y[(window_len/2):-(window_len/2)] 
    else:           #odd case
        return y[(window_len/2-1):-(window_len/2)] 
    
#--------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------    
#   FIT
#----------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------   
def bougline(x, a, b):
    return a*x + b
#-----------------------------------------------------------------------------------
def Varbougline(x,popt,pcov):

    Var=x*pcov[0,0]+pcov[1,1]+x*(pcov[0,1]+pcov[1,0])
    return Var
#---------------------------------------------------------------------------------    
    
#---------------------------------------------------------------------------------------------    
def FitBougherLine(theX,theY,theSigY):
    
    
    # range to return  the fit
    xfit=np.linspace(0,theX.max()*1.1,20)    
   
    
    # find a first initialisation approximation with polyfit
    theZ = np.polyfit(theX,theY, 1)
    
    
    # TEST IF  ERROR are null or negative (example simulation)
    if np.any(theSigY<=0):
        pol = np.poly1d(theZ)
        yfit=pol(xfit)
        errfit=np.zeros(len(xfit))
        
    else:
        # do the fit including the errors
        popt, pcov = curve_fit(bougline, theX, theY,p0=theZ,sigma=theSigY,absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
    
        #print "popt = ",popt,' pcov',pcov,' perr',perr
    
    
        #compute the chi-sq
        chi2=np.sum( ((bougline(theX, *popt) - theY) / theSigY)**2)
        redchi2=(chi2)/(len(theY)-2)
    
        #chi2sum=(Yfit-np.array(theY))**2/np.array(theSigY)**2
        #chi2=np.average(chi2sum)*chi2sum.shape[0]/(chi2sum.shape[0]-3)
        #print 'chi2',chi2
    
    
        #p = np.poly1d(theZ)
        #yfit=p(xfit)
    
        pol = np.poly1d(popt)
        yfit=pol(xfit)
        errfit=np.sqrt(Varbougline(xfit,popt,pcov))
    
    return xfit,yfit,errfit
    
        

#--------------------------------------------------------------------------------------
def FitAttenuationSmoothBin(airmasses,transmissions,thetitle,ZMIN=0,ZMAX=0,Wwidth=11,Bwidth=10,Mag=True):
    """
    
    FitAttenuationSmoothBin(airmasses,transmissions,Wwidth=21,Bwidth=20,Mag=True)
    
    """
    if (ZMIN==0 and ZMAX==0):
        WLMIN=WL.min()
        WLMAX=WL.max()
    elif ZMIN==0:
        WLMIN=WL.min()
        WLMAX=ZMAX
    elif ZMAX==0:
        WLMIN=ZMIN
        WLMAX=WL.max()
    else:
        WLMIN=ZMIN
        WLMAX=ZMAX
    
    jet =plt.get_cmap('jet')    
    cNorm  = colors.Normalize(vmin=WLMIN, vmax=WLMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    NBOBS=len(airmasses)
    
    ##########################################################################################
    # attenuation container 
    # attenuation rows: the file index, the airmass, the attenuation for each WL
    ##########################################################################################
    attenuation=np.zeros((NBOBS,2+len(WL)))
    
    # loop on observation
    for idx_obs in np.arange(NBOBS):
        transm=transmissions[:,idx_obs]
            
        tr_smooth=smooth(transm,window_len=Wwidth)
           
            
        # attenuation in data
        attenuation[idx_obs,0]=idx_obs
        attenuation[idx_obs,1]=airmasses[idx_obs]
        attenuation[idx_obs,2:]=tr_smooth  
            
      
         
      
    AIRMASS_MIN=airmasses.min()
    AIRMASS_MAX=airmasses.max()
    
   
   
    ###################################################################################
    ################### Plot the figure ###############################################
    ###################################################################################
    
    # collections to return
    
    all_WL=[]
    all_Y=[]
    all_EY=[]
    
    
    
    
    plt.figure(figsize=(18,10))
    # loop on wavelength indexes
    for idx_wl in np.arange(2,len(WL)+2,Bwidth): 
        if WL[idx_wl-2]<WLMIN:
            continue
        if WL[idx_wl-2]>WLMAX:
            break
        
        colorVal = scalarMap.to_rgba(WL[idx_wl-2],alpha=1)
        
        idx_startwl=idx_wl
        idx_stopwl=min(idx_wl+Bwidth-1,attenuation.shape[1])
        
        thelabel="{}-{} nm".format(int(WL[idx_startwl-2]),int(WL[idx_stopwl-2]) )
        
        WLBins=WL[idx_startwl-2:idx_stopwl-2]
        
        # slice of  flux in wavelength bins
        FluxBin=attenuation[:,idx_startwl:idx_stopwl]
        
                 
        # get the average of flux in that big wl bin
        FluxAver=np.average(FluxBin,axis=1)
        
        FluxAverErr=np.zeros(len(FluxAver))
        
        Y0=FluxAver
        Y1=Y0-FluxAverErr
        Y2=Y0+FluxAverErr
        
        
        if not Mag:        
            plt.fill_between(airmasses,y1=Y1,y2=Y2, where=Y1>0 ,color='grey', alpha=0.3 )        
            plt.yscale( "log" )
        
            # plot the attenuation wrt airmass
            plt.semilogy(airmasses,Y0,'o-',c=colorVal,label=thelabel)
        else:
            
            newY0=2.5*np.log10(Y0)
            newY1=np.zeros(len(Y0))
            newY2=np.zeros(len(Y0))
            
            plt.fill_between(airmasses,y1=newY1,y2=newY2, where=Y1>0 ,color='grey', alpha=0.3 )        
            
            # plot the attenuation wrt airmass
            plt.plot(airmasses,newY0,'o-',c=colorVal,label=thelabel)
            
            Xfit,Yfit,YFitErr=FitBougherLine(airmasses,newY0,theSigY=(newY2-newY1)/2.)
            plt.plot(Xfit,Yfit,'-',c=colorVal)
            plt.plot(Xfit,Yfit+YFitErr,':',c=colorVal)
            plt.plot(Xfit,Yfit-YFitErr,':',c=colorVal)
            
            all_WL.append(np.average(WLBins)) # average wavelength in that bin
            all_Y.append(Yfit[0])             # Y for first airmass z=0
            all_EY.append(YFitErr[0])         # EY extracpolated for that airmass z=0
            
            
    
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.grid(b=True, which='minor', color='red', linestyle='--')
    plt.title(thetitle)
    plt.xlabel("airmass")   
    if not Mag:  
        plt.ylabel("Attenuation") 
    else:
        plt.ylabel("Attenuation (mag)") 
    plt.legend(loc='right', prop={'size':10})  
    
   
    plt.xlim(0.,AIRMASS_MAX*1.3)
    
    plt.show() 
    
    return np.array(all_WL),np.array(all_Y),np.array(all_EY)
    
#-------------------------------------------------------------------------------------        
def PlotOpticalThroughput(wl,thrpt,err,title):
    
    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.errorbar(wl,thrpt,yerr=err,fmt='o',color='blue',ecolor='red')
    
    plt.xlabel('$\lambda$ (nm)' )
    plt.ylabel('total throughput (mag)')
    #plt.grid(b=True, which='major', color='black', linestyle='-')
    #plt.grid(b=True, which='minor', color='red', linestyle='--')
    plt.grid(b=True, which='both')
    plt.show()
    
