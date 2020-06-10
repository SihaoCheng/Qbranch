import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, votable, fits
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic # Low-level frames
import astropy.units as u


def get_v_and_ev(table, U0=10, V0=7.5, W0=6.5, v_drift=0, with_Oort=False):
    c = SkyCoord(ra=table['ra']*u.deg, dec=table['dec']*u.deg, \
             pm_ra_cosdec=table['pmra']*u.mas/u.yr, pm_dec=table['pmdec']*u.mas/u.yr)
    pml = c.transform_to(Galactic).pm_l_cosb.value
    pmb = c.transform_to(Galactic).pm_b.value
    factor = 1/1000 * (1/table['parallax']*1000)*1.5*10**8/3600/24/365.24
    if ~with_Oort:
        vL = pml*factor+(V0+v_drift)*np.cos(table['l']/180*np.pi)-\
              U0*np.sin(table['l']/180*np.pi)
        vB = pmb*factor+W0*np.cos(table['b']/180*np.pi) -\
                        ((V0+v_drift)*np.sin(table['l']/180*np.pi)+U0*np.cos(table['l']/180*np.pi))*np.sin(table['b']/180*np.pi)
    if with_Oort:
        A       = 15.3
        B       = -11.9
        C       = -3.2
        K       = -3.3
        d       = 1 / table['parallax']# * 1000
        l_rad   = table['l'] / 180 * np.pi
        b_rad   = table['b'] / 180 * np.pi
        pml      = pml - \
                  ( A * np.cos(2*l_rad) + B - C * np.sin(2*l_rad) ) * np.cos(b_rad) * d / factor + \
                  (V0 + v_drift) * np.cos(l_rad) / factor - U0 * np.sin(l_rad) / factor
        pmb      = pmb + \
                  (( (A * np.sin(2*l_rad) + C * np.cos(2*l_rad) + K) * np.sin(2*b_rad) / 2 ) * d + \
                  W0 * np.cos(b_rad) - ((V0 + v_drift) * np.sin(l_rad) + U0 * np.cos(l_rad)) * np.sin(b_rad)) / factor
    
    #v = ( vL**2 + vB**2 )**0.5
    #ev = (table['pmra_error']**2 + table['pmdec_error']**2)**0.5*factor
    U = -pml*factor * np.sin(table['l']/180*np.pi)#+U0
    V = pml*factor * np.cos(table['l']/180*np.pi)#+V0
    W = pmb*factor * np.cos(table['b']/180*np.pi)#+W0
    return U, V, W


def plot_HR(table,plt_list,pl_only_all=False,plot_type='HR',plt_box=True,\
            box=[-0.40,0.10,13.1,0.30,1],hess=False,res=50,option_plt_list=[],ms=3):
    xmin = box[0]
    xmax = box[1]
    offset = box[2]
    width = box[3]
    slope = box[4]
    func = lambda x, offset, slope: offset+x*slope
    func_select = lambda x, y, offset, slope, width, xmin, xmax:\
        (np.abs(offset+x*slope-y)<width) * (x>xmin) * (x<xmax)
    
    plt.figure(figsize=(9,6))
    if plot_type == 'HR':
        m_M = 5*np.log10(1/table['parallax']*1000)-5
        x = (table['bp_rp'])#(table['spec_g']) - table['spec_r']#
        y = (table['phot_g_mean_mag']-m_M)
        ym = 16; yM = 8; plt.ylim(ym, yM)
        xm = -0.6; xM = 1.25; plt.xlim(xm, xM)
        plt.xlabel('bp - rp')
        plt.ylabel('G')
    if plot_type == 'HR_ug':
        m_M = 5*np.log10(1/table['parallax']*1000)-5
        x = (table['sdss_u']) - table['sdss_g']#
        y = (table['phot_g_mean_mag']-m_M)
        ym = 16; yM = 8; plt.ylim(ym, yM)
        xm = -0.6; xM = 1.25; plt.xlim(xm, xM)
        plt.xlabel('u - g')
        plt.ylabel('G')
    if plot_type == 'HR_gr':
        m_M = 5*np.log10(1/table['parallax']*1000)-5
        x = (table['sdss_g']) - table['sdss_r']#
        y = (table['phot_g_mean_mag']-m_M)
        ym = 16; yM = 8; plt.ylim(ym, yM)
        xm = -0.6; xM = 1.25; plt.xlim(xm, xM)
        plt.xlabel('g - r')
        plt.ylabel('G')
    if plot_type == 'CC':
        x = (table['bp_rp'])
        y = (table['sdss_u']) - table['sdss_g']
        ym = -0.7; yM = 2; plt.ylim(ym, yM)
        xm = -0.6; xM = 1.25; plt.xlim(xm, xM)
        plt.xlabel('bp - rp')
        plt.ylabel('u - g')
    if plot_type == 'CC_gr_MWDD' or plot_type == 'CC_gr_sdss':
        if plot_type == 'CC_gr_MWDD':
            x = (table['MWDD_g']) - table['MWDD_r']
            y = (table['MWDD_u']) - table['MWDD_g']
        if plot_type == 'CC_gr_sdss':
            x = (table['sdss_g']) - table['sdss_r']
            y = (table['sdss_u']) - table['sdss_g']
        ym = -0.7; yM = 2; plt.ylim(ym, yM)
        xm = -0.6; xM = 1.25; plt.xlim(xm, xM)
        plt.xlabel('g - r')
        plt.ylabel('u - g')    
    if plot_type == 'Var':
        phot_noise_excess = table['phot_g_mean_flux_error']*4.4/\
                (table['phot_g_mean_flux']*4.4/table['phot_g_n_obs'])**0.5
        x = table['bp_rp']
        y = phot_noise_excess    
        ym = 0; yM = 10; plt.ylim(ym, yM)
        xm = -1; xM = 1.7; plt.xlim(xm, xM)
        plt.xlabel('bp - rp')
        plt.ylabel('photometric noise excess ( std_obs / std_poission )')
    if plot_type == 'lb':
        x = table['l']
        y = table['b']    
        ym = -90; yM = 90; plt.ylim(ym, yM)
        xm = 360; xM = 0; plt.xlim(xm, xM)
        plt.xlabel('l')
        plt.ylabel('b')
    
    if pl_only_all != False:
        if pl_only_all == True:
            if hess == False:
                plt.plot(x,y,'.',alpha=0.5,ms=1)
                plt.grid()
            else:
                H, xedges, yedges = np.histogram2d(x, y, bins=(res*2,res),\
                                                   range=((min(xm,xM),max(xm,xM)),\
                                                          (min(ym,yM),max(ym,yM))))
                H = H.T
                X_display, Y_display = np.meshgrid(xedges, yedges)
                plt.pcolormesh(X_display, Y_display, np.log10(H))
                plt.colorbar()
        else:
            plt.scatter(x,y,c=table[pl_only_all],s=1,\
                        vmin=np.percentile(table[pl_only_all],10),\
                        vmax=np.percentile(table[pl_only_all],90))
            plt.colorbar()
            plt.grid()
            plt.title(pl_only_all)
        if plt_box==True:
            plt.plot([xmin, xmin, xmax, xmax, xmin],\
                     [func(xmin,offset,slope)-width,func(xmin,offset,slope)+width,\
                      func(xmax,offset,slope)+width,func(xmax,offset,slope)-width,\
                      func(xmin,offset,slope)-width ])
        #plt.show()
        selected = func_select(x,y,offset,slope,width,xmin,xmax)
        return selected
    else:
        ismag = (table['ismag'].mask==False) + (table['ismag_more']==True)      
        DA = (table['spectype'] == 'DA') | (table['spectype'] == 'DAH') | (table['spectype'] == 'DAH:') |\
             (table['spectype'] == 'DA:') 
        DB = (table['spectype'] == 'DB') | (table['spectype'] == 'DBA') | (table['spectype'] == 'DO') |\
               (table['spectype'] == 'DBA:')|(table['spectype'] == 'DB:')|(table['spectype'] == 'DBAH:')
        DQ = (table['spectype'] == 'DQ')|(table['spectype'] == 'DQ_CI')|(table['spectype']=='DQ_CII')|\
                (table['spectype'] == 'DQ:')
        DC = (table['spectype'] == 'DC') | (table['spectype'] == 'DC:') | (table['spectype'] == 'DC-DQ')
        DZ = (table['spectype'] == 'DZ')|(table['spectype'] == 'DZA')|(table['spectype'] == 'DAZ')|\
                    (table['spectype'] == 'DZ:')|(table['spectype'] == 'DBAZ')
        DAeB = (table['spectype'] == 'DAB')|(table['spectype'] == 'DAB:')|(table['spectype'] == 'DAO')|\
                    (table['spectype'] == 'DAB:')
        DA_non_mag = DA*~ismag
        others = ~DA*~DB*~DC*~DQ
        non_DA = ~DA
        
        if 'all' in plt_list:
            plt.plot(x,y,'.',label='all',ms=ms,alpha=0.5)
        if 'DA_non_mag' in plt_list:
            plt.plot(x[DA_non_mag],y[DA_non_mag],'.',ms=ms,label='DA_non_mag',alpha=1)
        if 'DA' in plt_list:
            if len(plt_list)==1:
                alpha=1
            else: alpha=1
            plt.plot(x[DA],y[DA],'.',ms=ms,label='DA',alpha=alpha)
        if 'non_DA' in plt_list:
            plt.plot(x[non_DA],y[non_DA],'.',ms=ms,label='non_DA')
        if 'DB' in plt_list:
            plt.plot(x[DB],y[DB],'.',ms=ms,label='DB')              
        if 'DQ' in plt_list:
            plt.plot(x[DQ],y[DQ],'.',ms=ms,label='DQ')
        if 'DC' in plt_list:
            plt.plot(x[DC],y[DC],'.',ms=ms,label='DC')
        if 'DZ' in plt_list:
            plt.plot(x[DZ],y[DZ],'.',ms=ms,label='DZ')
        if 'DAeB' in plt_list:
            plt.plot(x[DAeB],y[DAeB],'.',ms=ms,label='DAeB')
        if 'mag' in plt_list:
            plt.plot(x[ismag],y[ismag],'.y',label='magnetic',markersize=15,alpha=0.3)
        if 'other' in plt_list:
            plt.plot(x[others],y[others],'.',ms=ms,label='other')
        for i in option_plt_list:
            selected = (table['spectype'] == i)
            plt.plot(x[selected],y[selected],'.',label=i,ms=ms)
        plt.legend()
        if plt_box == True:
            plt.plot([xmin, xmin, xmax, xmax, xmin],\
                     [func(xmin,offset,slope)-width,func(xmin,offset,slope)+width,\
                      func(xmax,offset,slope)+width,func(xmax,offset,slope)-width,\
                      func(xmin,offset,slope)-width ])
        selected = func_select(x,y,offset,slope,width,xmin,xmax)
        #plt.plot(x[selected],y[selected],'.',label='selected')
        plt.grid()
        #plt.show()
        return selected


def hess(x, y, xm, xM, res1, ym, yM, res2, pl=True,log=True):
    H, xedges, yedges = np.histogram2d(x, y, bins=(res1,res2),\
                                       range=((min(xm,xM),max(xm,xM)),\
                                              (min(ym,yM),max(ym,yM))))
    H = H.T
    X_display, Y_display = np.meshgrid(xedges, yedges)
    if pl==True:
        if log==True:
            plt.pcolormesh(X_display, Y_display, np.log10(H))
        else:
            plt.pcolormesh(X_display, Y_display, H)
    return H, X_display, Y_display


def hess_median(x, y, z, xm, xM, res1, ym, yM, res2):
    H, xedges, yedges = np.histogram2d(x, y, bins=(res1,res2),\
                                       range=((min(xm,xM),max(xm,xM)),\
                                              (min(ym,yM),max(ym,yM))))
    H = H.T
    X_display, Y_display = np.meshgrid(xedges, yedges)
    H_median = np.zeros_like(H)
    for j in range(len(xedges)-1):
        for i in range(len(yedges)-1):
            H_median[i,j] = np.median(z[(x>xedges[j]) * (x<xedges[j+1]) * (y>yedges[i]) * (y<yedges[i+1])])
    not_nan = ~np.isnan(H_median)
    plt.pcolormesh(X_display, Y_display, H_median,\
                   vmin=np.percentile(H_median[not_nan],10),\
                   vmax=np.percentile(H_median[not_nan],90))
    return H_median, X_display, Y_display


def running_median_x(x, y, xm, xM, res1, ym, yM, res2, pl=True, fmt='', method='median'):
    median = np.zeros(res2)
    for i in range(res2):
        dy = (yM-ym)/res2
        median[i] = eval('np.'+method)(x[(y>ym+dy*i)*(y<ym+dy*(i+1))])
    if pl==True:
        plt.plot(median, np.linspace(ym,yM,res2)+dy/2, fmt)
        plt.xlim(xm,xM)
        plt.ylim(ym,yM)
    return median

def plot_lines(element,**kwarg):
    if 'H' in element:
        plt.axvline(6563,c='orange',alpha=0.1,**kwarg)
        plt.axvline(4861.3615,c='orange',alpha=0.1,**kwarg)
        plt.axvline(4340.462,c='orange',alpha=0.1,**kwarg)
        plt.axvline(4101.74,c='orange',alpha=0.1,**kwarg)
        plt.axvline(3970.072,c='orange',alpha=0.1,**kwarg)
        plt.axvline(3646,c='orange',alpha=0.1,**kwarg)            
    if 'Na' in element:
        plt.axvline(5893,c='y',alpha=0.3,**kwarg)
    if 'He' in element:
        plt.axvline(5875.62,c='r',**kwarg)
        plt.axvline(6678.15,c='r',**kwarg)
        plt.axvline(5015.67,c='r',**kwarg)
        plt.axvline(4921.93,c='r',**kwarg)
        plt.axvline(4713.14,c='r',**kwarg)
        plt.axvline(4471.48,c='r',**kwarg)
        plt.axvline(3888.6,c='r')
        plt.axvline(4026.19,c='r')
        plt.axvline(4120.8,c='r')
        plt.axvline(5047.7,c='r')
        plt.axvline(4387.9,c='r')
        plt.axvline(7065,c='r')  
        plt.axvline(7281,c='r')  
        plt.axvline(5047.7,c='r')  
        plt.axvline(5047.7,c='r')  
    if 'Ca' in element:
        plt.axvline(3969,c='r',alpha=0.1)
        plt.axvline(3934,c='r',alpha=0.1)
    if 'C' in element:
        plt.axvline(5170.462,c='b',alpha=0.1)
        plt.axvline(4269.0197,c='b',alpha=0.1)
        plt.axvline(7119.90,c='b',alpha=0.1)
        plt.axvline(8335.1443,c='b',alpha=0.1)
        plt.axvline(6013.22  ,c='b',alpha=0.1)
        plt.axvline(5800.594,c='b',alpha=0.1)
        plt.axvline(5380.330,c='b',alpha=0.1)
        plt.axvline(5052.15,c='b',alpha=0.1)
        plt.axvline(5039.07,c='b',alpha=0.1)
        plt.axvline(4932.02,c='b',alpha=0.1)
        plt.axvline(4771.73,c='b',alpha=0.1)
        plt.axvline(4478.58,c='b',alpha=0.1)
        plt.axvline(4371.38,c='b',alpha=0.1)
        plt.axvline(4228.326,c='b',alpha=0.1)
        plt.axvline(4065.24,c='b',alpha=0.1)
        plt.axvline(4029.41,c='b',alpha=0.1)
        plt.axvline(3961.403,c='b',alpha=0.1)
        plt.axvline(3833.34,c='b',alpha=0.1)
        plt.axvline(9094.83,c='b',alpha=0.1)
        plt.axvline(9405.72,c='b',alpha=0.1)
        plt.axvline(5889.772,c='b',alpha=0.3)
        plt.axvline(6587,c='b',alpha=0.3)
        plt.axvline(6828,c='b',alpha=0.3)
        plt.axvline(7236,c='b',alpha=0.3)
       
        plt.axvline(4382,c='b',ls='--',alpha=0.1)
        plt.axvline(4737,c='b',ls='--',alpha=0.1)
        plt.axvline(5165,c='b',ls='--',alpha=0.1)
        plt.axvline(5635,c='b',ls='--',alpha=0.1)

        
        
        
def select_type(table,spectype):
    ismag = (table['ismag'].mask==False) + (table['ismag_more']==True)      
    DA = (table['spectype'] == 'DA') | (table['spectype'] == 'DAH') | (table['spectype'] == 'DAH:') |\
         (table['spectype'] == 'DA:') 
    DB = (table['spectype'] == 'DB') | (table['spectype'] == 'DBA') | (table['spectype'] == 'DO') |\
           (table['spectype'] == 'DBA:')|(table['spectype'] == 'DB:')|(table['spectype'] == 'DBAH:')
    DQ = (table['spectype'] == 'DQ')|(table['spectype'] == 'DQ_CI')|(table['spectype']=='DQ_CII')|\
            (table['spectype'] == 'DQ:')
    DC = (table['spectype'] == 'DC') | (table['spectype'] == 'DC:') | (table['spectype'] == 'DC-DQ')
    DZ = (table['spectype'] == 'DZ')|(table['spectype'] == 'DZA')|(table['spectype'] == 'DAZ')|\
                (table['spectype'] == 'DZ:')|(table['spectype'] == 'DBAZ')
    DAeB = (table['spectype'] == 'DAB')|(table['spectype'] == 'DAB:')|(table['spectype'] == 'DAO')|\
                (table['spectype'] == 'DAB:')
    DA_non_mag = DA*~ismag
    others = ~DA*~DB*~DC*~DQ
    non_DA = ~DA
    return eval(spectype)

def func_select(x, y, offset, slope, width, xmin, xmax):
    return (np.abs(offset+x*slope-y)<width) * (x>xmin) * (x<xmax)


def plot_func_select_box(offset, slope, width, xmin, xmax, **kwarg):
    func = lambda x, offset, slope: offset+x*slope
    plt.plot([xmin, xmin, xmax, xmax, xmin],\
                     [func(xmin,offset,slope)-width,func(xmin,offset,slope)+width,\
                      func(xmax,offset,slope)+width,func(xmax,offset,slope)-width,\
                      func(xmin,offset,slope)-width ],**kwarg)
    return None


def open_synthetic_result(name):
    mass = np.zeros(0)
    age = np.zeros(0)
    v = np.zeros(0)
    bp_rp = np.zeros(0)
    G = np.zeros(0)
    DB = np.zeros(0)

    synthetic_result = np.load(name)[0]['results']
    for i in range(len(synthetic_result)):
        mass = np.concatenate((mass,synthetic_result[i][1]))
        age = np.concatenate((age,synthetic_result[i][2]))
        v = np.concatenate((v,synthetic_result[i][3]))
        bp_rp = np.concatenate((bp_rp,synthetic_result[i][4]))
        G = np.concatenate((G,synthetic_result[i][5]))
        DB = np.concatenate((DB,synthetic_result[i][6]))
    return mass, age, v, bp_rp, G, DB