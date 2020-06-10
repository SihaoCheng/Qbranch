import emcee
# import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from astropy.coordinates import SkyCoord # High-level coordinates
from astropy.coordinates import ICRS, Galactic # Low-level frames
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u

import WD_HR # a function package 

import sys
sys.path.append('/home/scheng/Gaia')
import WD_models # a module to get the conversion between WD photometry and physical parameters according to WD models


test_number = '0'
t_gap_eff = 0.743 # the effective width of the Q branch in terms of photometric age
Q_IS_MERGER= True # Does the extra-delayed population also have the WD-WD merger delay?
DELAY_INDEX = 1000 # the absolute value of the power index of the delay time distribution for mergers. 
DELAY_CUT = 0.5 # 
AVR_choice = '201866'

# Set parameters
#------------------------------------------------------------------------------
Nv = 13
NQ = 7
delay_lim = 15

# Define star formation parametrization
#------------------------------------------------------------------------------
# t0: age from born, t1: merger from born, tc: cooling age
# t0 = t1 + tc
end_of_SF   = 11
end_of_SF_tc= delay_lim
age_T       = 7 # the starting age of thin-thick transition
age_trans   = 10 # the ending age of thin-thick transition
stromberg_k = 80
n           = 400
n_tc        = 8000
t0          = np.linspace(0, end_of_SF, n)

#------------------------------------------------------------------------------
def para_of_WD(WD_table, atm_type, model):
    WD_model = WD_models.load_model('f', 'f', model, atm_type)
    total_age = WD_model['HR_to_age_cool'](WD_table['bp_rp'], WD_table['G']) +\
                WD_models.MS_age(
                    WD_model['HR_to_mass'](
                        WD_table['bp_rp'], WD_table['G']
                    )
                )*1e-9
    WD_mass = WD_model['HR_to_mass'](WD_table['bp_rp'],WD_table['G'])
    return total_age, WD_mass

def select_WD(WD_table, bprp_excess=1.4, bprp_error=0.10, ev=2, G=22.5, d_ed=8,
              distance=500, astrometric_excess=1.5):
    selected = ((WD_table['phot_bp_rp_excess_factor'] < bprp_excess) *
                (WD_table['parallax'] / WD_table['parallax_error'] > d_ed) *
                (((WD_table['phot_bp_mean_flux_error'] / WD_table['phot_bp_mean_flux'])**2 +
                  (WD_table['phot_rp_mean_flux_error'] / WD_table['phot_rp_mean_flux'])**2
                 )**0.5 < bprp_error) *
                (WD_table['phot_g_mean_mag'] < G) * (WD_table['ev'] < ev) * 
                (1/WD_table['parallax']*1000 < distance) *
               (WD_table['astrometric_excess_noise'] < astrometric_excess))
    selected_table = WD_table[selected].copy()
    for atm_type, model in [['H','f'], ['H','o'], ['He','f'], ['He','o']]:
        total_age, WD_mass = para_of_WD(selected_table, atm_type, model) 
        selected_table['age_'+atm_type+'_'+model] = total_age
        selected_table['mass_'+atm_type+'_'+model] = WD_mass
    print('the length of table after selection is: ' + str(selected.sum()))
    return selected, selected_table

#------------------------------------------------------------------------------
def SFR_merger_1(t0, para):
    return (t0 >= 0.0) * (t0 < end_of_SF) * 1

def SFR_merger_2(t0, para):
    return (t0 >= 0.0) * (t0 < end_of_SF) * (t0 / 11 * 4 + 1)

def SFR_merger_3(t0, para):
    return (t0 >= 0.0) * (t0 < end_of_SF) * (np.exp(-(t0-2.5)**2/2/1.5**2) + 1) 

SFR_merger = SFR_merger_1
SFR = SFR_merger

# merger-delay distribution
def p_merger_delay(t1, dt, d_index):   
    if d_index < 0 and d_index > -5:
        return (t1 >= dt) * (np.abs(t1) + 0.01)**(d_index)
    if d_index > 0 and d_index < 5:
        return (t1 >= dt) * (np.abs(t1) + 0.01)**(-d_index)
    if np.abs(d_index + 2018)<1e-3 or np.abs(d_index - 2018)<1e-3:
        new_time = np.array([ 0.27,   0.81,   1.35,   1.89,   2.43,   2.97,   3.51,   4.05,   4.59,   5.13,
                   5.67,   6.21,   6.75,   7.29,   7.83,   8.37,   8.91,   9.45,   9.99,  10.53,
                  11.07,  11.61,  12.15,  12.69,  13.23,])
        new_rate = np.array([  0,   1.01587302e-13,   7.76334776e-14,   6.08946609e-14,
                       6.92640693e-14,   3.11688312e-14,   2.22222222e-14,   1.24098124e-14,
                       9.81240981e-15,   4.61760462e-15,   8.65800866e-16,   1.73160173e-15,
                       8.65800866e-16,   8.65800866e-16,   1.15440115e-15,   1.44300144e-15,
                       1.73160173e-15,   1.44300144e-15,   2.30880231e-15,   2.02020202e-15,
                       8.65800866e-16,   1.73160173e-15,   1.15440115e-15,   2.02020202e-15,
                       2.88600289e-15,]) * 1e13 #8.36940837e-15
        return interp1d(new_time, new_rate, kind='nearest', bounds_error=False, fill_value=0)(t1)
    new_time = np.array([  0.14,   0.42,   0.7 ,   0.98,   1.26,   1.54,   1.82,   2.1 ,
             2.38,   2.66,   2.94,   3.22,   3.5 ,   3.78,   4.06,   4.34,
             4.62,   4.9 ,   5.18,   5.46,   5.74,   6.02,   6.3 ,   6.58,
             6.86,   7.14,   7.42,   7.7 ,   7.98,   8.26,   8.54,   8.82,
             9.1 ,   9.38,   9.66,   9.94,  10.22,  10.5 ,  10.78,  11.06,
            11.34,  11.62,  11.9 ,  12.18,  12.46,  12.74,  13.02,  13.3 ,
            13.58,  13.86])
    # below are rate for all CO, He, and ONe mergers:
    if np.abs(d_index - 8)<1e-3:
        new_rate = np.array([   1.,   12.,  117.,  120.,  178.,  248.,  171.,  152.,   69.,
         24.,   20.,   33.,   25.,   40.,   38.,   35.,   40.,   35.,
         23.,   37.,   34.,   30.,   28.,   40.,   33.,   33.,   40.,
         30.,   38.,   42.,   47.,   36.,   45.,   22.,   16.,   17.,
         11.,   19.,   15.,   16.,   14.,   18.,   11.,   14.,   12.,
         11.,   17.,    5.,    2.,    0.])
    if np.abs(d_index - 9)<1e-3:
        new_rate = np.array([  0.,   60.,  111.,  207.,  179.,  156.,  210.,  217.,   96.,
         37.,   41.,   34.,   21.,   18.,   16.,   13.,   12.,    9.,
         11.,   23.,   33.,   46.,   40.,   33.,   30.,   45.,   46.,
         28.,   27.,   28.,   17.,   20.,   14.,   14.,   14.,    8.,
         10.,    9.,    8.,   11.,    5.,    7.,    2.,   10.,   13.,
          4.,    4.,    4.,    2.,    0.])
    if np.abs(d_index - 10)<1e-3:
         new_rate = np.array([ 6.,   42.,  185.,  212.,  227.,  215.,  225.,  221.,  122.,
         90.,   75.,   49.,   47.,   35.,   22.,   22.,    9.,   15.,
         11.,    3.,   11.,   12.,   21.,   17.,   17.,   23.,   14.,
         21.,   18.,   20.,   21.,   25.,   10.,   18.,   14.,   15.,
         18.,   19.,    7.,   17.,    9.,   10.,    9.,    3.,    7.,
          4.,    9.,    3.,    1.,    0.])
    if np.abs(d_index - 11)<1e-3:
        new_rate = np.array([ 19.,   38.,  122.,  121.,   75.,   79.,   76.,   66.,   92.,
         66.,   30.,   20.,   27.,   12.,   12.,   14.,   13.,   11.,
          4.,    3.,    0.,    1.,    0.,    0.,    1.,    0.,    4.,
          2.,    1.,    1.,    1.,    2.,    1.,    2.,    1.,    1.,
          3.,    2.,    1.,    1.,    3.,    2.,    0.,    3.,    1.,
          4.,    4.,    4.,    1.,    0.])
    if np.abs(d_index - 12)<1e-3:
        new_rate = np.array([  4.,   52.,  103.,   49.,   29.,   33.,   23.,   25.,   23.,
         30.,   39.,   24.,   16.,   19.,   11.,   12.,   11.,    3.,
          7.,    1.,    1.,    2.,    2.,    3.,    0.,    1.,    1.,
          1.,    0.,    0.,    0.,    1.,    0.,    0.,    0.,    0.,
          0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
          0.,    2.,    0.,    0.,    0.])
    return interp1d(new_time, new_rate, kind='nearest', bounds_error=False, fill_value=0)(t1)
        

def pdf_tct0_array(tc, t0, para, SFR, delay_cut, delay_index):
    array = SFR(t0, para) * p_merger_delay(t0 - tc, delay_cut, delay_index)
    return array, array.mean()

def pdf_tct0(tc, t0, para, normalization_factor_Q, SFR, delay_cut, delay_index):
    return SFR(t0, para) * p_merger_delay(t0 - tc, delay_cut, delay_index) / normalization_factor_Q

def prep_get_v(table):
    c   = SkyCoord(ra=table['ra'] * u.deg, dec=table['dec'] * u.deg,
                   pm_ra_cosdec=table['pmra'] * u.mas / u.yr,
                   pm_dec=table['pmdec'] * u.mas / u.yr)
    pml = c.transform_to(Galactic).pm_l_cosb.value
    pmb = c.transform_to(Galactic).pm_b.value
    factor = 1/1e3 * (1/np.array(table['parallax']) * 1e3) * (1.5 * 1e8 / 3600 / 24 / 365.24)
    return pml, pmb, factor

def get_v_delayed_3D(age, l, b, pml, pmb, factor, v_drift, U, V, W):
    A       = 15.3
    B       = -11.9
    C       = -3.2
    K       = -3.3
    d       = factor / (1.5 * 1e8 / 3600 / 24 / 365.24)
    l_rad   = l / 180 * np.pi
    b_rad   = b / 180 * np.pi
    vL      = pml * factor - \
              ( A * np.cos(2*l_rad) + B - C * np.sin(2*l_rad) ) * np.cos(b_rad) * d + \
              (V + v_drift) * np.cos(l_rad) - U * np.sin(l_rad)
    vB      = pmb * factor + \
              ( (A * np.sin(2*l_rad) + C * np.cos(2*l_rad) + K) * np.sin(2*b_rad) / 2 ) * d + \
              W * np.cos(b_rad) - ((V + v_drift) * np.sin(l_rad) + U * np.cos(l_rad)) * np.sin(b_rad)
    return vL, vB


# Define MCMC functions
#------------------------------------------------------------------------------
def m3(a, b):
    result = np.array([[(a[0,:]*b[:,0]).sum(), (a[0,:]*b[:,1]).sum(), (a[0,:]*b[:,2]).sum()],
                       [(a[1,:]*b[:,0]).sum(), (a[1,:]*b[:,1]).sum(), (a[1,:]*b[:,2]).sum()],
                       [(a[2,:]*b[:,0]).sum(), (a[2,:]*b[:,1]).sum(), (a[2,:]*b[:,2]).sum()],
                       [0,0,0]])
    return result


def LBR_XYZ(l, b, L, B, sx, sy, sz):
    Sigma = np.array([[sx**2, 0, 0], [0, sy**2, 0], [0, 0, sz**2]])
    l_rad = l / 180 * np.pi
    b_rad = b / 180 * np.pi
    M = np.array([[-np.sin(l_rad), -np.sin(b_rad)*np.cos(l_rad), np.cos(b_rad)*np.cos(l_rad)],
                  [np.cos(l_rad), -np.sin(b_rad)*np.sin(l_rad), np.cos(b_rad)*np.sin(l_rad)],
                  [0, np.cos(b_rad), np.sin(b_rad)]])
    #print(A.shape, Sigma.shape, sx.shape)
    Sigma_lbr   = m3(m3(M.T, Sigma), M)
    Sigma_lb    = Sigma_lbr[:2,:2]
    S00         = Sigma_lb[0,0]
    S11         = Sigma_lb[1,1]
    S01         = Sigma_lb[0,1]
    S10         = Sigma_lb[1,0]
    detS        = S00*S11 - S01*S10 # determinant of S
    p_LB      = 1/(detS)**0.5 / (2*np.pi) * np.exp(-1/2/detS *
                    np.dot(np.dot(np.array([L, B, 0]), 
                                  np.array([[S11, -S01, 0],
                                            [-S10, S00, 0],
                                            [0, 0, 0]])
                                 ), np.array([L, B, 0]).T
                          ) )
    p_L        = 1/Sigma_lb[0,0]**0.5 / (2*np.pi)**0.5 * np.exp(-L**2 / 2 / Sigma_lb[0,0])
    p_B        = 1/Sigma_lb[1,1]**0.5 / (2*np.pi)**0.5 * np.exp(-B**2 / 2 / Sigma_lb[1,1])
    return     p_LB, p_L, p_B


def velocity_scatter_3D_1(age, index, v4, vT, v0):
    return (age < age_T) * ((age/4)**index * v4 + v0) + \
           (age >= age_T) * (age < age_trans) * (vT - (age - age_trans) / (age_T - age_trans) * 
             (vT - ((age_T/4)**index * v4 + v0))) + \
           (age >= age_trans) * vT
        
def velocity_scatter_3D_2(age, index, v4, vT, v0):
    return 66*((0.5+age)/(0.5+12))**0.5 * v4


if AVR_choice == 'Wegg':
    velocity_scatter_3D = velocity_scatter_3D_2
    p_list = np.array([  0.46885813,  0.73146451, 1,  72.01905802,   
        1/2.2,   1/1.4,   7.9551574 ,   3.69441241,
        36.90342055,   0.63118769, ]) #2018, age-age_preWD
if AVR_choice == '2018':
    velocity_scatter_3D = velocity_scatter_3D_1
    p_list = np.array([  0.46885813,  0.73146451, 25.56372104,  72.01905802,   
        13.17708167,   0.68067921,   7.9551574 ,   3.69441241,
        36.90342055,   0.63118769, ]) #2018, age-age_preWD
if AVR_choice == '201866': # 66 means thick disc dispersion is 66km/s
    velocity_scatter_3D = velocity_scatter_3D_1
    p_list = np.array([  0.46885813,  0.73146451, 25.56372104,  66.64151581,   
        13.17708167,   0.68067921,   7.9551574 ,   3.69441241,
        32.55700613,   0.63118769, ]) #2018, age-age_preWD
if AVR_choice == '0.72':
    velocity_scatter_3D = velocity_scatter_3D_1
    p_list = np.array([  0.42112294,  0.67356749, 23.67809325,  66.64151581,   
        12.41945989,   0.68036111,   8.7523509 ,   3.96618107,
        32.55700613,   0.6524036 , ]) # 0.7, age-age_preWD
if AVR_choice == '0.7':
    velocity_scatter_3D = velocity_scatter_3D_1
    p_list = np.array([  0.35166413,  0.60137564, 26.04090789,  66.10729793,   
        12.00382117,   0.67964674,   5.88215585,   3.83109947,
        33.8437439 ,   0.6448131    ]) #0.7
    #p_list = np.array([  0.41025833,  0.65832128, 27.4895111 ,  70.49016799,   
    #    13.07623916,   0.68332618,   6.13807293,   3.55430252,
    #    37.20494307,   0.64895071,  ]) #2018

def velocity_density_3D(l, b, vL, vB, age, para_v, is_vL=True):
    #index_UV_t = para_v[0], V_10 = para_v[1], v_T = para_v[2]
    #index_W = para_v[3], W_10 = para_v[4]
    #sy_to_sx = para_v[5]
    #v0 = para_v[6], v0_z = para_v[7], v_T_z = para_v[8]
    #sy_to_sx_T = para_v[9]
    #U = para_v[10]
    #V = para_v[11]
    #W = para_v[12]
    sx = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])
    sy = velocity_scatter_3D(age, para_v[0], para_v[1]*para_v[5], 
                             para_v[2] * para_v[9], para_v[6] * para_v[5])
    sz = velocity_scatter_3D(age, para_v[3], para_v[4], para_v[8], para_v[7])
    p_vLvB, p_vL, p_vB = LBR_XYZ(l, b, vL, vB, sx, sy, sz)
    return p_vLvB


def velocity_density_3D_check(l, b, vL, vB, age, para_v, is_vL):
    sx = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])
    sy = velocity_scatter_3D(age, para_v[0], para_v[1]*para_v[5], 
                             para_v[2] * para_v[9], para_v[6] * para_v[5])
    sz = velocity_scatter_3D(age, para_v[3], para_v[4], para_v[8], para_v[7])
    p_vLvB, p_vL, p_vB = LBR_XYZ(l, b, vL, vB, sx, sy, sz)
    if is_vL == True:
        return p_vL
    else:
        return p_vB 

#------------------------------------------------------------------------------
# the main function for calculating likelihood
#------------------------------------------------------------------------------

def ln_likelihood_pheno(para,
                        mass, age, pml, pmb, factor, l, b,
                        mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                        is_check, velocity_density_3D_check_func=None, is_vL=True,
                        not_fit_UVW=False,not_fit_index=False,fixv=False):
    para_v      = para[0:Nv].copy()
    para_Q      = para[Nv:Nv+NQ].copy()
    delay_index = DELAY_INDEX
    delay_cut   = DELAY_CUT
    if not_fit_UVW == True:
        U   = 11
        V   = 7.5
        W   = 7
        U_Q = U
        V_Q = V
        W_Q = W
    else:
        U   = para_v[10]
        V   = para_v[11]
        W   = para_v[12]
        U_Q = para_Q[3]#11#para_v[10]#
        V_Q = para_Q[4]#7.5#para_v[11]#
        W_Q = para_Q[6]#7#para_v[12]#
    if not_fit_index == True:
        para_v[0] = 0.38
        para_v[3] = 0.5
    if fixv == True:
        para_v[1] = 35
        para_v[2] = 70
        para_v[4] = 17
        para_v[5] = 0.60
        para_v[6] = 0
        para_v[7] = 0
        para_v[8] = 40
        para_v[9] = 0.60
    if test_number == 'noQ':
        para_Q[0] = 0
    if test_number == 'nomerger':
        para_Q[5] = 0
    if test_number == 'noQmerger':
        para_Q[0] = 0
        para_Q[5] = 0
    para_Q[2] = 0

    early   = np.array(1.22 - (age - 0.6) * 0.2 > mass) #for ONe: 1.22; CO: 1.25
    late    = np.array(1.22 - (age - 0.6) * 0.2 < mass)    
    
    age_preWD = WD_models.MS_age(mass)*1e-9
    age_preWD_Q = WD_models.MS_age(mass_Q)*1e-9
    ## tct0_array is the pdf of tc, t0, range: tc: 0~age_lim+delay, t0: 0~end_of_SFR
    ## normalization_factor_Q is the mean of tct0_array
    tct0_array, normalization_factor_Q = pdf_tct0_array(
        np.linspace(0., end_of_SF_tc, n_tc).reshape(-1, 1),
        np.linspace(0, end_of_SF, n).reshape(1, -1),
        None, SFR, delay_cut, delay_index
    )
    ## tc_weight is the norm factor of pdf of t0 given a tc. tct0_array / tc_weight
    tc_weight = tct0_array.mean(1) / normalization_factor_Q
    tc_weight = interp1d(np.linspace(0., end_of_SF_tc, n_tc),
                         tc_weight,
                         fill_value=0.0,
                         bounds_error=False)
    
    ## For the ordinary merger population:
    tct0_array_merger               = tct0_array
    normalization_factor_Q_merger   = normalization_factor_Q
    tc_weight_merger                = tc_weight

    
    ## v_delay for different possible t0
    v_drift_delayed = velocity_scatter_3D(t0, para_v[0], para_v[1], para_v[2], para_v[6])**2 / stromberg_k

    vL_delayed, vB_delayed          = get_v_delayed_3D(
        t0.reshape(1, -1), l.reshape(-1, 1), b.reshape(-1, 1),
        pml.reshape(-1, 1), pmb.reshape(-1, 1), factor.reshape(-1, 1),
        v_drift_delayed.reshape(1,-1), U, V, W
    )    
    vL_Q_delayed, vB_Q_delayed      = get_v_delayed_3D(
        t0.reshape(1, -1), l_Q.reshape(-1, 1), b_Q.reshape(-1, 1),
        pml_Q.reshape(-1, 1), pmb_Q.reshape(-1, 1), factor_Q.reshape(-1, 1),
        v_drift_delayed.reshape(1, -1), U, V, W
    ) 
    vL_delayed_Q, vB_delayed_Q      = get_v_delayed_3D(
        t0.reshape(1, -1), l.reshape(-1, 1), b.reshape(-1, 1),
        pml.reshape(-1, 1), pmb.reshape(-1, 1), factor.reshape(-1, 1),
        v_drift_delayed.reshape(1,-1), U_Q, V_Q, W_Q
    )    
    vL_Q_delayed_Q, vB_Q_delayed_Q  = get_v_delayed_3D(
        t0.reshape(1, -1), l_Q.reshape(-1, 1), b_Q.reshape(-1, 1),
        pml_Q.reshape(-1, 1), pmb_Q.reshape(-1, 1), factor_Q.reshape(-1, 1),
        v_drift_delayed.reshape(1, -1), U_Q, V_Q, W_Q
    ) 
    v_drift     = velocity_scatter_3D(age, para_v[0], para_v[1],
                                      para_v[2], para_v[6]
                                     )**2 / stromberg_k
    v_drift_Q   = velocity_scatter_3D(age_Q, para_v[0], para_v[1],
                                      para_v[2], para_v[6]
                                     )**2 / stromberg_k
    vL, vB      = np.array(get_v_delayed_3D(age, l, b, pml, pmb, 
                                            factor, v_drift, U, V, W)
                          )
    vL_Q, vB_Q  = np.array(get_v_delayed_3D(age_Q, l_Q, b_Q, pml_Q, pmb_Q,
                                            factor_Q, v_drift_Q, U, V, W)
                          )
    if is_check == True:
        velocity_density_3D_func    = velocity_density_3D_check_func
        background_pdf_factor       = 1    
    else:
        velocity_density_3D_func    = velocity_density_3D
        background_pdf_factor       = 700
    ## pdf of v,tc =
        ## [pdf(v) for ordinary * uniform(tc)] * ordinary fraction +
        ## [pdf(v) for merger * merger fraction(t0)
    ## pdf_tct0 is (age_lim+delay) * end_of_SF * pdf(tc,t0)
    
    ## early
    if Q_IS_MERGER == True:
        n_Q_delay = 20
        p_s = velocity_density_3D_func(
            l[early], b[early], vL[early], vB[early], age[early], 
            para_v, is_vL
        ) * (1 - para_Q[0] - para_Q[2] - para_Q[5])
        p_em = (
            velocity_density_3D_func(
                l[early].reshape(-1, 1), b[early].reshape(-1, 1), 
                vL_delayed[early], vB_delayed[early], t0.reshape(1, -1), 
                para_v, is_vL
            ) * pdf_tct0(
                (age-age_preWD)[early].reshape(-1, 1), t0.reshape(1,-1), None,
                normalization_factor_Q, SFR, delay_cut, delay_index
            ) * para_Q[0]
            +
            velocity_density_3D_func(
                l[early].reshape(-1, 1),b[early].reshape(-1, 1),
                vL_delayed[early], vB_delayed[early], t0.reshape(1, -1),
                para_v, is_vL
            ) * pdf_tct0(
                (age-age_preWD)[early].reshape(-1, 1), t0.reshape(1,-1), para_Q[3:],
                normalization_factor_Q_merger, SFR_merger,
                delay_cut, delay_index
            ) * para_Q[5]
        ).mean(1)
        p_bg = 1/400/background_pdf_factor*1 * para_Q[2]
        pdf_norm = (1 - para_Q[0] - para_Q[5]) + \
            para_Q[0] * tc_weight((age-age_preWD)[early]) + \
            para_Q[5] * tc_weight_merger((age-age_preWD)[early])
        density_array_early = np.log(p_s + p_em + p_bg) - np.log(pdf_norm)
        
        ## late
        p_s = velocity_density_3D_func(
            l[late], b[late], vL[late], vB[late], age[late], 
            para_v, is_vL
        ) * (1 - para_Q[0] - para_Q[2] - para_Q[5])
        p_em = (
            velocity_density_3D_func(
                l[late].reshape(-1, 1), b[late].reshape(-1, 1),
                vL_delayed[late], vB_delayed[late], t0.reshape(1,-1),
                para_v, is_vL
            ) * pdf_tct0(
                (age-age_preWD)[late].reshape(-1, 1) + para_Q[1], t0.reshape(1,-1), None,
                normalization_factor_Q, SFR, delay_cut, delay_index
            ) * para_Q[0]
            +
            velocity_density_3D_func(
                l[late].reshape(-1, 1), b[late].reshape(-1,1),
                vL_delayed[late], vB_delayed[late], t0.reshape(1, -1),
                para_v, is_vL
            ) * pdf_tct0(
                (age-age_preWD)[late].reshape(-1, 1), t0.reshape(1,-1), None,
                normalization_factor_Q_merger, SFR_merger,
                delay_cut, delay_index
            ) * para_Q[5]
        ).mean(1)
        p_bg = 1/400/background_pdf_factor * para_Q[2]
        pdf_norm = (1 - para_Q[0] - para_Q[5]) + \
            para_Q[0] * tc_weight((age-age_preWD)[late] + para_Q[1]) + \
            para_Q[5] * tc_weight_merger((age-age_preWD)[late])
        density_array_late = np.log(p_s + p_em + p_bg) - np.log(pdf_norm)
        # 0: index of WD (tc) ,  1: t0,  2: delay on branch 
        ## Q
        a1, a2 = vL_Q_delayed.shape
        p_s = velocity_density_3D_func(
            l_Q, b_Q, vL_Q, vB_Q, age_Q, para_v, is_vL
        ) * (1 - para_Q[0] - para_Q[2] - para_Q[5])
        p_e = (
            velocity_density_3D_func(
                l_Q.reshape(-1, 1), b_Q.reshape(-1, 1),
                vL_Q_delayed, vB_Q_delayed, t0.reshape(1, -1),
                para_v, is_vL
            ).reshape(a1, a2, 1) * pdf_tct0(
                (age_Q-age_preWD_Q).reshape(-1,1,1) + 
                np.linspace(0, 1, n_Q_delay).reshape(1,1,-1) * para_Q[1],
                t0.reshape(1,-1,1), None,
                normalization_factor_Q, SFR, delay_cut, delay_index
            )
        ).mean((1,2)) * (para_Q[1] + t_gap_eff) / t_gap_eff * para_Q[0]
        p_m = (
            velocity_density_3D_func(
                l_Q.reshape(-1, 1),b_Q.reshape(-1, 1),
                vL_Q_delayed, vB_Q_delayed, t0.reshape(1, -1), para_v, is_vL
            ) * pdf_tct0(
                (age_Q-age_preWD_Q).reshape(-1, 1), t0.reshape(1,-1), None,
                normalization_factor_Q_merger, SFR_merger,
                delay_cut, delay_index
            )
        ).mean(1) * para_Q[5]
        p_bg = 1/400/background_pdf_factor * para_Q[2]
        pdf_norm = (1 - para_Q[0] - para_Q[5]) + \
            para_Q[0] * tc_weight(
                (age_Q-age_preWD_Q).reshape(-1, 1) + 
                np.linspace(0, 1, n_Q_delay).reshape(1, -1) * para_Q[1]
            ).mean(1) * (para_Q[1] + t_gap_eff) / t_gap_eff + \
            para_Q[5] * tc_weight_merger(age_Q-age_preWD_Q)
        density_array_Q = np.log(p_s + p_e + p_m + p_bg) - np.log(pdf_norm)
        
    if Q_IS_MERGER == False:
        ## early
        fQ = para_Q[0]
        p_se = velocity_density_3D_func(
            l[early], b[early], vL[early], vB[early], age[early],
            para_v, is_vL
        ) * (1 - para_Q[2] - para_Q[5] - para_Q[0] + fQ)
        p_m = (
            velocity_density_3D_func(
                l[early].reshape(-1, 1),b[early].reshape(-1, 1),
                vL_delayed[early], vB_delayed[early], t0.reshape(1,-1),
                para_v, is_vL
            ) * pdf_tct0(
                age[early].reshape(-1, 1), t0.reshape(1,-1), None,
                normalization_factor_Q_merger, SFR_merger,
                delay_cut, delay_index
            ) * para_Q[5]
        ).mean(1)
        p_bg = 1/400/background_pdf_factor * para_Q[2]
        pdf_norm = (1 - para_Q[5] - para_Q[0]) + fQ + \
            para_Q[5] * tc_weight_merger(age[early])
        density_array_early = np.log(p_se + p_m + p_bg) - np.log(pdf_norm)
        
        ## late
        v_drift_late    = velocity_scatter_3D(
            age + para[1], para_v[0], para_v[1], para_v[2], para_v[6]
        )**2 / stromberg_k
        vL_late, vB_late= np.array(get_v_delayed_3D(age + para[1], l, b, 
                                                    pml, pmb, factor, 
                                                    v_drift_late, U, V, W)
                                   )
        vL_late     = vL_late[late]
        vB_late     = vB_late[late]
        too_late    = age[late] + para_Q[1] > end_of_SF
        fQ_late     = fQ * (~too_late)
        p_s = velocity_density_3D_func(
            l[late], b[late], vL[late], vB[late], age[late], para_v, is_vL
        ) * (1 - para_Q[0] - para_Q[2] - para_Q[5])
        p_e = velocity_density_3D_func(
            l[late], b[late], vL_late, vB_late, age[late]+para_Q[1],
            para_v, is_vL
        ) * fQ_late
        p_m = (
            velocity_density_3D_func(
                l[late].reshape(-1, 1),b[late].reshape(-1, 1),
                vL_delayed[late], vB_delayed[late], t0.reshape(1, -1),
                para_v, is_vL
            ) * pdf_tct0(
                age[late].reshape(-1, 1), t0.reshape(1, -1), None,
                normalization_factor_Q_merger, SFR_merger,
                delay_cut, delay_index
            ) * para_Q[5]
        ).mean(1)
        p_bg = 1/400/background_pdf_factor * para_Q[2]
        pdf_norm = (1 - para_Q[5] - para_Q[0]) + fQ_late + \
            para_Q[5] * tc_weight_merger(age[late])
        density_array_late = np.log(p_s + p_e + p_m + p_bg) - np.log(pdf_norm)

        # 0: index of WD (tc) ,  1: t0,  2: delay on branch 
        ## Q
        n_Q_delay = 50
        v_drift_Q_Qdelayed              = velocity_scatter_3D(
            age_Q.reshape(-1,1) + np.linspace(0,1,n_Q_delay).reshape(1, -1) * para_Q[1],
            para_v[0], para_v[1], para_v[2], para_v[6]
        )**2 / stromberg_k
        vL_Q_Qdelayed, vB_Q_Qdelayed    = get_v_delayed_3D(
            0, l_Q.reshape(-1, 1), b_Q.reshape(-1, 1),
            pml_Q.reshape(-1, 1), pmb_Q.reshape(-1, 1), factor_Q.reshape(-1,1),
            v_drift_Q_Qdelayed, U, V, W
        )
#        vL_Q_Qdelayed, vB_Q_Qdelayed    = get_v_delayed_3D(
#            0, l_Q.reshape(-1, 1), b_Q.reshape(-1, 1),
#            pml_Q.reshape(-1, 1), pmb_Q.reshape(-1, 1), factor_Q.reshape(-1,1),
#            0, U_Q, V_Q, W_Q
#        ) 
        too_late = age_Q.reshape(-1, 1) + np.linspace(0, 1, n_Q_delay).reshape(1, -1) * para_Q[1] > end_of_SF
        p_s = velocity_density_3D_func(
            l_Q, b_Q, vL_Q, vB_Q, age_Q, para_v, is_vL
        ) * (1 - para_Q[0] - para_Q[2] - para_Q[5])
        p_e = (
            velocity_density_3D_func(
                l_Q.reshape(-1, 1), b_Q.reshape(-1, 1), vL_Q_Qdelayed, vB_Q_Qdelayed, 
                age_Q.reshape(-1,1) + np.linspace(0, 1, n_Q_delay).reshape(1, -1) * para_Q[1],
                para_v, is_vL
            ) * ~(too_late)
        ).mean(1) * (para_Q[1] + t_gap_eff) / t_gap_eff * fQ
#        p_e = velocity_density_3D_func(
#            l_Q, b_Q, vL_Q_Qdelayed.flatten(), vB_Q_Qdelayed.flatten(), 
#            10.5, para_v, is_vL
#        ) * (para_Q[1] + t_gap_eff) / t_gap_eff * fQ
        p_m = (
            velocity_density_3D_func(
                l_Q.reshape(-1, 1),b_Q.reshape(-1, 1),
                vL_Q_delayed, vB_Q_delayed, t0.reshape(1, -1), para_v, is_vL
            ) * pdf_tct0(
                age_Q.reshape(-1, 1), t0.reshape(1,-1), None,
                normalization_factor_Q_merger, SFR_merger, 
                delay_cut, delay_index
            )
        ).mean(1) * para_Q[5]
        p_bg = 1/400/background_pdf_factor * para_Q[2]
        pdf_norm = (1 - para_Q[0] - para_Q[5]) + \
            fQ * (~too_late).mean(1) * (para_Q[1] + t_gap_eff) / t_gap_eff + \
            para_Q[5] * tc_weight_merger(age_Q)
#        pdf_norm = (1 - para_Q[0] - para_Q[5]) + \
#            fQ * (para_Q[1] + t_gap_eff) / t_gap_eff + \
#            para_Q[5] * tc_weight_merger(age_Q)
        density_array_Q = np.log(p_s + p_e + p_m + p_bg) - np.log(pdf_norm)
        
    #density_array_Q = 0
    density = density_array_early[~np.isnan(density_array_early)].sum() +\
              density_array_late[~np.isnan(density_array_late)].sum() + \
              density_array_Q[~np.isnan(density_array_Q)].sum()

    if np.isnan(density) or density == 0:
        return -np.inf, density_array_early, density_array_late, density_array_Q
    else:
        return density, density_array_early, density_array_late, density_array_Q


def ln_prob(para, 
            mass, age, pml, pmb, factor, l, b,
            mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
            not_fit_UVW, not_fit_index, fixv):
    # "power index", "v4", "v_T",
    # "index_z","v4_z",
    # "sy/sx",
    # "v0","v0_z","v_T_z",
    # "sy/sx_T"
    # UVW
    #"fraction", "delay", "background", "?SFR_step?", "no use", "merger fraction"
    para_v = para[0:Nv].copy()
    para_Q = para[Nv:Nv+NQ].copy()
    def within(x, lower_lim, upper_lim):
        return (x >= lower_lim) * (x <= upper_lim)
    
    if not (within(para_v[0], 0, 1) * within(para_v[1], 0, 50) * within(para_v[2], 40, 90) *
         within(para_v[3], 0, 1) * within(para_v[4], 0, 50) *
         within(para_v[5], 0, 1) *
         within(para_v[6], 0, 20) * within(para_v[7], 0, 20) * within(para_v[8], 0, 80) *
         within(para_v[9], 0, 1) *
         within(para_v[10], -5, 20) * within(para_v[11], -5, 20) * within(para_v[12], -5, 20) *
         within(para_Q[0], 0, 0.35) * within(para_Q[1], 0, delay_lim) * within(para_Q[2], 0, 0.05) *
         within(para_Q[3], -100, 100) * within(para_Q[4], -100, 100) * within(para_Q[5], 0, 0.5) *
         within(para_Q[6], -100, 100)):
        return -np.inf
    else:
        density, _, _, _ = ln_likelihood_pheno(
            para,
            mass, age, pml, pmb, factor, l, b,
            mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
            False, not_fit_UVW=not_fit_UVW, not_fit_index=not_fit_index, fixv=fixv
        )
        return density


#------------------------------------------------------------------------------
# the main function for calculating likelihood (only with merger component)
#------------------------------------------------------------------------------


def ln_likelihood_pheno_merger_rate(
    para,
    mass, age, pml, pmb, factor, l, b, mass_f, age_f, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
    is_check, velocity_density_3D_check_func=None, is_vL=True,
    not_fit_UVW=False,not_fit_index=False,fixv=False
):
    para_v      = para[0:3].copy()
    para_v      = np.concatenate((np.empty(10),para_v))
    para_Q      = np.empty(NQ)
    para_Q[5]   = para[3]
    delay_index = DELAY_INDEX#para_Q[3]#
    delay_cut   = DELAY_CUT
    if not_fit_UVW == True:
        U = 11
        V = 7.5
        W = 7
    else:
        U = para_v[10]
        V = para_v[11]
        W = para_v[12]
    if not_fit_index == True:
        para_v[0] = p_list[0]#0.47#0.42#0.35#0.41#0.45
        para_v[3] = p_list[1]#0.73#0.67#0.60#0.66#0.62
    if fixv==True:
        para_v[1] = p_list[2]#26#24#26#27#
        para_v[2] = p_list[3]#72#67#66#70#
        para_v[4] = p_list[4]#13#12#12#13#
        para_v[5] = p_list[5]#0.68
        para_v[6] = p_list[6]#8.0#8.8#5.9#6.1#
        para_v[7] = p_list[7]#3.7#4.0#3.8#3.6#
        para_v[8] = p_list[8]#37#33#34#37#
        para_v[9] = p_list[9]#0.63#0.65#0.645#0.65#
    if test_number == 'nomerger':
        para_Q[5] = 0
    para_Q[2] = 0
    para_Q[0] = 0
    early   = np.array(age < 10)
    
    age_preWD = WD_models.MS_age(mass_f)*1e-9
    ## tct0_array is the pdf of tc, t0, range: tc: 0~age_lim+delay, t0: 0~end_of_SFR
    ## normalization_factor_Q is the mean of tct0_array
    tct0_array, normalization_factor_Q = pdf_tct0_array(
        np.linspace(0.,end_of_SF_tc,n_tc).reshape(-1,1),
        np.linspace(0,end_of_SF,n).reshape(1,-1),
        None, SFR, delay_cut, delay_index
    )
    ## tc_weight is the norm factor of pdf of t0 given a tc. tct0_array / tc_weight
    tc_weight = tct0_array.mean(1) / normalization_factor_Q
    tc_weight = interp1d(np.linspace(0.,end_of_SF_tc,n_tc),
                         tc_weight, fill_value=0.0, bounds_error=False)
    
    ## For the ordinary merger population:
    tct0_array_merger               = tct0_array
    normalization_factor_Q_merger   = normalization_factor_Q
    tc_weight_merger                = tc_weight

    ## v_delay for different possible t0
    v_drift_delayed = velocity_scatter_3D(t0, para_v[0], para_v[1], para_v[2], para_v[6])**2 / stromberg_k

    vL_delayed, vB_delayed      = get_v_delayed_3D(
        t0.reshape(1, -1), l.reshape(-1, 1), b.reshape(-1, 1),
        pml.reshape(-1, 1), pmb.reshape(-1,1), factor.reshape(-1, 1),
        v_drift_delayed.reshape(1, -1), U, V, W
    )
    v_drift     = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])**2 / stromberg_k
    vL, vB      = np.array(get_v_delayed_3D(age, l, b, pml, pmb,
                                            factor, v_drift, U, V, W)
                          )
    if is_check == True:
        velocity_density_3D_func    = velocity_density_3D_check_func
        background_pdf_factor       = 1    
    else:
        velocity_density_3D_func    = velocity_density_3D
        background_pdf_factor       = 400
    ## pdf of v,tc =
        ## [pdf(v) for ordinary * uniform(tc)] * ordinary fraction +
        ## [pdf(v) for merger * merger fraction(t0)
    ## pdf_tct0 is (age_lim+delay) * end_of_SF * pdf(tc,t0)
    
    if Q_IS_MERGER == False or Q_IS_MERGER == True:
        ## early
        fQ = 0
        SFR_normalization = SFR(np.linspace(0.,end_of_SF_tc,n_tc),[]).mean()
        p_se = velocity_density_3D_func(
            l[early], b[early], vL[early], vB[early], age[early],
            para_v, is_vL
        ) * (1 - para_Q[5]) * SFR(age,[])[early]/SFR_normalization
        p_m = (
            velocity_density_3D_func(
                l[early].reshape(-1, 1),b[early].reshape(-1, 1),
                vL_delayed[early], vB_delayed[early], t0.reshape(1,-1),
                para_v, is_vL
            ) * pdf_tct0(
                (age_f-age_preWD)[early].reshape(-1, 1), t0.reshape(1,-1), None,
                normalization_factor_Q_merger, SFR_merger,
                delay_cut, delay_index
            ) * para_Q[5]
        ).mean(1)
        p_bg = 1/400/background_pdf_factor * para_Q[2]
        pdf_norm = (1 - para_Q[5]) * SFR(age,[])[early]/SFR_normalization + \
            para_Q[5] * tc_weight_merger((age_f-age_preWD)[early])
        density_array_early = np.log(p_se + p_m + p_bg) - np.log(pdf_norm)

    #density_array_Q = 0
    density = density_array_early[~np.isnan(density_array_early)].sum()

    if np.isnan(density) or density == 0:
        return -np.inf, density_array_early
    else:
        return density, density_array_early

def ln_prob_merger_rate(para, 
                        mass, age, pml, pmb, factor, l, b,
                        mass_f, age_f, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                        not_fit_UVW, not_fit_index, fixv):
    # "power index", "v4", "v_T",
    #"index_z","v4_z",
    #"sy/sx",
    #"v0","v0_z","v_T_z",
    #"sy/sx_T"
    #UVW
    #"fraction", "delay", "background", 
    #"U_W", "V_W", "merger fraction", 'W_W'
    #para_v = para[0:3].copy()
    #para_Q = para[3:3+NQ].copy()

    def within(x, lower_lim, upper_lim):
        return (x >= lower_lim) * (x <= upper_lim)

    if not (within(para[0], -5, 20) * within(para[1], -5, 20) * within(para[2], -5, 20) *
         within(para[3], 0, 0.8)
        ):
        return -np.inf
    else:
        density, _ = ln_likelihood_pheno_merger_rate(
            para,
            mass, age, pml, pmb, factor, l, b,
            mass_f, age_f, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
            False, None, True,
            not_fit_UVW=not_fit_UVW, not_fit_index=not_fit_index, fixv=fixv
        )
        return density