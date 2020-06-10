import sys

import emcee
import matplotlib.pyplot as plt
# from multiprocessing import Pool
import numpy as np

from astropy.io import ascii
from astropy.table import Table
import astropy.units as u

import WD_HR
import WD_MCMC_func
import WD_models


test_number = sys.argv[2]
WD_MCMC_func.test_number = test_number

if sys.argv[3] == 'M':
    METHOD = 'Run_MCMC'
    WD_MCMC_func.t_gap_eff = 0.743
if sys.argv[3] == 'S':
    METHOD = 'Simulate'
    WD_MCMC_func.t_gap_eff = 0.505
if sys.argv[4] == 'T':
    NOT_FIT_UVW= True
if sys.argv[4] == 'F':
    NOT_FIT_UVW = False
if sys.argv[5] == 'T':
    NOT_FIT_INDEX = True
if sys.argv[5] == 'F':
    NOT_FIT_INDEX = False
if sys.argv[6] == 'T':
    FIXV = True
if sys.argv[6] == 'F':
    FIXV = False
if sys.argv[7] == 'T':
    WD_MCMC_func.Q_IS_MERGER = True
if sys.argv[7] == 'F':
    WD_MCMC_func.Q_IS_MERGER = False
    
WD_MCMC_func.DELAY_INDEX = -float(sys.argv[8])    
WD_MCMC_func.DELAY_CUT = float(sys.argv[9])

if len(sys.argv) > 10:
    DELAY = int(sys.argv[10])

##--------------------------------------------------------------------------------------------------
from WD_MCMC_func import Nv, NQ, end_of_SF, age_T, DELAY_INDEX, DELAY_CUT, Q_IS_MERGER, stromberg_k

agents      = 1
chunksize   = 1
number      = agents
burning     = 200
then_run    = 1200
gap         = 10
ndim, nwalkers = Nv + NQ, 500


# Load WD table
##------------------------------------------------------------------------------------------------------------------------
SELECTION_PARA = [1.4, 0.10, 2, 22, 8, 300]
WD_warwick_smaller = np.load('/datascope/menard/group/scheng/Gaia/WD_warwick_smaller.npy')[0]['WD_warwick_smaller']
_, WD_warwick_smaller = WD_MCMC_func.select_WD(WD_warwick_smaller,SELECTION_PARA[0],SELECTION_PARA[1],SELECTION_PARA[2],
                                               SELECTION_PARA[3],SELECTION_PARA[4],SELECTION_PARA[5])
if WD_MCMC_func.Q_IS_MERGER == False:
    WD_MCMC_func.n = 400
    WD_MCMC_func.n_tc = 8000


# Select the WDs Suitable for MCMC
##------------------------------------------------------------------------------------------------------------------------
mass_min    = 1.08#1.10
mass_max    = 1.23#1.28
distance1   = 0
distance2   = int(sys.argv[1])
atm_type    = 'H'
model       = 'o'
# WD_model    = WD_models.load_model('f', 'f', model, atm_type)
age_lim     = 3.5
WD_warwick_smaller['mass']  = WD_warwick_smaller['mass_' + atm_type + '_' + model]
WD_warwick_smaller['age']   = WD_warwick_smaller['age_' + atm_type + '_' + model]
Q_branch    = np.array((WD_warwick_smaller['mass'] > mass_min) *
                       (WD_warwick_smaller['mass'] < mass_max) *
                       (1/WD_warwick_smaller['parallax']*1000 > distance1) *
                       (1/WD_warwick_smaller['parallax']*1000 < distance2) *
                       WD_HR.func_select(WD_warwick_smaller['bp_rp'],
                                         WD_warwick_smaller['G'],
                                         13.20, 1.2, 0.20, -0.40, 0.10)
                      )
WD      = WD_warwick_smaller[np.array((WD_warwick_smaller['mass'] > mass_min) *
                                      (WD_warwick_smaller['mass'] < mass_max) *
                                      (1/WD_warwick_smaller['parallax']*1000 > distance1) *
                                      (1/WD_warwick_smaller['parallax']*1000 < distance2) *
                                      ~Q_branch )]
WD_Q    = WD_warwick_smaller[Q_branch]

print('length of WD: ',len(WD), 'length of WD_Q: ',len(WD_Q))


# prepare to get v
pml, pmb, factor        = WD_MCMC_func.prep_get_v(WD)
pml_Q, pmb_Q, factor_Q  = WD_MCMC_func.prep_get_v(WD_Q)
v_drift                 =   (((WD['age'] + 0.1) / 10.1)**0.2 * 40)**2 / stromberg_k
v_drift_Q               = (((WD_Q['age'] + 0.1) / 10.1)**0.2 * 40)**2 / stromberg_k
vL, vB                  = np.array(WD_MCMC_func.get_v_delayed_3D(WD['age'], WD['l'], WD['b'],
                                                                 pml, pmb, factor, v_drift,
                                                                 11, 7.5, 7))
vL_Q, vB_Q              = np.array(WD_MCMC_func.get_v_delayed_3D(WD_Q['age'], WD_Q['l'], WD_Q['b'],
                                                                 pml_Q, pmb_Q, factor_Q, v_drift_Q,
                                                                 11, 7.5, 7))

selection   = np.array((WD['age'] < age_lim) * (WD['age'] > 0.1) * 
                       ((vL**2 + vB**2)**0.5 < 300))
mass        = np.array(WD['mass'][selection])
age         = np.array(WD['age'][selection])
l           = np.array(WD['l'][selection])
b           = np.array(WD['b'][selection])
vL          = vL[selection]
vB          = vB[selection]
pml         = pml[selection]
pmb         = pmb[selection]
factor      = factor[selection]

selection_Q = np.array((WD_Q['age'] < age_lim) * (WD_Q['age'] > 0.1) * 
                       ((vL_Q**2 + vB_Q**2)**0.5 < 300))
mass_Q      = np.array(WD_Q['mass'][selection_Q])
age_Q       = np.array(WD_Q['age'][selection_Q])
l_Q         = np.array(WD_Q['l'][selection_Q])
b_Q         = np.array(WD_Q['b'][selection_Q])
vL_Q        = vL_Q[selection_Q]
vB_Q        = vB_Q[selection_Q]
pml_Q       = pml_Q[selection_Q]
pmb_Q       = pmb_Q[selection_Q]
factor_Q    = factor_Q[selection_Q]

early       = np.array(1.22 - (age - 0.6) * 0.2 > mass) #for ONe: 1.22; CO: 1.25
late        = np.array(1.22 - (age - 0.6) * 0.2 < mass)

print('length of WD: ', selection.sum() + selection_Q.sum(), 
      ' early: ', early.sum(), ' Q: ', selection_Q.sum(), ' late: ', late.sum())


##--------------------------------------------------------------------------------------------------
def parallel(i):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, WD_MCMC_func.ln_prob, threads=100,
                                    args=[mass, age, pml, pmb, factor, l, b,
                                          mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                                          NOT_FIT_UVW, NOT_FIT_INDEX, FIXV])
    a_random_number = np.random.randint(0,100000)
    np.random.seed(i+a_random_number)
    # "power index", "v4", "v_T",
    # "index_z","v4_z",
    # "sy/sx",
    # "v0","v0_z","v_T_z",
    # "sy/sx_T"
    # UVW
    # "fraction", "delay", "background", "?SFR_step?", "no use", "merger fraction", "no use"
    def sampling(width, center, N=1):
        return np.random.rand(N) * width * 2 + center - width
                       
    p0 = [np.concatenate((
        sampling(0.1, 0.3), sampling(10, 30), sampling(10, 65),
        sampling(0.15, 0.5), sampling(10, 15),
        sampling(0.1, 0.67),
        sampling(4, 5), sampling(4, 5), sampling(10, 40),
        sampling(0.1, 0.63),
        np.array([7, 5, 5]) + np.random.rand(3) * np.array([5, 5, 5]),
        np.array([0.02, 8, 0]) + np.random.rand(3) * np.array([0.15, 4, 0.01]),
        np.array([-75, -75, 0.15, -75])+np.random.rand(4)*np.array([150, 150, 0.15, 150]) ))
          for j in range(nwalkers)]
    pos, _, _ = sampler.run_mcmc(p0, burning)
    sampler.reset()
    sampler.run_mcmc(pos, then_run)
    a = sampler.flatchain[::gap,:]
    return a



# Run MCMC
##--------------------------------------------------------------------------------------------------
if METHOD == 'Run_MCMC':
    result = parallel(1)
    sampling_per_agent = nwalkers * (then_run//gap)
    para_v = np.empty((number * sampling_per_agent, Nv))
    para_Q = np.empty((number * sampling_per_agent, NQ))
    for i in range(number):
        para_v[(i*sampling_per_agent):((i+1)*sampling_per_agent)] = result[:, :Nv]
        para_Q[(i*sampling_per_agent):((i+1)*sampling_per_agent)] = result[:, Nv:Nv+NQ]
    para_v = para_v.reshape(agents, nwalkers, then_run//gap, Nv)\
                .transpose((2, 1, 0, 3)).reshape(number * sampling_per_agent, Nv)
    para_Q = para_Q.reshape(agents, nwalkers, then_run//gap, NQ)\
                .transpose((2, 1, 0, 3)).reshape(number * sampling_per_agent, NQ)
    #---------------------------------------------------------------------------------------------------------------------------
    para_input = np.median(np.concatenate((para_v, para_Q), axis=1)[-50000:, :], axis=0)
    
    x_list = ['np.arange(0, 15, 0.2)', 'np.arange(0, 0.45, 0.01)', 'np.arange(0, 0.45, 0.01)']
    PDF_test_name = ['delay_test', 'mfraction', 'Qfraction']
    changed_para = [Nv + 1, Nv + 5 , Nv + 0]
    for PDF_test_index in range(3):
        pdf_sim     = np.empty_like(eval(x_list[PDF_test_index]))
        pdf_Q_sim   = np.empty_like(eval(x_list[PDF_test_index]))
        pdf_e_sim   = np.empty_like(eval(x_list[PDF_test_index]))
        pdf_l_sim   = np.empty_like(eval(x_list[PDF_test_index]))
    
        for i, x in enumerate(eval(x_list[PDF_test_index])):
            para = para_input.copy()
            para[changed_para[PDF_test_index]] = x
            pdf_sim[i], temp1, temp2, temp = WD_MCMC_func.ln_likelihood_pheno(
                para, mass, age, pml, pmb, factor, l, b,
                mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q, False,
                not_fit_UVW=NOT_FIT_UVW, not_fit_index=NOT_FIT_INDEX, fixv=FIXV)
            pdf_e_sim[i]    = temp1[~np.isnan(temp1)].sum()
            pdf_l_sim[i]    = temp2[~np.isnan(temp2)].sum()
            pdf_Q_sim[i]    = temp[~np.isnan(temp)].sum()
        exec( PDF_test_name[PDF_test_index] + '= [pdf_sim, pdf_e_sim, pdf_l_sim, pdf_Q_sim ]')

    #---------------------------------------------------------------------------------------------------------------------------
    if Q_IS_MERGER == True:
        suffix = '.npy'
    if Q_IS_MERGER == False:
        suffix = 'Qisnotmerger.npy'
    
    np.save('/datascope/menard/group/scheng/Gaia/WD_vel_age_MCMC_Feb12/MCMC_power_parallel_' +
            sys.argv[8] + '_' + sys.argv[9] + '_' +
            str(mass_min) + '_' + str(distance2) + '_' + str(age_lim) + '_' +
            atm_type + '_' + model + '_' + str(end_of_SF) + '_T' +
            str(age_T) + '_' + sys.argv[4] + sys.argv[5] + sys.argv[6] + test_number + suffix,
            np.array([{'para_Q':para_Q, 'para_v':para_v,
                       'delay_test':delay_test, 'mfraction':mfraction, 'Qfraction':Qfraction,
                       'data_length':[selection.sum(), selection_Q.sum()],
                       'para_input':para_input,
                       'selection_para':SELECTION_PARA,
                       'delay_index_cut':[DELAY_INDEX, DELAY_CUT]}]) )
    print('all finished')


