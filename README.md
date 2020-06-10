# Qbranch

This is a page associated with this paper: https://ui.adsabs.harvard.edu/abs/2019ApJ...886..100C/abstract, which is about a discovery of white dwarf cooling anomaly with Gaia data. If you have any questions, please do not hesitate to ask me through email: s.cheng@jhu.edu

In this folder:

1. WD_Qbranch_paper.ipynb is the notebook to make Figure 1-4, 8-10 in the paper. You may also need to download two additional files https://pages.jh.edu/~scheng40/Qbranch/WD_MWDD.npy and https://pages.jh.edu/~scheng40/Qbranch/WD_warwick.csv. Also, you need to install this python package: https://github.com/SihaoCheng/WD_models, which we put a lot of efforts into and should be handy to use ^_^

2. WD_view_MCMC.ipynb is the notebook to make Figure 5-7, 11 in the paper. They are related to MCMC samplings of the posterior of our model parameters, including the delay time, population fractions, and some milkyway kinematic parameters. 

3. WD_MCMC_Qbranch.py conducts the MCMC sampling.

4. WD_highmass_paper.csv, WD_early_paper.csv, WD_Q_paper.csv, WD_late_paper.csv are tables of highmass white dwarfs within 250 pc selected in this paper from Gaia DR2.

5. columns.txt explains the columns in the above four tables.

6. WD_HR.py and WD_MCMC_func.py provide some functions used in the aforementioned notebooks and python script.
