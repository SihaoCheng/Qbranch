---
layout: default
---

# A Cooling Anamaly of White Dwarfs
Most white dwarf stars shine at the cost of cooling, but not all. Using the wonderful Gaia DR2 data, I discovered some weired high-mass white dwarfs stopping their cooling when they reach a specific mass-dependent temperature. This slowing-down or "stop" of cooling produces a "trafic jam" on the white dwarf cooling tracks, and was named the "Q branch" on the Hertzsprung--Russell diagram. The corresponding temperature of the Q branch is around the freezing temperature of the Coulomb plasma in white dwarf cores. Using kinematics as an indicator of stellar true age, We demonstrated that the "stop" of cooling is several billion years long, and canno be created by crystallization alone. We also show that another mechanism, i.e., the settling of the neutron-rich 22Ne, can possibly produce this cooling anomaly. It seems a conspiracy of nature that the effect of 22Ne settling is peaked around the white dwarf freezing temperature! Here are my papers for the [discovery](https://ui.adsabs.harvard.edu/abs/2019ApJ...886..100C/abstract) and [theory](https://www.nature.com/articles/s41586-024-07102-y). Here are the catalog of high-mass white dwarfs used in the study
<br>
[WD_early.csv](https://pages.jh.edu/~scheng40/Qbranch/WD_early.csv)
<br>
[WD_Q.csv](https://pages.jh.edu/~scheng40/Qbranch/WD_Q.csv)
<br>
[WD_late.csv](https://pages.jh.edu/~scheng40/Qbranch/WD_late.csv)
<br>
and an explanation of the columns [column.txt](https://pages.jh.edu/~scheng40/Qbranch/columns.txt)

## Observations from Gaia
The Q branch is revealed by the Hertzsprung--Russell diagram (a luminosity vs. color plot) of white dwarf stars, obtained from Gaia Data Release 2 (DR2). It is not only a pile-up or overdensity, but also a region with more fast-moving white dwarfs. These fast white dwarfs should be much older than their predicted cooling age, according to the age--velocity-dispersion relation of the Milky-Way disk.
<br>
<img src="https://pages.jh.edu/~scheng40/Qbranch/images/Qbranch.png" width="500" />


## Two populations of white dwarfs
To explain both the overdensity and the velocity excess on the Q branch, there must be more than one population of white dwarfs with different cooilng behaviors. The simplest scenario is a two population scenario, with one normal cooling population and another population of white dwarfs subject to an extra cooling delay around the branch, as shown below in the animations. Our quantitative analysis shows that the fraction of the extra-delayed population is about 6% among high-mass white dwarfs, and the delay is 8 Gyr long. This poses a challenge to white dwarf models.
<br>
<img src="https://pages.jh.edu/~scheng40/Qbranch/images/gif_two_pops.gif" width="500" />


## Normal white dwarf cooling
White dwarfs cool down from the bright, blue end to the faint, red end of their cooling tracks. The cooling rate changes quite smoothly, so the number density of white dwarfs of the normal-cooling population is also smooth on the H--R diagram.
<br>
<img src="https://pages.jh.edu/~scheng40/Qbranch/images/gif_normal_pop.gif" width="500" />


## Extra-delayed cooling
A fraction of white dwarfs are subject to an extra cooling delay that makes them pile-up on the Q branch and accumulate age discrepancy.
<br>
<img src="https://pages.jh.edu/~scheng40/Qbranch/images/gif_extra_delayed_pop.gif" width="500" />
<br>
Illustration of how this simple two-population, two-parameter model can recover both the overdensity and the velocity (and age) excess.
<br>
<img src="https://pages.jh.edu/~scheng40/Qbranch/images/gif_both_pops.gif" width="500" />
<br>
<img src="https://pages.jh.edu/~scheng40/Qbranch/images/gif_age.gif" width="500" />

## The physics behind this cooling anomaly
It is natural to ask for the physics behind this selective, powerful, and highly-peaked extra cooling delay. In my discovery paper I pointed out that gravitational energy from 22Ne settling is likely the energy source.  In a recent [Nature paper](https://www.nature.com/articles/s41586-024-07102-y), we showed that the settling is actually triggered by 22Ne distillation during crystallization, as illustrated below: in the traditional picture, crystals are heaviers and thus form a solid core; in contrast, when 22Ne is more than about 2%, the crystals become lighter because the solid phase has less neutron-rich 22Ne. In this case, the crystals will float up and then melt. This micro convection efficiently transports C and O upwards and 22Ne downwards, releasing exactly the amount of gravitational energy required to make the star shine.
<br>
<img src="https://github.com/SihaoCheng/Qbranch/blob/master/distillation.png?raw=true" width="500" />



<h1 id="Contacts">Contacts</h1>
scheng@ias.edu
<br>
+1 443 207 1532
<br>
Institute for Advanced Study
<br>
Princeton, NJ 08540, USA
