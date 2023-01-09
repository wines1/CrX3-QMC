# CrX3-QMC
Diffusion Monte Carlo data and relevant scripts for manuscript titled: "Systematic DFT+U and Quantum Monte Carlo Benchmark of Magnetic Two-Dimensional (2D) CrX3 (X = I, Br, Cl, F)" by Daniel Wines, Kamal Choudhary and Francesca Tavazza, Journal of Physical Chemistry C, 2023, https://doi.org/10.1021/acs.jpcc.2c06733


Each zip contains DMC data associated with 2D CrI3 and CrB3 used to calculate a bound on the magnetic exchange (J). This DMC data can be analyzed using the qmca tool https://qmcpack.readthedocs.io/en/develop/analyzing.html. In addition, we have provided a pseudopotential directory and an example Nexus (https://nexus-workflows.readthedocs.io/en/latest/) workflow script (cri3-u.py) that facilitates Density Functional Theory, Variational Monte Carlo and Diffusion Monte Carlo calculations. A script that determine the optimal supercell size/shape based on lattice geometry and number of atoms (tile.py), a script which uses JARVIS to facilitate DFT (VASP) calculations to determine the magnetic exchange and anisotropy (vasp_workflow_Tc_jarvis.py) and a post processing script to determine the 2D Curie Temperature using the method of Torelli and Olsen (Tc-2d.py) are also included. Questions regarding the data and scipts can be directed towards corresponding author Daniel Wines: daniel.wines@nist.gov
