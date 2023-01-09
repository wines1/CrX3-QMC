from pymatgen.io.vasp.outputs import Vasprun, Poscar
import math
vrun1 = Vasprun('/users/dtw2/crx3/jarvis-test-correctecut/scan/cri3/Calc-1/Calc-1/vasprun.xml')
vrun2 = Vasprun('/users/dtw2/crx3/jarvis-test-correctecut/scan/cri3/Calc-2/Calc-2/vasprun.xml')
vrun3 = Vasprun('/users/dtw2/crx3/jarvis-test-correctecut/scan/cri3/Calc-3/Calc-3/vasprun.xml')
vrun4 = Vasprun('/users/dtw2/crx3/jarvis-test-correctecut/scan/cri3/Calc-4/Calc-4/vasprun.xml')


#C is #dimensionless constant based on crystal type
C = 1.52 #Honeycomb lattice 
#C = 2.27 Cubic/Kagome 
#C = 3.64 hexagonal/trigonal

k_b = 0.08617 #Boltzmann Constant 

gamma = 0.033 #dimensionless constant 

S = (3/2)
#S = (1/2)
N_nn = 3 #nearest neighbors 
N_tm = 2 #number of transition metal atoms in the cell 
fm_afm_diff =vrun2.final_energy - vrun4.final_energy


fm_001 = vrun2.final_energy
fm_100 = vrun1.final_energy
afm_001 = vrun4.final_energy
afm_100 = vrun3.final_energy


#Calculation of J (meV)

J = -(1000*(fm_001/N_tm - afm_001/N_tm)/N_nn/S**2) 
print('J: {0} meV' .format(float(J)))


#Calculation of D (meV)

D = -(1000*((fm_001/N_tm - fm_100/N_tm) + (afm_001/N_tm - afm_100/N_tm))/2/S**2)
print('D: {0} meV' .format(float(D)))

#Calculation of lambda (meV)

l = -(1000*((fm_001/N_tm - fm_100/N_tm) - (afm_001/N_tm - afm_100/N_tm))/N_nn/S**2)
print('lambda: {0} meV' .format(float(l)))

#Calculation of delta (meV)

d = D*(2*S-1)+l*S*N_nn
#print(d)

#Calculation of function to compute Tc
x_f = d/(J*(2*S-1))
#print(x_f)

f = (math.tanh((6/N_nn)*math.log(1+gamma*x_f)))**(1/4) 
#print(f)

#Calculation of Ising temp

T_ising = S**2*(J*C/k_b)
#print(T_ising)

#Calculation of 2D Critical Temp using Torelli/Olsen method

T_2d = T_ising*f
print('2D Critical temperature: {0} K' .format(float(T_2d)))
