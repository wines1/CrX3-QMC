#! /usr/bin/env python
from nexus import settings,job,run_project
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack
from nexus import generate_qmcpack,vmc
from nexus import read_structure
from structure import *
prim = read_structure('POSCAR')
opt_matrix, opt_ws = optimal_tilematrix(axes = prim.axes, volfac=8)
print(opt_ws)
print(opt_matrix)
r_ws = prim.rwigner()
print(r_ws)
