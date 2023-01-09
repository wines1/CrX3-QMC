#! /usr/bin/env python

from nexus import settings,job,run_project,obj
from nexus import generate_physical_system
from nexus import generate_pwscf, generate_pw2qmcpack
from nexus import generate_qmcpack,vmc,loop,linear,dmc
from nexus import read_structure
from numpy import mod, sqrt
from qmcpack_input import spindensity
import numpy as np
from structure import optimal_tilematrix
from numpy.linalg import det
settings(
    results = './results',
    pseudo_dir = './pseudopotentials',
    sleep   = 1,
    runs    = 'cri3/u-2',
    machine = 'kisir',
    )

twod_prim_12 = read_structure('POSCAR-2d-12A', format='poscar')

boundaries = 'ppp'
supercells = [[[2, 1, 0], [-2, 3, 0], [0, 0, 1]]]


linopt1 = linear(
    energy               = 0.0,
    unreweightedvariance = 1.0,
    reweightedvariance   = 0.0,
    timestep             = 2.0,
    samples              = 160000,
    blocks               = 100,
    warmupsteps          = 200,
    steps                = 1,
    stepsbetweensamples  = 1,
    substeps             = 4,
    nonlocalpp           = True,
    usebuffer            = True,
    minwalkers           = 0.1,
    usedrift             = True,
    minmethod            = 'OneShiftOnly',
    shift_i              = 0.01,
    shift_s              = 1.0
    )

j3 = False
#if you want to run VMC (if false you run DMC)
run_vmc = False

# you want to run bulk
run_bulk = False
always_bulk = False # Always use bulk cell to optimize jastrows
bulk_prim = None
bulk_prim_temp = None
##

#DMC options
dmc_eqblocks = 200
dmcblocks    = 300
dt_dmc       = 0.01
tmoves       = False
##
if run_bulk:
    bulk_prim = read_structure('POSCAR-bulk', format='poscar')
shared_qe = obj(
    occupations = 'smearing',
    smearing    = 'gaussian',
    degauss     = 0.005,
    input_dft   = 'pbe',
    conv_thr    = 1.0e-7,
    mixing_beta = 0.1,
    ecut = 300,
    nosym       = True,
    use_folded  = True,
)
qe_presub = 'module load intel impi openmpi-3.0.1/intel'
qmcpack_presub = 'module load intel openmpi-3.0.1/intel qmcpack-3.9.0'

qe_job = job(nodes=4,threads=40,app='pw.x', presub=qe_presub)
qmc_job = job(nodes=8,threads=40,app='qmcpack_kisir_cpu_comp_SoA', presub=qmcpack_presub)
vmc_dmc_dep = None
for scale in [1.0]:
    twod_prim_12_temp = twod_prim_12.copy()
    twod_prim_12_temp.stretch(scale, scale , 1.0)
    twod_12_system = generate_physical_system(
        structure = twod_prim_12_temp,
        Cr = 14,
        I = 7,
        kshift   = (0,0,0),
        net_spin = 6,
    )
    twod_12_scf = generate_pwscf(
        identifier = 'scf',                      # log output goes to scf.out
        input_type  = 'scf',
        path       = 'scf-twod-12-{}'.format(scale),      # directory to run in
        job        = qe_job,# pyscf must run w/o mpi
        system     = twod_12_system,
        pseudos    = ['Cr.opt.upf','I.ccECP.AREP.upf'],
        hubbard_u   = obj(Cr=2,I=0),
        kgrid      = (3,3,1),
        wf_collect = True,
        **shared_qe
    )
    
    for supercell in supercells:
        scell_vol = det(supercell)
        nscf_kgrid_k = int(np.ceil(3/sqrt(scell_vol)))
        nscf_grid = (nscf_kgrid_k, nscf_kgrid_k, 1)
        twod_nscf_12_system = generate_physical_system(
            structure = twod_prim_12_temp,
            Cr = 14,
            I = 7,
            tiling = supercell,
            kgrid  = nscf_grid,
            kshift = (0,0,0),
            net_spin = 6,
        )
        twod_nscf_12 = generate_pwscf(
            identifier  = 'nscf',
            path        = 'nscf-twod-20-{}-{}'.format(scale,scell_vol),
            job         = qe_job ,# pyscf must run w/o mpi
            system      = twod_nscf_12_system,
            input_type  = 'nscf',
            pseudos    = ['Cr.opt.upf','I.ccECP.AREP.upf'],
            hubbard_u   = obj(Cr=2,I=0),
            wf_collect   = True,
            dependencies = (twod_12_scf,'charge_density'),
            **shared_qe
            )

        twod_p2q_12 = generate_pw2qmcpack(
            identifier   = 'p2q',
            path         = 'nscf-twod-20-{}-{}'.format(scale,scell_vol),
            job          = job(cores= 1,threads= 1,app='pw2qmcpack.x',presub=qe_presub),
            write_psir   = False,
            dependencies = (twod_nscf_12,'orbitals'),
            )
        

        qmc_pseudos = ['Cr.opt.xml','I.ccECP.AREP.xml']
        
        if scale==1.00:
            twod_super = twod_prim_12_temp.tile(supercell)
            rcut_bulk = 1000
            if run_bulk:
                bulk_super = bulk_prim_temp.tile(bulk_scell_mat.tolist())
                rcut_bulk = bulk_super.rwigner()
                
            rcut = 0
            opt_system = None
            opt_orbitals = None
            opt_name = ''
            
            if (not always_bulk and (twod_super.rwigner() < rcut_bulk)):
                rcut = twod_super.rwigner()
                opt_system = twod_nscf_12_system
                opt_orbitals = twod_p2q_12
                opt_name = 'twod'
                rcut = twod_super.rwigner() - 0.001;
                print("For supercell factor {} using 2d with Rwigner {:.6f}, not Bulk with Rwigner {:.6f}".format(scell_vol, twod_super.rwigner(), rcut_bulk ))
            else:
                rcut = bulk_super.rwigner
                opt_system = bulk_nscf_system
                opt_orbitals = bulk_p2q
                opt_name = 'bulk'
                rcut = bulk_super.rwigner() - 0.001;
                print("For supercell factor {} using Bulk with Rwigner {:.6f}, not 2d with Rwigner {:.6f}".format(scell_vol, bulk_super.rwigner(), twod_super.rwigner() ))
            #end if
            
            if always_bulk:
                rcut = min(twod_super.rwigner(), bulk_super.rwigner()) - 0.001
                print("Always optimize Bulk rcut is updated to be reused in 2d {:.6f}".format(rcut))
            #end if

            linopt1t = linopt1.copy()
            linopt1t.samples *= sqrt(scell_vol)
            
            linopt2 = linopt1t.copy()
            linopt2.energy = 0.95
            linopt2.unreweightedvariance = 0.05
            linopt2.samples *= 2 
            linopt2.minwalkers = 0.5

            linopt3 = linopt2.copy()
            linopt3.samples *= 2

            opt_J2 = generate_qmcpack(
                #block           = True, #Comment if you want to block rest of the calculations
                identifier      = 'opt',
                path            = 'optJ2-{}-{}-{}'.format(opt_name, scale, scell_vol),
                job             = qmc_job,
                input_type      = 'basic',
                bconds          = boundaries,
                system          = opt_system,
                pseudos         = qmc_pseudos,
                lr_handler      = 'ewald', 
                lr_dim_cutoff   = 30,
                twistnum        = 1,
                #jastrows       = 'generate12',
                jastrows        = [('J1', 'bspline', 8, rcut),
                                  ('J2', 'bspline', 8, rcut)],
                #estimators = [spindensity(grid=grid_density)],
                corrections = ['mpc','chiesa'],
                spin_polarized = True,
                calculations =  [loop(max=42,qmc=linopt1t), loop(max=42, qmc=linopt2)],
                dependencies    = (opt_orbitals,'orbitals'),
                )
            vmc_dmc_dep = opt_J2
        if run_vmc:
            vmc_path = 'vmc-J2-twod-3-{}'.format(scell_vol)
            if j3:
                vmc_path = 'vmc-J3-twod-3-{}'.format(scell_vol)
            #end if
            vmcrun = generate_qmcpack(
                    identifier      = 'vmc',
                    path            = vmc_path,
                    job             = qmc_job,
                    input_type      = 'basic',
                    bconds          = boundaries,
                    system          = twod_nscf_3_system,
                    pseudos         = qmc_pseudos,
                    corrections     = [],
                    jastrows        = [],
                    lr_handler      = 'ewald',
                    lr_dim_cutoff   = 30,
                    calculations    = [
                        vmc(
                            warmupsteps         = 100,
                            samples             = 256000,
                            blocks              = 160,
                            steps               = 10,
                            stepsbetweensamples = 1,
                            walkers             = 1,
                            timestep            = 0.3,
                            substeps            = 4
                        )
                    ],
                    dependencies    = [(twod_p2q_12,'orbitals'), (vmc_dmc_dep, 'jastrow')]
                )
        
        else:
            #run DMC 
               nkgrid = len(twod_nscf_12_system.structure.kpoints)
               dmc_nnodes = max(8, nkgrid)
               dmc_job = job(nodes=dmc_nnodes,threads=40,app='qmcpack_kisir_cpu_comp_SoA', presub=qmcpack_presub)
               dmcrun = generate_qmcpack(
                    identifier      = 'dmc',
                    path            = 'dmc-twod-12-{}-{}-{}'.format(scale, scell_vol, dt_dmc),
                    job             = dmc_job,
                    input_type      = 'basic',
                    bconds          = boundaries,
                    system          = twod_nscf_12_system,
                    pseudos         = qmc_pseudos,
                    corrections     = [],
                    jastrows        = [],
                    lr_handler      = 'ewald',
                    lr_dim_cutoff   = 30,
                    calculations = [vmc(warmupsteps         = 100,
                                        blocks              = 800,
                                        steps               = 1,
                                        stepsbetweensamples = 1,
                                        walkers             = 1,
                                        timestep            = 0.3,
                                        substeps            = 4,
                                        samplesperthread    = 150
                                        ),
                                    dmc(warmupsteps   = dmc_eqblocks,
                                        blocks        = 240,
                                        steps         = 10,
                                        timestep      = 0.01,
                                        nonlocalmoves = True
                                        ),
                                    ],
                    dependencies    = [(twod_p2q_12,'orbitals'), (vmc_dmc_dep, 'jastrow')]
                )
        #end if
    #end if
run_project() 



    
