

def initialize():
    """
    Does three things to initialize (any) job:
        1. Delete old error and output files for restarted jobs
        2. Load job input parameters
        3. Write initial Atoms object to init.traj
    """
    import json,os,glob
    import ase.io as aseio
    import ase.parallel as asepar

    def keepNewest(string):                 # Step 1
        listPth = glob.glob(string)
        ordered = sorted([(os.path.getmtime(pth),pth) for pth in listPth])
        for t,pth in ordered[:-1]: 
            os.remove(pth)

    if rank()==0:
        keepNewest('*.error')
        keepNewest('*.out')

    try:   os.remove('result.json')
    except OSError: pass
    try:   os.remove('runtime.json')
    except OSError: pass

    with asepar.paropen('params.json','r') as f: prms  = json.loads(f.read())    # Step 2 
    atoms   = makeAtoms(prms)

    aseio.write('init.traj',atoms)         # Step 3

    return prms,atoms


def makeAtoms(params): 
    import cPickle
    return cPickle.loads(str(params['inittraj_pckl']))


def makeCalc(params):
    dftcode = params['dftcode']
    jobkind = params['jobkind']
    relax   = jobkind in ['latticeopt','relax','bulkmod']
    kpt     = makeKPT(params)
    spinpol = makeSpinpol(params)

    def makeGPAWcalc(p):
        from gpaw import GPAW,PW,Davidson,Mixer,MixerSum,FermiDirac
        return GPAW(mode         = PW(p['pw'])                        
                    ,xc          = p['xc']
                    ,kpts        = kpt
                    ,spinpol     = spinpol
                    ,convergence = {'energy':p['econv']} #eV/electron
                    ,mixer       = ((MixerSum(beta=p['mixing'],nmaxold=p['nmix'],weight=100)) 
                                    if spinpol else (Mixer(beta=p['mixing'],nmaxold=p['nmix'],weight=100)))
                    ,maxiter       = p['maxstep']
                    ,nbands        = p['nbands']
                    ,occupations   = FermiDirac(p['sigma'])
                    ,setups        = p['psp']
                    ,eigensolver   = Davidson(5)
                    ,poissonsolver = None 
                    ,txt           ='log'
                    ,symmetry={'do_not_symmetrize_the_density': True}) 

    def makeQEcalc(p):
        from espresso import espresso   

        pspDict =   {'sherlock': {'gbrv15pbe':'/home/vossj/suncat/psp/gbrv1.5pbe'}
                    ,'suncat':   {'gbrv15pbe':'/nfs/slac/g/suncatfs/sw/external/esp-psp/gbrv1.5pbe'}}
        pspPath =  pspDict[getCluster()][params['psp']]

        return espresso( pw         = p['pw']
                        ,dw         = p['pw']*p['dwrat']
                        ,xc         = p['xc']
                        ,kpts       = kpt
                        ,spinpol    = spinpol
                        ,convergence=   {'energy':  p['econv']
                                        ,'mixing':  p['mixing']
                                        ,'nmix':    p['nmix']
                                        ,'maxsteps':p['maxstep']
                                        ,'diag':    'david'}
                        ,nbands     = p['nbands']
                        ,sigma      = p['sigma']
                        ,dipole     = {'status': p['kind'] == 'surface'}
                        ,outdir     = 'calcdir'
                        ,startingwfc= 'atomic+random' 
                        ,psppath    = pspPath
                        ,mode       = 'vc-relax' if jobkind=='vcrelax' else 'scf'
                        ,cell_factor= 2 if jobkind == 'vcrelax' else 1
                        ,output     = {'removesave':True})
    def makeQEVCcalc(p):
        from espresso import espresso
        pspDict =   {'sherlock': {'gbrv15pbe':'/home/vossj/suncat/psp/gbrv1.5pbe'}
                    ,'suncat':   {'gbrv15pbe':'/nfs/slac/g/suncatfs/sw/external/esp-psp/gbrv1.5pbe'}}
        pspPath =  pspDict[getCluster()][params['psp']]
        return  espresso( pw            = p['pw']
                        ,dw             = p['pw']*p['dwrat']
                        ,xc             = p['xc']
                        ,kpts           = kpt
                        ,nbands         = p['nbands']
                        ,dipole         = {'status': False}
                        ,sigma          = p['sigma']
                        ,mode           = 'vc-relax'
                        ,cell_dynamics  = 'bfgs'
                        ,opt_algorithm  = 'bfgs'
                        ,cell_factor    = 2.
                        ,spinpol        = spinpol
                        ,outdir         = 'calcdir'  
                        ,output         = {'removesave':True}
                        ,psppath        = pspPath
                        ,convergence=   {'energy':  p['econv']
                                        ,'mixing':  p['mixing']
                                        ,'nmix':    p['nmix']
                                        ,'maxsteps':p['maxstep']
                                        ,'diag':    'david'})

    def makeQEvibcalc(p):
        from espresso.vibespresso import vibespresso

        pspDict =   {'sherlock': {'gbrv15pbe':'/home/vossj/suncat/psp/gbrv1.5pbe'}
                    ,'suncat':   {'gbrv15pbe':'/nfs/slac/g/suncatfs/sw/external/esp-psp/gbrv1.5pbe'}}
        pspPath =  pspDict[getCluster()][p['psp']]

        return vibespresso( pw          = p['pw']
                            ,dw         = p['pw']*p['dwrat']
                            ,xc         = p['xc']
                            ,kpts       = kpt
                            ,spinpol    = spinpol
                            ,convergence=   {'energy':  p['econv']
                                            ,'mixing':  p['mixing']
                                            ,'nmix':    p['nmix']
                                            ,'maxsteps':p['maxstep']
                                            ,'diag':    'david'}
                            ,nbands     = p['nbands']
                            ,sigma      = p['sigma']
                            ,dipole     = {'status': p['kind'] == 'surface'}
                            ,outdir     = 'calcdir'  
                            ,startingwfc= 'atomic+random' 
                            ,psppath    = pspPath
                            ,output     = {'removesave':True})

    if dftcode =='gpaw':    
        if relax: return makeGPAWcalc(params)
        else: raise NotImplementedError, 'no GPAW calculator-maker for this kind of job'
    elif dftcode =='quantumespresso': 
        if relax:                   return makeQEcalc(params)
        elif jobkind=='vcrelax':    return makeQEVCcalc(params)
        elif jobkind == 'vib':      return makeQEvibcalc(params)
        else: raise NotImplementedError, 'no QE calculator-maker for this kind of job'


def makeKPT(params):    
    """
    Convert k-point density to Monkhorst-Pack grid size. Values forced to be even numbers.
    Special considerations if modeling molecule/bulk/surface. 
    """
    import math  as m
    import numpy as np

    recipcell,kpts = makeAtoms(params).get_reciprocal_cell(),[]
    for i in range(3):
        k = 2 * 3.14159 * m.sqrt((recipcell[i]**2).sum()) * params['kptden'] 
        kpts.append(2 * int(np.ceil(k / 2)))

    kind = params['kind']
    if   kind=='surface':   return np.array(kpts[:2]+[1])
    elif kind=='molecule':  return np.array([1,1,1])
    else:                   return np.array(kpts)


def makeSpinpol(params):
    magmomsinit = makeAtoms(params).get_initial_magnetic_moments()
    return any([x>0 for x in magmomsinit])


def optimizePos(atoms,calc,fmax):
    import ase.optimize as aseopt

    atoms.set_calculator(calc)
    dyn = aseopt.BFGS(atoms=atoms, logfile='qn.log', trajectory='qn.traj',restart='qn.pckl')
    dyn.run(fmax=fmax)


def trajDetails(atoms):
    """ Returns dictionary summary of an (optimized) Atoms object """
    import cPickle
    import numpy as np

    try: mag = atoms.get_magnetic_moments()
    except: mag = np.array([0]*len(atoms))
    return {'finaltraj_pckl':cPickle.dumps(atoms)
            ,'finalpos_pckl':cPickle.dumps(atoms.get_positions())
            ,'finalcell_pckl':cPickle.dumps(atoms.get_cell())
            ,'finalmagmom_pckl':cPickle.dumps(mag)}


def log(params,optatoms):
    import datalog
    datalog.log(optatoms,job_name=params['name'])


def VCRelaxScript():
    import ase,json,cPickle,os
    #######################
    print "Initializing..."
    #----------------------
    params,atoms = initialize()  # Remove old .out/.err files, load from fw_spec, and write 'init.traj'
    
    if not os.path.exists('intermediate.traj'):
        ##########################################
        print "Running VC Relax for first time..."
        #-----------------------------------------
        atoms.set_calculator(makeCalc(params))
        energy = atoms.get_potential_energy()   # Trigger espresso to be launched
        ase.io.write('intermediate.traj',atoms.calc.get_final_structure())

    ###########################################
    print "Running VC Relax for second time..."
    #------------------------------------------

    atoms = ase.io.read('intermediate.traj')
    atoms.set_calculator(makeCalc(params))
    energy = atoms.get_potential_energy() #trigger espresso to be launched
    ase.io.write('final.traj',atoms.calc.get_final_structure())
 
    ################################
    print "Storing Results..."
    #-------------------------------
    e0,f0   = atoms.get_potential_energy(),atoms.get_forces()
    atoms   = ase.io.read('final.traj')

    resultDict  = mergeDicts([params,trajDetails(atoms),
                                {'raw_energy':  e0
                                ,'forces_pckl': cPickle.dumps(f0)}])

    with open('result.json', 'w') as outfile:   outfile.write(json.dumps(resultDict))
    with open('result.json', 'r') as outfile: json.loads(outfile.read()) #test that dictionary isn't 'corrupted'
    log(params,atoms)
    return 0


def rank():
    import sys
    # Check for special MPI-enabled Python interpreters:
    if '_gpaw' in sys.builtin_module_names:
        import _gpaw        # http://wiki.fysik.dtu.dk/gpaw
        world = _gpaw.Communicator()
    elif '_asap' in sys.builtin_module_names:
        import _asap # http://wiki.fysik.dtu.dk/asap, can't import asap3.mpi here (import deadlock)
        world = _asap.Communicator()
    elif 'asapparallel3' in sys.modules: # Older version of Asap
        import asapparallel3
        world = asapparallel3.Communicator()
    elif 'Scientific_mpi' in sys.modules:
        from Scientific.MPI import world
    elif 'mpi4py' in sys.modules:
        world = MPI4PY()
    else:
        from ase.parallel import DummyMPI
        world = DummyMPI()# This is a standard Python interpreter:
    rank = world.rank
    size = world.size
    return rank


def mergeDicts(listDicts): 
    import itertools
    return dict(itertools.chain.from_iterable([x.items() for x in listDicts])) #presumes no overlap in keys


def getCluster():
    import os
    hostname = os.environ['HOSTNAME'].lower()
    if      'sh'    in hostname: return 'sherlock'
    elif   'gpu-15' in hostname: return 'sherlock'
    elif    'su'    in hostname: return 'suncat' #important to distinguish suncat2 and 3?
    elif    'kris'  in hostname: return 'kris'
    else: raise ValueError, "getCluster did not detect SH or SU in %s"%hostname


if __name__ == '__main__': VCRelaxScript()

