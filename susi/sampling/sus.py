
import numpy as np
import random
import math
import copy

from . import acs
from . import sortbi


def sus(N,strurel_obj,U_seeds,seedp,p,bi,lambda_iter,vers,dropbi,fixcsteps,pa=0.2,reuse=1,bstar=0.0,i0=0.0,susi=0,xk="none"):
    '''
    ###################################################################
    description: sus-like step for both, sus and susi

    ###################################################################
    parameters:

        N: int

                    number of samples per subset

        strurel_obj: strurel object

                    the underlying strurel object which defines the analyzed strurel_obj

        U_seeds: array

                    subset seeds in U-space (independent standard normal space)

        seedp: float in (0,1]

                    percent of seeds which are used, e.g. 1.0 means all seed samples are used
                    as a starting point for a Markov chain

        p: float in (0,1)

                    intermediate subset proability

        bi: float

                    subset threshold

        lambda_iter: float

                    scaling parameter adjusting the proposal spread

        vers: string in {"a","b"}

                    select a version for variance adaption in acs:
                    if "a" is selected, adapt all variables the same way 
                    if "b" is selected, adapt according to the relevance
                    of the individual components with respect to the ls value

                    note: choose "a" if all random variables have similar 
                    impact on the ls value, and "b" else

        dropbi: int in {0,1,2}
        
                    decide whether samples that have ls equal to the threshold
                    of the subset or are close to such a sample by MCMC are not
                    used for MCMC in the next subset (not used as seed samples)

                    0: use all samples
                    1: drop samples with equal ls
                    2: 1 + drop samples close to 1 by MCMC

        fixcsteps: integer, default=10

                   defines the steps made to create the single MCMC chains,
                   for creation of new samples in function acs (see /sampling/acs)

        pa: float value in (0,1), default=0.2

                 defines the percentage of chains in MCMC sampling after which
                 the proposal spread is updated

        reuse: int in {0,1}, default=1

                decide whether to reuse sample seeds (1)
                          or not reuse sample seeds (0)

        bstar: float, default=0.0

               defines the threshold for failure of the structure

        i0: float, default=0.0

                changes the way we update the proposal spread
                e_iter=1.0/math.sqrt(i0+chain/Na)

        susi: int in {0,1}, default=0
    
                adapts the limit state function as necessary
                decide whether we keep the same limit state 
                function always (0)
                or use the previous ls for creation of samples
                and another one for evaluation as required in susi (1) 

        xk: str in names of strurel variables

                name of dynamic variable, not relevant for sus evaluations

                e.g.
                    "x0"


    ###################################################################
    returns: [U_ls_sorted,U_fail,U_no_fail,evals,lambda_iter,bi,p,savechains]

        U_ls_sorted: array

              samples in Uspace (row-wise)
              containing the limit state of each sample in the last column  
              and being in ascending order according to the limit state values

        U_fail: array

              subset of U_ls_sorted, only failed samples
                (limit states smaller than subset threhsold bi)

        U_no_fail: array

                subset of U_ls_sorted, only non-failed samples
                (limit states greater than subset threshold bi)

        evals: int

                number of limit state evaluations performed 

        lambda_iter: float

                    scaling parameter adjusting the proposal spread

        bi: float

                    subset threshold

        p: float in (0,1)

                    intermediate subset proability

        savechains: array

                this is dedicated to later evaluate the dependencies
                of within subset samples in each subset

    '''

    d=len(strurel_obj.ls.argslist)
    Nseeds=len(U_seeds)

    if reuse==1:
        Nnew=max(N-Nseeds,0) 
        #non-negative -> this can happen if we adapt N because of predicted value to higher than previous N and were in SuS with high p0
    else:
        Nnew=N

    #number of chains
    if fixcsteps==0:
        Nc=int(Nseeds*seedp)
    else:
        #if Nc>Nseeds had to choose seeds repetitive - in this case let Nc less so that chainlength increases
        Nc=min(int(float(Nnew)/fixcsteps),Nseeds)

    #new generated steps in each chain (this is only equal to clen if reuse!=1
    if int(Nc)!=0:
        csteps=int(math.floor(Nnew/float(Nc)))
    else:
        csteps=0

    csteps_last_add=int(Nnew-csteps*Nc)

    if int(Nc)!=0:
        csteps=csteps+math.floor(csteps_last_add/Nc)
    else:
        csteps=0    
    csteps_last_add=int(Nnew-csteps*Nc)   

    if int(Nc)!=0:
        #using seeds for start of markov chain #already randomly selected
        choose_seeds_randomly=random.sample(range(0,Nseeds),Nc)
        #avoid list collisions
        U_seeds2=copy.deepcopy(U_seeds)
        U_seeds_use=U_seeds2[choose_seeds_randomly]
    else:
        U_seeds_use=U_seeds

    if susi==1: #change limit state to generate new samples based on previous subset (prev uk)
        strurel_obj.ls.det(xk,strurel_obj.result.uk[-2])

    if csteps==0: #no need to create new samples, we already have enough seeds
        U_ls=U_seeds_use
        evals=0
        savechains=None #this is error handling but savechains is for computation of dependencies anyways and does not effect the result of susi
    else:
        ##Markov Chain Monte Carlo for generating new seeds:
        [U_new,lambda_iter,savechains]=acs(U_seeds,U_seeds_use,Nc,csteps,csteps_last_add
            ,strurel_obj,bi,lambda_iter,pa,i0,vers,reuse)  

        #new samples generated 
        evals=len(U_new)

        if reuse==1:
            #combine seeds samples and newly simulated samples
            U_ls=np.row_stack([U_seeds,U_new])
        else:
            U_ls=U_new

    if susi==1: #change limit state for evaluation: this the uk returned by the prediction step
        strurel_obj.ls.det(xk,strurel_obj.result.uk[-1])
        #evaluate with new ls (samples w/o ls by leaving out last col [:-1])
        U_next_lsonly=np.array([
            strurel_obj.ls.gfun(U_ls[r,:-1])
            for r in range(len(U_ls[:,0]))])

        #ls added to sample    
        U_ls=np.column_stack([U_ls[:,:-1],U_next_lsonly])
        #could save the samples here for later use        

        #double evals by SuSI
        evals=evals+len(U_next_lsonly)


    if susi==1: #susi: p not given in advance, instead bi given
        [U_ls_sorted,U_fail,U_no_fail,bi,p]=sortbi(U_samples_ls=U_ls,p=0,bi=bi,bstar=bstar)
    else: #sus: p given in advance
        [U_ls_sorted,U_fail,U_no_fail,bi,p]=sortbi(U_ls,p,bi,bstar)

    return([U_ls_sorted,U_fail,U_no_fail,evals,lambda_iter,bi,p,savechains])




