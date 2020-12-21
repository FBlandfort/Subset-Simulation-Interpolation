

import numpy as np
import warnings



def sortbi(U_samples_ls,p,bi,bstar): #better name: set threshold or p on Usamples
    '''
    ###################################################################
    description: 

    this function sorts the given samples and selects a threshold
    that suits the Subset Simulation procedure (either according to
    a specific probability p of exceeding the threshold, or according
    to a threshold bstar that corresponds to failure

    ###################################################################
    parameters:

        U_samples_ls: array
    
                samples in standard normal space with corresponding
                limit state (in X-space) as last entry in every row

        p: float in (0,1)

                intermediate subset probability

        bi: float

                threshold value

        bstar: float

                critical threshold value 


    ###################################################################
    returns: [U_samples_ls_sorted,U_fail,U_no_fail,bi,p]

        U_samples_ls_sorted: array

                    samples in standard normal space with corresponding
                    limit state (in X-space) as last entry in every row,
                    sorted according to their limit state value (ascending order)

        U_fail: array

                subset of U_samples_ls_sorted containing all samples whose limit state values
                exceeds the threshold bi

        U_no_fail: array

                subset of U_samples_ls_sorted containing all samples whose limit state values
                does not exceed the threshold bi

        bi: float

            threshold of the subset

        p: float in (0,1)

            intermediate subset probability


    '''
    #sort samples according to ls, ascending
    U_samples_ls_sorted=U_samples_ls[U_samples_ls[:,-1].argsort()]
    #highest ls in and lowest ls out the next p0
    N=len(U_samples_ls_sorted)
    psave=p

    if p!=0: #p given in advance, search bi (susi, p>0)
        bi_in=U_samples_ls_sorted[int(p*N)-1,-1] #p*N-th sample because counting starts at 0: -1
        bi_out=U_samples_ls_sorted[int(p*N),-1]
        U_fail=U_samples_ls_sorted[0:int(p*N)]
        U_no_fail=U_samples_ls_sorted[int(p*N):N]
        bi=bi_in 

    else: #bi given in advance (susi , set p=0 as not allowed indicator)
        U_fail=U_samples_ls_sorted[np.where(U_samples_ls_sorted[:,-1]<bi)]
        U_no_fail=U_samples_ls_sorted[np.where(U_samples_ls_sorted[:,-1]>=bi)]

    p=float(len(U_fail))/len(U_samples_ls)  

    #if failure reached, use failure threshold
    if bi<bstar:
        if psave==0:
            warnings.warn("Attention, threshold bi={0} was set fixed but got below critical threshold bstar={1}, bstar={1} was used for evaluation".format(bi,bstar),Warning) 
        bi=bstar
        p=float(len(np.where(U_samples_ls_sorted[:,-1]<bi)[0]))/len(U_samples_ls_sorted)
        U_fail=U_samples_ls_sorted[0:int(p*N)]
        U_no_fail=U_samples_ls_sorted[int(p*N):N]   

    return([U_samples_ls_sorted,U_fail,U_no_fail,bi,p])     




