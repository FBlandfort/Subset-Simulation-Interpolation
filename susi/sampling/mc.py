



import numpy as np

from . import sortbi





def mc(N,strurel_obj,p=0,bi=0,bstar=0):
    '''
    ###################################
    description:
    
    Crude Monte Carlo simulation with sophisticated returns,
    dependent on limit state values of samples


    ###################################
    parameters: N,strurel_obj,p=0,bi=0,bstar=0

    N: number of samples per subset

    strurel_obj: main/strurel object 

    p: intermediate subset probability, 
        set to zero if threshold is fixed (bi, bstar) as e.g. in susi 

    bi: subset threshold, if p!=0 given it is set according to p
        if bi<bstar by p then set bi=bstar

    bstar: final threshold, typical value bstar=0, if p yields bi<bstar, 
            instead bstar set as new threshold providing a new probability 

    ##################################
    returns: [U_samples_ls_sorted,U_fail,U_no_fail,N,bi,p]
        
    U_samples_ls_sorted: samples in U-space [:,:-2] 
                        with limit state value in U-space U[:,-1] (last entry)

    U_fail: U_samples below threshold

    U_no_fail: U_samples above threshold

    N: number of samples per subset

    bi: subset threshold, if p given it is set according to p
        if bi<bstar by p then set bi=bstar

    p:  intermediate subset probability
     
    '''

    if int(p*N)!=p*N:
        print("Warning: p*N not integer")

    #produce d(dimension of attribute collection) samples in U-space (standard normal)
    d=len(strurel_obj.ls.argslist)
    U_samples=np.array(np.random.multivariate_normal(tuple(np.zeros(d)),np.identity(d),(N)))

    #compute limit state values of samples in X-space (true thresholds)
    #note: the ordering of the limit state values in X and U-space is the same
    limitstate_vals=np.array([
            strurel_obj.ls.gfun(U_samples[r,:])
            for r in range(len(U_samples[:,0]))])

    #stack limit states to corresponding samples (last column)
    U_samples_ls=np.column_stack([U_samples,limitstate_vals])

    #sort samples according to limitstate_value and take p0 lowest valued samples # ascending order, starting with lowest [0] up to highest [Nnew-1]
    U_samples_ls_sorted=U_samples_ls[U_samples_ls[:,d].argsort()]

    #sort samples, split into in and above threshold samples and return intermediate probability and threshold
    [U_samples_ls_sorted,U_fail,U_no_fail,bi,p]=sortbi.sortbi(U_samples_ls,p,bi,bstar=bstar) 

    return([U_samples_ls_sorted,U_fail,U_no_fail,N,bi,p])




