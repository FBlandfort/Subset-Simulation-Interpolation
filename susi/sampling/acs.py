
import math
import numpy as np
import copy

def acs(U_seeds,U_seeds_use,Nc,csteps,csteps_last_add,strurel_obj,bi,lambda_iter,pa,i0,vers,reuse):
    '''
    ###################################################################
    description: 

    adaptive Conditional Sampling according to (paper reference)
    starts at given seeds U_seeds and returns new samples by MCMC conditional sampling that are in 
    the subset with limitstate smaller than given bi.
    the proposal spread is varied by varying lambda_iter (scaling of the proposal),
    the last lambda_iter is also returned so that it can be used in the next subset as a good
    starting value for scaling of the proposal spread

    ###################################################################
    parameters:

        U_seeds: samples from the last subset (in standard normal 
                    space which landed in the next (now actual)
                    subset and are kept for further evaluation 

        U_seeds_use: subset of U_seeds_use
                    samples in standard normal space which will be used
					as seeds for creating new samples inthe next subset 
					(not necessarily equal to the set of samples which 
					are kept from the previous subset)

        Nc: int

                    number of MCMC chains

        csteps: int

                    number of steps in each Markov chain

        csteps_last_add: int

                    extends the last Markov chain so that we get
                    the exact number of samples we want to after 
                    MCMC sampling. Note that this is often necessary
                    and may not be reached by only adapting csteps alone.

        strurel_obj: strurel object

                    the underlying strurel object which defines the analyzed strurel_obj

        bi: float

                threshold value

        lambda_iter: float

                    scaling parameter adjusting the proposal spread

        pa: float value in (0,1), default=0.2

                 defines the percentage of chains in MCMC sampling after which
                 the proposal spread is updated

        i0: float, default=0.0

                changes the way we update the proposal spread
                e_iter=1.0/math.sqrt(i0+chain/Na)

        vers: str in {"a","b"}, default="b"
                
                version "a" results in equally updated proposal spreads
                for all variables in acs (see /sampling/acs),
                version "b" weights according to the importance of specific
                variables

        reuse: int in {0,1}, default=1

                decide wheter to reuse sample seeds (1)
                          or not reuse sample seeds (0)



    ###################################################################
    returns: [U_set,lambda_iter,savechains]

        U_set: array

                samples in U-space (row-wise), containing the newly created
                samples by MCMC
                As usual, the last column contains the corresponding
                limit state value of the samples   

        lambda_iter: float

                    scaling parameter adjusting the proposal spread

        savechains: array

                this is dedicated to later evaluate the dependencies
                of within subset samples in each subset
    '''

    #Na chosen according to pa e.g. after 10 percent of chains (0.1)
    Na=pa*Nc 

    #but next we have to be sure that we do not adapt IN a chain as this may lead to a bias
    Na=math.ceil(Na)
    Ulen=len(U_seeds_use)    

    csteps=int(csteps)
    csteps_last_add=int(csteps_last_add)

    #target acceptance rate, which was found to yield a good performance (compare
    astar=0.44  

    if vers=="a":        
        #1a in ERA initial sigma
        sigma_alli=np.ones(len(U_seeds_use[0,:])-1) #-1 in range due ls
        sigma_zero=np.ones(len(U_seeds_use[0,:])-1)
    else:  #use all U_seeds here as we want to compute best possible sample stds
        seed_std=np.array([np.std(U_seeds[:,seedi]) for seedi in range(len(U_seeds[0,:])-1)]) 
        sigma_alli=seed_std
        sigma_zero=copy.deepcopy(seed_std)
    
    #2 in ERA
    #note: no need to permute the seeds to remain unbiased (usually ordered by ls) - did that above already in SuS, when selecting seeds
    U_seeds_permute=copy.deepcopy(U_seeds_use)
    U_seeds_permute_nols=copy.deepcopy(U_seeds_permute[:,:-1])


    #initialize
    sigma_alli=np.minimum(sigma_zero*lambda_iter,np.ones(len(sigma_alli)))
    rho_alli=np.sqrt(np.ones(len(sigma_alli))-(sigma_alli**2))
    cov_set=np.identity(len(U_seeds_permute_nols[0,:]))*(1.0-rho_alli**2)
    acc=0

    if reuse==1:
        savechains=np.zeros((csteps+1,Ulen))
        savechains[0,:]=U_seeds_permute[:,-1]
        fixstart=1
    else:
        savechains=np.zeros((csteps,Ulen))
        fixstart=0

    #3:iterations
    for chain in range(Ulen):

        #adapt after chain that best suits pa 
        if (chain%Na==0 and chain!=0):

            'accrate -> new lambda_iter'
            acc_rate=acc/float(csteps_now*Na)

            #e_iter is hard to choose if N is small (or p0 high)
            e_iter=1.0/math.sqrt(i0+chain/Na)
            lambda_iter=math.exp(math.log(lambda_iter)+e_iter*(acc_rate-astar))  

            sigma_alli=np.minimum(sigma_zero*lambda_iter,np.ones(len(sigma_alli)))
            rho_alli=np.sqrt(np.ones(len(sigma_alli))-(sigma_alli**2))   
            #note: we use (1.0-rho_alli**2) for cov_set, which is the variance
            #(squared standard deviation sqrt(1.0-rho_alli**2)
            cov_set=np.identity(len(U_seeds_permute_nols[0,:]))*(1.0-rho_alli**2)

            acc=0       

        #if we have no acceptance have to get back, for first iteration k that is the seed
        U_save_prev=U_seeds_permute[chain,:]
        
        #initialize steps on the seed of the chain
        mean_set=tuple(U_seeds_permute_nols[chain,:]*rho_alli)
        
        csteps_now=csteps

        if chain==Ulen-1: #last chain
            csteps_now=csteps+csteps_last_add

        for iteration in range(csteps_now):

            #produce the next random sample in the chain (note that we have parallelization
            #because samples are multi-dimensional in the use case of the algorithm
            U_next=np.array(np.random.multivariate_normal(mean_set,cov_set,(1,1))[0])
            
            #corresponding limitstates
            U_next_lsonly=np.array([
                strurel_obj.ls.gfun(U_next[r,:])
                for r in range(len(U_next[:,0]))])

            #ls added to sample    
            U_next_ls=np.column_stack([U_next,U_next_lsonly])

            #count acceptance - in our case here that is just 1 dim (true or false)'
            acc=acc+np.sum(U_next_lsonly<=bi)  
            
            #only take new samples which fulfull the set's limitstate condition, else take seeds           
            U_next_ls=np.array([
                U_next_ls[r]
                if U_next_lsonly[r]<=bi
                else U_save_prev
                for r in range(len(U_next_ls))])[0]

            #for the next iteration we go back to the current if not accepted
            #CAREFUL if dimension >1
            U_save_prev=U_next_ls
            
            #for mean get rid of last column that is ls 
            mean_set=tuple(U_next_ls[:-1]*rho_alli)    

            #add samples and ls to the set of states
            if (chain==0 and iteration==0):
                U_set=copy.deepcopy(U_next_ls)
            else:
                U_set=np.row_stack([U_set,U_next_ls])     
    
            if iteration <csteps: #ATTENTION depends on if we re-use seeds? above then no useeds set...
                savechains[fixstart+iteration,chain]=U_next_ls[-1] #start at 1+ because first is for seed
       
    return([U_set,lambda_iter,savechains])

