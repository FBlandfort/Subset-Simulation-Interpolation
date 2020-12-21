

import numpy as np
import math
from scipy.stats import norm
from . import sortbi
import random
from scipy.stats import beta
from scipy.interpolate import UnivariateSpline

def predstep(strurel_obj,Npred,U_seeds,pl,pu,xk,prob_b,q,mono,fixNpred,bstar,found_cand=0,predict_by_prev=0,maxsteps=5):
    '''
    ###################################################################
    description:  

    #add function predict_by_prev:
    strurel_obj.results[-1].pfi
    strurel_obj.results[-1].xk
    that should also be added above so that know good xk intial values (y[1])

    #note that fixcsteps is fixed in the prediction step to 10 or if not possible less

    ###################################################################
    parameters: 

        strurel_obj: strurel object

                    the underlying strurel object which defines the analyzed strurel_obj

        Npred: positive integer

               number of samples used to evaluate in the prediction step

        U_seeds: array

                sample seeds in standard normal space (U-space)

        pl: float in (0,pu)

                lower boundary for admissible intermediate probability
                note that (pl+pu)*0.5 as an intermediate probability 
                should refer to the desired grid point selection

        pu: float in (pl,1)

                upper boundary for admissible intermediate probability

        xk: str in names of strurel variables

            name of dynamic variable

            e.g.
                "x0"

        prob_b: float in (0,1), default=0.8

               this is the probability that sets the stopping criterion for
               the prediction step, if we have a probability higher than prob_b
               (by prediction) to have found a grid point that yields an 
                intermediate failure probability in [pl,pu], then we stop and take it

        q: float in (0,1), default=0

                quantile of xk selected for first evaluations
                if set zero, then we use max_error to select an
                appropriate value 

        mono: str in {"i","d"}

              monotonicity of the conditional failure function
              if "i", then we assume the failure probability is 
              increasing if xk is increased
              if "d", then we assume the failure probability is
              decreasing if xk is increased

        fixNpred: positive integer, default=0

                If fixNpred=0, we use only available samples in the given subset
                otherwise we create fixNpred new ones, meaning that we guarantee
                fixNpred samples for prediction (however this is more expensive 
                than fixNpred=0)  

        bstar: float, default=0.0

               defines the threshold for failure of the structure

        found_cand: float

                if presampling is used to find a candidate state in advance,
                it is used instead of guessing a new one
                if set to zero: found_cand=0, then we start without any sample given
                and produce a new one instead 

        predict_by_prev: int in {0,1},default=0
        
               if 0, the prediction step is normally performed by extrapolation
               if 1, previous results are used to select the next grid point for xk

        maxsteps: positive integer, default=5

                maximum steps for the prediction step,
                if there is no satisfying candidate for
                a grid point found after this number of
                preditions we stop anyhow and take the 

    ###################################################################
    returns: [float(next_point),evals,p,steps] deterministic value xk that was predicted for intermediate prob p

        float(next_point): float

                value of the dynamic variable xk which will
                be used for the next subset (set xk equal to it
                in the limit state function in the next subset)
   
        evals: int

                number of limit state evaluations

        p: float in (0,1)
    
                estimated intermediate probability by the prediction step samples
                note that Npred is typically lower than the number of samples N
                in the evaluation afterwards, so that this estimation is not
                an accurate one in general. However, it is used to adjust the number
                of samples N for evaluation according to this estimated probability
                so that samples are efficiently split accross different subset levels 

        steps: int

                number of prediction steps, which were performed
                until termination returning the "next point" for
                the next subset

    '''


    prepoints_sorted=0
    ps=0.5*(pl+pu)
    lambda_iter_not_updated=1   
    d=len(strurel_obj.ls.argslist)
    Nseeds=len(U_seeds)

    #count evaluations
    evals=0

    #to avoid doubles by adjusting special values
    count_adj=2

    #linear extrapolation
    order=1 

    last_fail_prob=strurel_obj.result.pfi[-1]
    prepoints=np.array([last_fail_prob,strurel_obj.result.uk[-1]]) #last failure prob and deterministic set xk 

    if np.sum([itype=="susi" for itype in strurel_obj.result.itype])==0: #initial step, no susi result before

        if found_cand==0:
            #have to keep this, it does a step away from the boundary where we are in advance
            if mono=="i":        
                candidate=norm.ppf(0.5)
            elif mono=="d":
                candidate=norm.ppf(0.5)
        else:
            candidate=found_cand

    else: #have more points already so that a better extrapolation is possible
        
        second_last_fail_prob=strurel_obj.result.pfi[-2]
        prepoints=np.row_stack((prepoints,[second_last_fail_prob,strurel_obj.result.uk[-2]]))
        prepoints_sorted=prepoints[prepoints[:,0].argsort()] 

        #define extrapolation function
        extrapolation_f=UnivariateSpline(prepoints[:,0], prepoints[:,1], k=order,s=0)

        #find a new candidate   
        candidate=extrapolation_f(ps*last_fail_prob)   

    if float(Nseeds)>=float(Npred): #no need for new samples in every case if this condition is fulfilled
        fixNpred=0

    if fixNpred==0:
        #as Nseeds might be low - could also sample some samples additionally - and should for cases if Nseeds becomes very small
        Npred=min(Nseeds,Npred)
        choose_seeds_randomly=random.sample(range(0,Nseeds),Npred)
        U_seeds_use=U_seeds[choose_seeds_randomly]

    else:  #create new samples by acs for testing, all Npred samples:
        Nnew=Npred-Nseeds

        #fixcsteps_pred to 10
        fixcsteps=min(10,Nnew)
        Nc=min(int(float(Nnew)/fixcsteps),Nseeds)

        csteps=int(floor(Nnew/float(Nc)))
        csteps_last_add=int(Nnew-csteps*Nc)
        csteps=csteps+floor(csteps_last_add/Nc)
        csteps_last_add=int(Nnew-csteps*Nc)   

        #using seeds for start of markov chain #already randomly selected
        choose_seeds_randomly=random.sample(range(0,Nseeds),Nc)

        #avoid list collisions
        U_seeds2=copy.deepcopy(U_seeds)

        U_seeds_use=U_seeds2[choose_seeds_randomly]
        [U_seeds_use,_]=acs(U_seeds,U_seeds_use,Nc,csteps,csteps_last_add,strurel_obj,0.0,0.6,0.2,1.0,"a",0) #[:,:-1] U_seeds_use w/o last?
        #later: bi=bstar, reuse does not matter because of savechains, "a" could be changed...      
        #got U_seeds_use in desired len
        U_seeds_use=np.row_stack((U_seeds,U_seeds_use))
        evals=evals+Nnew
            
    #to track if we repetitively do not find useful values in prediction step
    countp1=0
    countp0=0
    marginal_change=1
    p_check=0.0 
    steps=0

    while p_check<prob_b and steps<maxsteps: #have to search for candidate in range

        if steps>0:
            prepoints_sorted=prepoints[prepoints[:,0].argsort(axis=0)] 
            extrapolation_f=UnivariateSpline(prepoints_sorted[:,0], prepoints_sorted[:,1], k=order,s=0)
            candidate=extrapolation_f(ps*last_fail_prob)

        #avoid to overshoort too much, this is important as it can happen if p=1 above, note that 12 belongs to standard normal space
        if candidate>12:
            candidate=12-marginal_change
            marginal_change=marginal_change+1e-10
        if candidate<-12:
            candidate=-12+marginal_change
            marginal_change=marginal_change+1e-10
            
        steps=steps+1

        #new limit state function according to candidate value
        strurel_obj.ls.det(xk,val=candidate)

        #corresponding limitstates
        U_next_lsonly=np.array([
            strurel_obj.ls.gfun(U_seeds_use[r,:-1]) ####changed to :-1!! check
            for r in range(len(U_seeds_use[:,0]))])
        
		#replace limitstates with new ones    
        U_seeds_use[:,-1]=U_next_lsonly
            
        #order samples and give p
        [U_ls_sorted,U_fail,U_no_fail,bi,p]=sortbi.sortbi(U_samples_ls=U_seeds_use,p=0,bi=0.0,bstar=bstar) ##7-11 changed from N=Npred to default
    
        ##candidate is given now with evaluated p ->maybe 0, 1 or ok

        more_steep_step=1 #it is hard to control the case =1 so that we might have far steps there, need to compensate in p==0
        if p==0:
            p=p-more_steep_step+marginal_change*1e-04 #so that extrapolation possible
            marginal_change=marginal_change+1e-02 #avoid equalities in samples so that we can use extrapolation function on it later
            maxsteps=maxsteps+1
            countp0=countp0+1
            if countp0>1:
                prepoints[-1]=[last_fail_prob*p,candidate]
            else:
                prepoints=np.row_stack((prepoints,[last_fail_prob*p,candidate])) 
        elif p==1:
            p=p-marginal_change*1e-02
            marginal_change=marginal_change+1e-02      
            maxsteps=maxsteps+1  
            countp1=countp1+1
            if countp1>1:
                prepoints[-1]=[last_fail_prob*p,candidate]
            else:
                prepoints=np.row_stack((prepoints,[last_fail_prob*p,candidate])) 
        else:
            prepoints=np.row_stack((prepoints,[last_fail_prob*p,candidate]))     
            if (type(prepoints_sorted)!=int):  
                if p*last_fail_prob in prepoints_sorted[:,0]:
                    next_point=candidate
                    break  #this can be made better, actually we avoid an error here but more elegant solutions are possible
 
        p_check=beta(int(p*Npred),int(Npred)).cdf(pu)-beta(int(p*Npred),int(Npred)).cdf(pl)  

        if math.isnan(p_check)==True:
            p_check=0.0        

        #avoid to overshoort too much, this is important as it can happen if p=1 above
        if candidate>12:
            candidate=12
        if candidate<-12:
            candidate=-12

        if float('-inf') < candidate < float('inf'):                    
            next_point=candidate #set from 7 to 15 below
        if candidate>10 and steps>15: #note that this belongs to probability one in CPU approximation anyways (as it is with respect to N(0,1) distribution
            next_point=10
            break
        if candidate<-10 and steps>15: #steps>7 to avoid first steps with overshoot make us break the loop too early
            next_point=-10
            break

        if maxsteps>70: #if we remain estimation of p=0 or p=1 for whatever reason we do not end in an endless loop
            break

    evals=evals+steps*Npred

    return([float(next_point),evals,p,steps])


