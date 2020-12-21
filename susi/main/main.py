

import numpy as np
import copy
import math
import scipy

from susi import props
from susi import sampling
from .result import result_obj, strurel_result


from scipy.stats import norm


class strurel(object): 
    '''
    ###################################################################
    Description:

    The strurel object
    - is defined by the properties in susi.props (stochastic attributes and limit states)
    - we apply sampling methods (susi.sampling) on it
    - all results from the sampling are collected in it and saved in the strurel_result class,
        this yields a discrete set of results in the strurel_result class
    - a continuous result is created by the given interpolation methods 

    ###################################################################
    Parameters:

        attrs: object of class col_attr (see /props/attrs), no default

                stochastic properties of the structure, related to the given 
                limit state equation (ls)

				e.g. 
                    props.col_attr([props.attr(name="x{0}".format(i),rtype="n",mx=1e-10,vx=1e10) 
                            for i in range(3)])

        ls: object of limitstate class (see /props/limitst), no default

            limit state function of the stochastic reliability
            formulation
			
            e.g.
                ls=props.limitstate(name="ls1"
                ,f_str='-(np.sum([x[0],x[1],x[2]]))/math.sqrt(3)+4.7534243088229'    
                ,argslist=["x2","x0","x1"]
                ,prob=None)

		
        corrX:  matrix (type: numpy array), default=independent 

                correlation matrix of stochastic properties,
                default is np.eye (independence),

                e.g.
                    np.array([[1,0,0],[0,1,0.2],[0,0.2,1]])


    ###################################################################
    Returns:

        Creates strurel object for further analysis by e.g. SuS or SuSI

    '''
    def __init__(self,attrs,ls,corrX=None):
        self.attrs=attrs
        self.ls=ls
        self.corr_z=0
        if corrX is None: #default set independent variables
            self.corrX=np.identity(len(self.ls.argslist))
        else:
            self.corrX=corrX
        self.rvs=props.col_rv(rvs=self.attrs.get_rvs(),corrX=corrX)
        if ls!=None:
            self.match=props.limitstate_match(attrs,ls.argslist)
        self.results=list()
        self.result=None

        self.ls.prob=self

    def get_result(self               
                    ,method="sus"            

                    ,Nlist=[500],plist=[0.1],bstar=0.0

                    ,fixcsteps=10,palist=[0.2],reuse=1,vers="b",saveU=0
                    ,choose_seedN=0,dropbi=0,seedplist=[1.0],i0=0.0

                    ,boundpf=1e-15    

                    ,xk="x0",mono="i",max_error=1e-10,bi=0.0

                    ,Npred=100,predict_by_prev=0,prevNpredict=1,prob_b=0.8

                    ,pl=0.2,pu=0.4,firstvalsampling=0,firstvalsamplingN=20,Nbound=15000,raise_err=1
                    ,fixNpred=0,testmc=0,maxsteps=5,q=0):

        
        '''
        ###################################################################
        #Description:

        Use Subset Simulation or Subset Simulation Interpolation to compute
        the reliability of the strurel_object. Note that there are many
        option parameters for scientific analysis of the properties of 
        the algorithms. By default, they are set to values that provide
        good efficiency in most cases. 

        #Important Remark:
        SuSI also uses the SuS parameters
        as it is based on and utilizes ordinary SuS


        ###################################################################
        Parameters:

        ########
        #general
        
        method: str, "sus" or "susi", default="sus"
                choose method Subset Simulation or Subset Simulation Interpolation


        ########                    
        #SuS parameters - obligatory (note these are lists to allow adaptiveness with
                                respect to the subset level)

        Nlist: list of integer values, default=[500]
            
               defines the sample number in each subset, starting with 
               the first list element Nlist[0] for Monte Carlo, then taking
               Nlist[i] samples in subset i,
               if last list element is reached, last element is taken for all
               higher level subsets

               e.g.
                    [500,400] (500 samples for Monte Carlo, then 400 for
                    all higher level subsets)        

        plist: list of float values in (0,1), default=0.1            

               defines the intermediate subset probability

               as in Nlist, the list allows to select different values
               for different subsets
        
               e.g.
                   [0.1,0.3,0.5] 


        bstar: float, default=0.0

               defines the threshold for failure of the structure
                


        ########  
        #SuS parameters - optional (alteration for scientific examination mostly,
                             default settings provide good efficiency)
  

        fixcsteps: integer, default=10

                   defines the steps made to create the single MCMC chains,
                   for creation of new samples in function acs (see /sampling/acs)
                                   

        palist: list of float values in (0,1), default=[0.2]

                defines the percentage of chains in MCMC sampling after which
                the proposal spread is updated

        reuse: int in {0,1}, default=1

                decide wheter to reuse sample seeds (1)
                          or not reuse sample seeds (0)


        vers: str in {"a","b"}, default="b"
                
                version "a" results in equally updated proposal spreads
                for all variables in acs (see /sampling/acs),
                version "b" weights according to the importance of specific
                variables

        saveU: int in {0,1}, default=0

                decide wheter samples in Uspace and corresponding limitstates
                in Xspace are saved (1) or not (0)
 
        choose_seedN: integer>=0, default=0

                if set zero, the parameter remains unused,
                if >0, then we select choose_seedN seeds for MCMC
                so the number of seeds is explicitly stated then

                e.g. 
                    10

        dropbi: int in {0,1,2}, default=0

                decide whether we drop sample seeds that are related to 
                the limit state threshold, these are not in the stationary
                distribution, if 0 then nothing is dropped,
                if 1 then samples with limit state value equal to the
                threshold are dropped
                if 2 also MCMC chain elements close to samples with
                limit state value equal to the threshold are dropped,
                samples are dropped up to distance "rem_s_samps=5" 
                from such samples within chains

        seedplist: float (0,1], default=[1.0]

                list of percentage of seeds used for MCMC sampling
                note that fixcsteps overwrites this if set !=0

        i0: float, default=0.0

                changes the way we update the proposal spread
                e_iter=1.0/math.sqrt(i0+chain/Na)

        #######
        #SuS parameters for parameter state model like results: 
                specify a probability where SuS stops calculation
                otherwise SuS would not terminate (R>0)

        boundpf: float value in (0,1)

                sets the minimum of considered failure probabilities,
                if it is reached, the algorithm terminates

                e.g.
                   1e-12    
                    
        #######
        #SuSI parameters obligatory (+SuS parameters obligatory)

        xk: str in names of strurel variables

            name of dynamic variable

            e.g.
                "x0"

        mono: str in {"i","d"}

              monotonicity of the conditional failure function
              if "i", then we assume the failure probability is 
              increasing if xk is increased
              if "d", then we assume the failure probability is
              decreasing if xk is increased


        max_error: float in (0,1)

              the maximum error allowed by approximation/extrapolation
              this refers to the sum of both errors by extrapolation,
              extrapolation above the maximum xk evaluated and 
              extrapolation below the minimum xk value evaluated
              these parts of the domain are evaluated under the safety principle
              by default
 
              e.g.
                1e-10,bi=0.0


        #prediction step
        
        Npred: positive integer, default=100

               number of samples used to evaluate in the prediction step

            
        predict_by_prev: int in {0,1},default=0
        
               if 0, the prediction step is normally performed by extrapolation
               if 1, previous results are used to select the next grid point for xk

        prevNpredict: positive integer, default=1

               only activate if predict_by_prev=1, then we use the average of
               the results by interpolating with the prevNpredict last results
               for selecting the optimal value for the next grid point for xk

        prob_b: float in (0,1), default=0.8

               this is the probability that sets the stopping criterion for
               the prediction step, if we have a probability higher than prob_b
               (by prediction) to have found a grid point that yields an 
                intermediate failure probability in [pl,pu], then we stop and take it


        pl: float in (0,pu), default=0.2

                lower boundary for admissible intermediate probability
                note that (pl+pu)*0.5 as an intermediate probability 
                should refer to the desired grid point selection

        pu: float in (pl,1), default=0.4

                upper boundary for admissible intermediate probability



        #######
        #SuSI parameters optional


        firstvalsampling: int in {0,1}, default=0
            
                decide whether to start with interval search or SuS to
                initialize SuSI, if 0 start by SuS

        firstvalsamplingN: positive integer, default=20
        
                defines the number of samples used for interval sampling,
                only active in use if firstvalsampling=1

        Nbound: positive integer, default=15000

                this is a maximum number of samples used for evaluation of 
                intermediate probabilities, because we adapt the sample
                number so that effort is distributed equally among
                all subsets, depending on their predicted intermediate 
                probability, we can get very high values here if
                the interemdiate probability is close to zero

        raise_err: int in {0,1}, default=1
            
                raise an error if an intermediate probability is zero or 1
                if set 1, an error is raised, otherwise p=1 is set to a value
                close to 1 and if p=0 it is set to a value very close to zero

        fixNpred: positive integer, default=0

                If fixNpred=0, we use only available samples in the given subset
                otherwise we create fixNpred new ones, meaning that we guarantee
                fixNpred samples for prediction (however this is more expensive 
                than fixNpred=0)  

        testmc: int in {0,1}, default=0

                if activated (1) then we only do one Monte Carlo simulation
                to check for failures given a specific xk value

        maxsteps: positive integer, default=5

                maximum steps for the prediction step,
                if there is no satisfying candidate for
                a grid point found after this number of
                preditions we stop anyhow and take the 
                currently best value

        q: float in (0,1), default=0

                quantile of xk selected for first evaluations
                if set zero, then we use max_error to select an
                appropriate value 


        ###################################################################
        returns:

        appends an object of class strurel_result (/main/result)
        to self.results

        for sus, we now have a result by self.results.pfi[-1]
        for susi we need to interpolate next -> method self.susipf


        '''
       
        #clean previous result:
        self.result=strurel_result(method,xk,Nlist,plist,seedplist,palist)

        #add error handling
        #if fixcsteps==0 and seedp*plist[0]*Nlist[0]>
        if fixcsteps==0:
            print("Warning: Using seedp method with {0} seeds, better use fixcsteps=10 for fixed MCMC chain length".format(seedplist[0]*Nlist[0]))

        #if N changing in levels or p0 changing in levels, thus a list and take first
        actual_subset=0
        N=Nlist[actual_subset] 
        p=plist[actual_subset]

        #initial prediction values, if SuS remain zero
        evals_pred=0
        pred_steps=0

        if method=="susi": #update ls w.r.t. xk - for the first we go to a quantile of Xk

            #transformation to and from standard normal space with respect to the dynamic variable xk
            attr_xk=self.attrs.all[np.where(np.array([attri.name for attri in self.attrs.all])==xk)[0]][0]
            rv_xk=attr_xk.get_convert()

            #bound is the stopping pf of conditional failure func. we could also have some other if distribution restricted
            bound=max_error/2.0

            if firstvalsampling!=1:
                #not searching for candidate in advance
                found_cand=0
                if q==0:
                    q=max_error/2.0
                
                if mono=="i":
                    self.ls.det(xk,val=scipy.stats.norm.ppf(1.0-q))
                elif mono=="d":
                    self.ls.det(xk,val=scipy.stats.norm.ppf(q))
            else:
                test3_p=1 #check with small sample numbers where to start
                if mono=="i":
                    val1=0.99
                    val2=1-q  
                else:
                    val1=q
                    val2=0.01  

                self.ls.det(xk,val=scipy.stats.norm.ppf(val1))
                [test1_U_samples_ls_sorted,test1_U_seeds,test1_U_no_fail
                    ,test1_evals,test1_bi,test1_p]=sampling.mc(N=firstvalsamplingN,strurel_obj=self,p=0,bi=bi,bstar=bstar)

                self.ls.det(xk,val=scipy.stats.norm.ppf(val2))
                [test2_U_samples_ls_sorted,test2_U_seeds,test2_U_no_fail
                    ,test2_evals,test2_bi,test2_p]=sampling.mc(N=firstvalsamplingN,strurel_obj=self,p=0,bi=bi,bstar=bstar)  

                endlessloop=0
                while (test3_p<0.8 or test3_p==1) and endlessloop<500: #Nested Intervals until p<test3_p<1, fixed p=0.8 to not lose so much information directly

                    #also set up candidates for prediction step later
                    val3=(val1+val2)/2.0
                    self.ls.det(xk,val=scipy.stats.norm.ppf(val3))   
                    [test3_U_samples_ls_sorted,test3_U_seeds,test3_U_no_fail
                    ,test3_evals,test3_bi,test3_p]=sampling.mc(N=firstvalsamplingN,strurel_obj=self,p=0,bi=bi,bstar=bstar)

                    if mono=="i":
                        if test3_p<p:
                            val1=val3
                            test1_p=test3_p
                        elif test3_p==1:
                            val2=val3
                            test2_p=test3_p
                        else:
                            pass #we are done
                        found_cand=val1
                    else:
                        if test3_p<p:
                            val2=val3
                            test2_p=test3_p
                        elif test3_p==1:
                            val1=val3
                            test1_p=test3_p
                        else:
                            pass #we are done
                        found_cand=val2     

                    endlessloop=endlessloop+1   
                    #if endlessloop>50 and test3_p==0:
                    #    break                           
              
            #add uk value                                        
            self.result.uk.append(self.ls.dets[-1]) 
            #add xk also 
            self.result.xk.append(float(rv_xk.ppf(scipy.stats.norm.cdf(self.result.uk[-1]))))

        if testmc==1: #only test mc
            [U_samples_ls_sorted,U_seeds,U_no_fail,evals,bi,p]=sampling.mc(N=N,strurel_obj=self,p=0,bi=bi,bstar=bstar)
        else:
            [U_samples_ls_sorted,U_seeds,U_no_fail,evals,bi,p]=sampling.mc(N,self,p,bi,bstar=bstar)

        if dropbi==1 or dropbi==2: #do not use samples that have ls equal to threshold (as by definition those always exist)
            U_seeds=U_seeds[np.where(U_seeds[:,-1]<bi)[0]] #in MC this is only one for sure

        if saveU==1: #save U to analyze its properties
            self.result.U.append(U_samples_ls_sorted)

        self.result.add("MC",evals,bi,p,N,evals_pred,pred_steps)
        self.result.gam.append(0.0) # Monte Carlo subset - independent samples

        if testmc==1:
            return(p)

        #starting values for acs (set to best practice values)
        lambda_iter=0.6
        astar=0.44

        #sus #note that susi also utilizes sus here if bi>0 for MC
        while bi>bstar and self.result.pfi[-1]>boundpf: #bi>=0 #in addition boundpf to allow for sus span R cdf

            #actual subset bi and N
            actual_subset=actual_subset+1
            N=Nlist[min(actual_subset,len(Nlist)-1)] 
            p=plist[min(actual_subset,len(plist)-1)]
            seedp=seedplist[min(actual_subset-1,len(seedplist)-1)]
            pa=palist[min(actual_subset-1,len(palist)-1)]

            [U_ls_sorted,U_seeds,U_no_fail,evals,lambda_iter,bi,p,savechains]=sampling.sus(N,self,U_seeds,seedp,p
                                                           ,bi,lambda_iter=lambda_iter,fixcsteps=fixcsteps
                                                           ,pa=pa,reuse=reuse,bstar=bstar,i0=i0,susi=0,xk=xk,vers=vers,dropbi=dropbi)

            self.result.add("sus",evals,bi,p,N,evals_pred,pred_steps)
            if saveU==1: #save U to analyze its properties
                self.result.U.append(U_ls_sorted)
            if method=="susi":    
                #add uk (gridpoints)            
                self.result.uk.append(self.ls.dets[-1])  
                #add xk also 
                self.result.xk.append(float(rv_xk.ppf(scipy.stats.norm.cdf(self.result.uk[-1]))))    

            #compute gamma
            savechains_I=savechains<bi
            Is=savechains_I.astype(np.int)
            Nc1=len(Is[0,:]) #number of chains
            cl1=len(Is[:,0]) #chainlength
            Rk=[  (1.0/(Nc1*cl1-k*Nc1))* np.sum([np.sum([ np.sum([Is[l,j]*Is[l+k,j] for l in range(0,cl1-k)]) for j in range(Nc1) ])])- (p**2)  for k in range(1,cl1)] 
            R0=p*(1-p)
            corrk=[Rk[k]/R0 for k in range(1,len(Rk))]
            corrk=[1.0]+corrk
            gamma=2*np.sum([ (1.0-(k*Nc1)/float(Nc1*cl1))*corrk[k]   for k in range(1,cl1-1)]) #noch nicht ganz richtig irgendwo range(cl1-1---) for k
            self.result.gam.append(gamma)



            #remove seeds that are related to the threshold if dropbi=1 or dropbi=2
            if dropbi==1: #do not use samples that have ls equal to threshold (as by definition those always exist)
                U_seeds=U_seeds[np.where(U_seeds[:,-1]<bi)[0]]
            if dropbi==2:
                drop_bi_vals=[]
                #for case not in chain we also add the threshhold value itself, doubles are ok ignored afterwards anyways as we use threshold
                drop_bi_vals.append(bi)
                #drop in chains related samples
                rem_s_samps=5
                find_ind_bi=np.where(savechains==bi) #2d
                for i in range(len(find_ind_bi[0])):
                    drop_bi_vals=drop_bi_vals+list(savechains[find_ind_bi[0][i]-rem_s_samps:find_ind_bi[0][i],find_ind_bi[1][i]])
                    drop_bi_vals=drop_bi_vals+list(savechains[find_ind_bi[0][i]:find_ind_bi[0][i]+rem_s_samps,find_ind_bi[1][i]])

                #take all seeds that are not in the set of samples related to threshold
                mask=np.isin(U_seeds[:,-1], drop_bi_vals,invert=True) 
                U_seeds=U_seeds[mask]

            if choose_seedN!=0: #can or should set pseeds to 1 then as we control by choose_seedN
                choose_seeds_randomly=random.sample(range(0,len(U_seeds)),choose_seedN)
                U_seeds=U_seeds[choose_seeds_randomly]


        #now sus would be finished as we found a sus failure, but susi continues now to explore the found failure region next
        if method=="susi":

            #end of the list settings for susi
            N=Nlist[len(Nlist)-1] 
            p=plist[len(plist)-1]
            seedp=seedplist[len(seedplist)-1]
            pa=palist[len(palist)-1]
            count_illp=0
            terminate=0

            while self.result.pfi[-1]>bound:

                if self.result.uk[-1]<-8 or self.result.uk[-1]>8: #note that 8 belongs to standard normal space and that we had set 10 if inf in pred_step
                    break

                #could also specify a distribution of xk in advance to have a faster stopping criterion (*condv_b)
                
                #reset N because we change it in each step depending on p_predict
                Np=copy.deepcopy(Nlist[len(Nlist)-1])

                if predict_by_prev==0:
                    [next_point,evals_pred,p_predict,pred_steps]=sampling.predstep(strurel_obj=self
                                          ,Npred=Npred,U_seeds=U_seeds,pl=pl,pu=pu
                                          ,xk=xk,prob_b=prob_b,q=q,mono=mono,fixNpred=fixNpred,bstar=bstar
                                            ,predict_by_prev=predict_by_prev,maxsteps=maxsteps,found_cand=found_cand)
                else: #this should also be utilized for the presampling step!, start with specific xk
                    #predict the  next point by already given knot points from previous results
                    rsp_res=lambda x: np.mean([self.interp(x=x,selec=j,mono=mono) for j in range(max(0,len(self.results)-prevNpredict),len(self.results))])

                    last_xk_pred=float(rv_xk.ppf(rv_xk.cdf(self.result.xk[-1])))
                    #import scipy.optimize as optimize
                    try: # we use bisection which is fast and very stable here
                        if mono=="d":
                            nextp=optimize.bisect(lambda x: 100*(rsp_res(x)/rsp_res(last_xk_pred)-0.5*(pl+pu)), last_xk_pred, rv_xk.ppf(1-max_error),maxiter=10**4)
                        else:
                            nextp=optimize.bisect(lambda x: 100*(rsp_res(x)/rsp_res(last_xk_pred)-0.5*(pl+pu)), rv_xk.ppf(max_error), last_xk_pred,maxiter=10**4) 
                    except: #there is no root in the interval when xk becomes insensitive with respect to changes, then we should just take the extreme value
                        #that corresponds to the admissible max_error
                        if mono=="d":
                            nextp=rv_xk.ppf(1-max_error)
                            terminate=1
                        else:
                            nextp=rv_xk.ppf(max_error)
                            terminate=1

                    [next_point,evals_pred]=[float(scipy.stats.norm.ppf(rv_xk.cdf(nextp))),0]
                    p_predict=0.5*(pl+pu)
                
                if N!=0:
                    N=max(int(Np*(math.log(p_predict)/math.log(plist[len(plist)-1]))),50)
                if N>Nbound:
                    print("error low p predict?")
                    return("High N,low p predict?")

                self.result.uk.append(next_point) 
                #add xk also 
                self.result.xk.append(float(rv_xk.ppf(scipy.stats.norm.cdf(self.result.uk[-1]))))

                [U_ls_sorted,U_seeds,U_no_fail,evals,lambda_iter,bi,p,savechains]=sampling.sus(N,self,U_seeds,seedp,p=0
                                                                ,bi=bstar,lambda_iter=lambda_iter,fixcsteps=fixcsteps
                                                                ,pa=pa,reuse=reuse
                                                                ,bstar=bstar,i0=i0,susi=1,xk=xk,vers=vers,dropbi=dropbi)               

                if saveU==1: #save U to analyze its properties
                    self.result.U.append(U_ls_sorted)

                #cases that might lead to wrong results (should rarely happen but if they do we should neglect the run)
                #if this would happen frequently e.g. p=0 often then we might get a bias
                if raise_err==1 and (p==1 or p==0):
                    raise ValueError("Evaluation returned a non admissible p value: p={0}.If this frequently happens reconsider the parameter settings".format(p))
                if raise_err==0 and (p==1 or p==0):
                    if p==1:
                        p=1.0-max_error  
                        count_illp=count_illp+1           
                        #self.result.add("susi",evals+evals_pred,bi,p,N,evals_pred,pred_steps)
                    if p==0:
                        p=0.0+max_error
                        count_illp=count_illp+1
                        #self.result.add("susi",evals+evals_pred,bi,p,N,evals_pred,pred_steps)
                    #break
                    
                #prediction step and insert according to monotonicity                
                self.result.add("susi",evals+evals_pred,bi,p,N,evals_pred,pred_steps)

                #dropbi==2
                if dropbi==1: #do not use samples that have ls equal to threshold (as by definition those always exist)
                    U_seeds=U_seeds[np.where(U_seeds[:,-1]<bi)[0]]
                if count_illp>3:
                    break

                if terminate==1:
                    print("terminate")
                    break

            self.ls.reset()

        self.result.asnp()
        self.results.append(self.result)
        print("Terminated. Last pfi: {0}" .format(self.result.pfi[-1]))
        return(self.result.pfi[-1])

    def show_res(self,pf): #change here - ACHTUNG ales show_res pcor...
        meval=np.mean([np.sum(self.results_e[i]) for i in range(len(self.results_e))])
        phat=np.array([np.prod(self.results_p[i]) for i in range(len(self.results_p))])
        phatc=np.array([np.prod(pcor(self.results_p[i],N=self.results_N[i])) for i in range(len(self.results_p))])
        meanval=[np.mean(phat),np.mean(phatc)]
        coeffvar=[np.std(phat)/pf,np.std(phatc)/pf]
        return([meval,phat,phatc,meanval,coeffvar])

    def susipf(self,mx,vx,rtype,ftype="interp",domain=[-1e10,1e10],mono="i",stairb="n",selec=-1): #change domain for e.g. lognormals >0
        '''
        ###################################################################
        Description:

        After having calculated a result object by get_result, we can now
        base on this result to derive a failure probability by interpolation.
        This function computes the failure probability for a specific interpolation
        method and a stochastic distribution of the dynamic variable xk.
        We can also choose on which of the calculated results, the interpolation
        shall be based on by adjusting parameter "selec" 

        ###################################################################
        Parameters:
        
        mx: float

            mean value of the dynamic variable

        vx: positive float

            coefficient of variation of the dynamic variable

        rtype: str in {"n","ln","e","u","g"}

            distribution type of the dynamic variable
            {"n":normal,"ln":lognormal,"e":exponential,
            "u"; uniform, "g": Gumbel}


        ftype: str in {"stair","interp"}, default= "interp"
            
            selects the interpolation method, 
            {"stair": staircase approach according to monotonicity "mono" parameter,
            "interp": PCHIP interpolalation with monotonicity in "mono" parameter}

        domain: [float,float], default=[-1e10,1e10]

            domain boundaries for interpolation, values of xk exceeding the domain
            are ignored!

        mono: str in {"i","d"}

              monotonicity of the conditional failure function
              if "i", then we assume the failure probability is 
              increasing if xk is increased
              if "d", then we assume the failure probability is
              decreasing if xk is increased

        stairb: str in {"y", "n"}, default="n"

              if "n", we interpolate normally
              if "y", we can derive boundaries for the result 
              for interpolation method "stair", indeed "y" allows
              to compute the result as if the underlying conditional
              failure probability function was e.g. "i" instead of "d"

        selec: integer, default=-1

              is the index of the result element from self.results
              that is chosen-1 refers to the last computed result 
              in the list of all computed results
        

        ###################################################################
        Returns:

        [result,outxspan_lower,outxspan_upper,x1]

        result: the result by interpolation

        other variables: additional information


        '''

        #added hold for staircase boundaries (no reversing here)
               
        first_susi=self.results[selec].itype.index("susi")

        #remove doubles sus, first order by yvals to know where mc and sus estimates are
        sortedxy=sorted(zip(self.results[selec].xk,self.results[selec].pfi),key=lambda x: x[1])
        [xvals,yvals]=[np.array(sortedxy)[:,0],np.array(sortedxy)[:,1]]
        if "sus" in self.results[selec].itype: #this decision is needed otherwise neglecting first xk
            [xvals,yvals]=[xvals[0:min(-first_susi+1,-1)],yvals[0:min(-first_susi+1,-1)]] #min with -1 if =1 then 0:0        
        else:
            pass #all xk used

        #increasing order in xvals
        if mono=="d": #reverse order
            [xvals,yvals]=[xvals[::-1],yvals[::-1]] 
        
        condv_rv=props.attr(name="check_xk",rtype=rtype,mx=mx,vx=vx).get_convert()   
        f_lower=condv_rv.cdf_b(b1=domain[0],b2=xvals[0])
        f_upper=condv_rv.cdf_b(b1=xvals[-1],b2=domain[-1])

        if stairb=="n": #normal stair approximation, not for the boundary
            if mono=="i":
                outxspan_lower=f_lower*yvals[0]
                outxspan_upper=f_upper*1.0
            else:
                outxspan_lower=f_lower*1.0
                outxspan_upper=f_upper*yvals[-1]
        else: #act as if mono but is boundary
            if mono=="i":
                outxspan_lower=f_lower*yvals[0]
                outxspan_upper=f_upper*0.0
            else:
                outxspan_lower=f_lower*0.0
                outxspan_upper=f_upper*yvals[-1]            

        #original order
        if mono=="d": #reverse order
            [xvals,yvals]=[xvals[::-1],yvals[::-1]] 

        if ftype=="interp":
            if mono=="i":
                xspan_result=scipy.integrate.quad(lambda x: max(0.0,self.interp(x,selec,mono=mono))*condv_rv.pdf(x),max(xvals[0],domain[0]),min(xvals[-1],domain[1]))[0]
            else:
                xspan_result=scipy.integrate.quad(lambda x: max(0.0,self.interp(x,selec,mono=mono))*condv_rv.pdf(x),max(xvals[-1],domain[0]),min(xvals[0],domain[1]))[0]
            x1=0
        elif ftype=="stair":         
            if mono=="i":
                #if hold="y": reverse same below
                xspan_result=np.sum([condv_rv.cdf_b(xvals[i],xvals[i+1])*yvals[i+1] for i in range(len(xvals)-1)])
                x1=[condv_rv.cdf_b(xvals[i],xvals[i+1])*yvals[i+1] for i in range(len(xvals)-1)]
            else:
                #above: ordered according to x, but as we want to have a boundary for not really decreasing/increasing functions resp.   
                #[xvals,yvals]=[xvals[::-1],yvals[::-1]]
                xspan_result=np.sum([condv_rv.cdf_b(xvals[i],xvals[i+1])*yvals[i] for i in range(len(xvals)-1)])    
                x1=[condv_rv.cdf_b(xvals[i],xvals[i+1])*yvals[i] for i in range(len(xvals)-1)]                 
     
        result=outxspan_lower+xspan_result+outxspan_upper
         
        return([result,outxspan_lower,outxspan_upper,x1])


    def interp(self,x,selec,mono="i"):
        '''
        ###################################################################
        #description:

        Compute the function value by PCHIP (monotone cubic interpolation) 
        at a specific point x, based on the last result (selec=-1) or any other (selec=?)
        The monotonicity of the conditional failure probability function q 
        with respect x has to be given, mono="i" or mono="d",
        this supports easy implementation of the boundaries for worst case evaluations
        according to the principle of safety

        ###################################################################
        parameters: x, selec, mono

        x: point of the dynamic variable Xk which has to be evaluated

        selec: select the result that is used for creating the interpolation function
                e.g. -1 refers to the last result in self.results

        mono: monotonicity of the conditional failure probability with respect
               values of the dynamic variable Xk  

        ###################################################################
        returns:

        function value at x by interpolaiton according to safety principle

        '''

        #find the first grid point - sus repetitively uses the same grid points, susi points are unique grid points
        first_susi=self.results[selec].itype.index("susi")

        #remove doubles by sus, first order by yvals
        sortedxy=sorted(zip(self.results[selec].xk,self.results[selec].pfi),key=lambda x: x[1])
        [xvals,yvals]=[np.array(sortedxy)[:,0],np.array(sortedxy)[:,1]]
        if "sus" in self.results[selec].itype:
            [xvals,yvals]=[xvals[0:min(-first_susi+1,-1)],yvals[0:min(-first_susi+1,-1)]]     
        else:
            pass #if no sus performed, use all xk for interpolation, starting at the MC subset   

        #increasing order in xvals
        if mono=="i": #correct order
            pass 
        else: #reverse order
            [xvals,yvals]=[xvals[::-1],yvals[::-1]] 

        #interpolation by PCHIP, difference with respect to monotonicity at boundary values
        if mono=="i":       
            if x<=xvals[-1] and x >=xvals[0]: 
                interp=scipy.interpolate.PchipInterpolator(xvals, yvals)
                return(max(0.0,interp(x)))           
            elif x < xvals[0]: 
                return(yvals[0])
            else:
                return(1.0)
        else:
            if x<=xvals[-1] and x >= xvals[0]:
                interp=scipy.interpolate.PchipInterpolator(xvals, yvals)
                return(max(0.0,interp(x)))
            elif x < xvals[0]:
                return(1.0)
            else:
                return(yvals[len(yvals)-1])      

    def regspline(self,n_knots=100,boundary_dist=5,smooth_k=3,startres=0,endres=None):
        '''
        ###################################################################
        description:

        use regression spline based on all results and datapoints for creation [spline,datapoints]

        ###################################################################
        parameters:
            see function self.regresult

        ###################################################################
        returns:

            a spline object which will be used for integration

        '''

        if endres==None:
            endres=len(self.results)

        #get data points for regression spline by simulation results
        datapoints=[[float(self.results[-i].xk[j]),float(self.results[-i].pfi[j])] if j<len(self.results[-i].xk) else [-1,0] for i,j in itertools.product(range(startres,endres),range(0,200)) ]

        #for error handling we had [-1,0] defaults, remove them next
        datapoints=list(filter(lambda elt: elt != [-1,0], datapoints))

        #unique elements
        datapoints=list(set(map(tuple,datapoints)))
        #get array
        datapoints=np.array(datapoints)
        #sort
        datapoints.sort(axis=0)

        #get unique elements (missing: and average function value of doubles)
        datapoints2=np.zeros((len(datapoints),2))
        datapoints2[0]=datapoints[0]
        j=1
        for i in range(1,len(datapoints)):
            avgval=[]
            if np.isclose(datapoints[i,0],datapoints2[j-1,0])==False:
                datapoints2[j]=datapoints[i]
                j=j+1
                avgval.append(datapoints[i,1])
            else:
                avgval.append(datapoints[i,1])

            #set the value equal to the average of its appearances for multiple appearances
            datapoints2[j-1,1]=np.mean(avgval) 

        datapoints2=datapoints2[:np.where(datapoints2[2:,0]==0)[0][0]] #start at 2 because there are cases where the first value is zero

        #define knots, total number of knots given, need to provide a good tradeoff for smoothing and accuracy, indeed good choice depends on pf
        lend=len(datapoints2)
        spacex=max(int(lend/(n_knots+2)),1) #+2 because we leave out last and first knot later to avoid violation of outer point choosing

        #define knots
        interior_knots_spline=datapoints2[boundary_dist:-boundary_dist:spacex,0]

        #define smoothing spline
        rsp=LSQUnivariateSpline(datapoints2[:,0], datapoints2[:,1],t=interior_knots_spline[1:-1],k=smooth_k)
        return([rsp,datapoints2])

    def regresult(self,mx,vx,rtype,n_knots=100,boundary_dist=5,smooth_k=3,startres=0,endres=None):
        '''
        ###################################################################
        Description:

        Compute the failure probability by regression, applying smoothing splines 
        on the data points of several susi results
        Most derivations are performed in function self.regspline 

        ###################################################################
        Parameters:

        mx: float

            mean value of the dynamic variable

        vx: positive float

            coefficient of variation of the dynamic variable

        rtype: str in {"n","ln","e","u","g"}

            distribution type of the dynamic variable
            {"n":normal,"ln":lognormal,"e":exponential,
            "u"; uniform, "g": Gumbel}

        n_knots: positive integerm default=100

            number of knots for building the smoothing spline for regression
    
        boundary_dist: positive integer, default=5

            do avoid taking knots right at the end points, select distance

        smooth_k: k in LSQUnivariateSpline

        startres: integer, default=0

            select index of first result in self.results that is used for regression

        endres: integer

            select index of last result in self.results that is used for regression
            
            e.g.
                startres+10 means we use the grid points of 10 results

        ###################################################################
        returns:

        estimated failure probability by regression    
        

        '''
        condv_rv=props.attr(name="check_xk",rtype=rtype,mx=mx,vx=vx).get_convert()   
        if endres==None:
            endres=len(self.results)
        [rsp,datapoints]=self.regspline(n_knots,boundary_dist,smooth_k,startres,endres)
        r1=scipy.integrate.quad(lambda x: max(0.0,rsp(x))*condv_rv.pdf(x),datapoints[0,0],datapoints[-1,0])
        return(r1[0])

    def info(self):
        print("--------limit state--------------")
        print(self.ls.info())
        print("--------attr collection----------")
        print(self.attrs.info())
        print("-------- rvs ------------")
        print(self.rvs.info())
        print("-------- corrX------------")
        print(self.corrX)
        print("--------match ls attr------------")        
        print(self.match)








