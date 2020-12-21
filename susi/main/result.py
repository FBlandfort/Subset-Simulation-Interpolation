

import numpy as np





class result_obj(object):
    '''
    a dummy class to assign non formal results 
    where we want to have attributes for a more clear expression
    '''
    pass



class strurel_result(object):
    '''
    this class is created to save all results of the main class "strurel" 
    provided by the function get_result (either method "sus" or "susi" in 
    an easy to handle manner, many characteristics are saved to get a full
    view on the results

    furthermore, it allows to define some methods on the result object
    that help at printing information of the result, adding new calculations
    by simulation to the result and convert lists to arrays at once

	################
	#in particular:

		.method
		.xkname

		.uk:
		.xk:
		.bi
		.pi
		.pfi:

		.evals
		.evals_pred
		.Nreal

		.pred_steps
		.gam
		.U
		.palist
		.seedplist


    '''
    def __init__(self,method,xk,Nlist,plist,seedplist,palist):
        self.xk=list()
        self.uk=list()
        self.xkname=xk
        self.method=method
        self.pi=list()
        self.evals=list()
        self.evals_pred=list()
        self.pred_steps=list()
        self.Nlist=Nlist
        self.plist=plist
        self.seedplist=seedplist
        self.palist=palist
        self.itype=list()
        self.bi=list()
        self.pfi=list()
        self.U=list()
        #self.X=list() #add later for illustration of transformation between spaces 
        self.gam=list()
        self.Nreal=list()
    def add(self,itype,evals,bi,p,N,evals_pred,pred_steps):
        self.itype.append(itype)
        self.evals.append(evals)
        self.evals_pred.append(evals_pred)
        self.pred_steps.append(pred_steps)
        self.bi.append(bi)
        self.pi.append(p)
        self.Nreal.append(N)
        if itype=="MC":
            self.pfi.append(p)      
        else:
            self.pfi.append(p*self.pfi[-1])
    def asnp(self):
        self.xk=np.array(self.xk).astype("float")
        self.uk=np.array(self.uk).astype("float")
        self.pfi=np.array(self.pfi).astype("float")
        self.pi=np.array(self.pi).astype("float")
        self.bi=np.array(self.bi).astype("float")
        self.evals=np.array(self.evals).astype("float")
    def info(self):
        print("-- method: {0}, Nlist: {1}, plist: {2}, seedplist: {3}, palist: {4}, xkname: {5} --".format(self.method,self.Nlist,self.plist,self.seedplist,self.palist,self.xkname))
        print("-------- uk: ----------")
        print(self.uk)
        print("-------- xk: ----------")
        print(self.xk)
        print("-------- itype: ----------")
        print(self.itype)
        print("-------- evals: ----------")
        print(self.evals)
        print("-------- bi: ----------")
        print(self.bi)
        print("-------- pfi: ----------")
        print(self.pfi)
        print("-------- Nreal: ----------")
        print(self.Nreal)
        print("-------- intermediate pi: ----------")
        print(np.append(self.pfi[0],np.array([self.pfi[i+1]/self.pfi[i] for i in range(len(self.pfi)-1)])))
        print("-------- pred_steps: ----------")
        print(self.pred_steps)
        print("-------- evals_pred: ----------")
        print(self.evals_pred)












