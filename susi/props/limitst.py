



import numpy as np
import math
import copy

from scipy.stats import norm
from susi.props import nataf






#####################    limit states  ##############################

class limitstate(object):
    '''
    .info() for information about created objects of the limitstate class

    #parameters:

    name: name of the object, not relevant but good for documentation

    f_str: limit state function of x[0],x[1],... given as a str,
            e.g. "x[0]+x[1]"

    argslist: arguments of the function, !important: same order as in col_attr
            of the structure to be analyzed

    prob: corresponding structure, given as strurel object


    '''
    def __init__(self,name,f_str,argslist,prob=None):
        self.name=name
        self.f_str=f_str #keeps the original form saved
        self.f_str2=copy.copy(f_str) #str to work with
        self.argslist=argslist
        self.dets=list() #empty list, nothing deterministic at first, set deterministic values by function .det
        self.f=lambda x: eval(self.f_str2)
        self.prob=prob
    def reset(self):
        self.f_str2=self.f_str
    def det(self,arg,val,keep=0):
        argindex=self.argslist.index(arg)
        #print("Changing variable *{0}* to deterministic value *{1}*".format(self.argslist[argindex],val))
        attr_xk=self.prob.attrs.all[np.where(np.array([attri.name for attri in self.prob.attrs.all])==arg)[0]][0]
        rv_xk=attr_xk.get_convert()
        if keep==0:       
            self.f_str2=self.f_str.replace("x[{0}]".format(argindex),str(float(rv_xk.ppf(norm.cdf(val))))) # here we also switch to original random variable
            self.dets=[argindex,val] # this sets dets in standard normal space
            #print("Deterministic argument: {0}" .format(self.argslist[argindex]))
        #NOT YET DONE detvalidx for several...
        else: #wish to successively replace several variables with deterministic values
            #ACHTUNG wenn vorher mit keep0 schon etwas gemacht wurde wird es problematisch
            if self.f_str2!=self.f_str:
                print("""Warning: In case you already used .det() before, 
                then you have to reset first by typing #lsname#.reset()
                - otherwise the previous settings will be kept (note: you might want this)!""")
            self.f_str2=self.f_str2.replace("x[{0}]".format(argindex),str(val))
        self.f=lambda x: eval(self.f_str2)     
        return("--done--")   
    def gfun(self,U):
        result_nataf=nataf.nataf(self.prob,U,self.prob.corr_z)
        self.prob.corr_z=result_nataf.corr_z        
        gfun=lambda x: eval(self.f_str2)
        return(gfun(result_nataf.X))             
    def info(self):
        print('name: {0}'.format(self.name))
        print('argslist: {0}'.format(self.argslist))
        print('func0: {0}'.format(self.f_str))
        print('func: {0}'.format(self.f_str2))

class limit_state_list(object):
    '''
    TODO: list of limitstate functions
          to allow joint considerations,
          on the other hand this can also be achieved
          by packing the joint function into one
          normal limit state function object
    '''
    def __init__(self,name):
        self.name=name
        self.ls=[]
    def new(self,name_ls,arglist):
        #self.ls.append()
        pass
