


import numpy as np
from susi.props import randvars
from susi.props import funcs

#####################    attributes      ############################ 


class attr(object):
    '''
    properties:

    name: name of attribute, choose as parameter name in limit state function
    rtype: type of probability distribution (n,ln,g,u), if deterministic set to "d"
    mx: expected value of the property
    vx: relative standard deviation:  std/mx
        --> for rtype "u": [mx,vx] as borders

    methods:
    .get_convert()  --> get corresponding random variables for attribute
    .info()  -> print information

    '''
    def __init__(self,name,rtype,mx,vx=-1):
        #vx predefined -> no error for deterministic
        self.name=name
        self.rtype=rtype 
        self.mx=mx
        self.vx=vx
    def get_convert(self):
        converted=funcs.convert_inputs(mx=self.mx, vx=self.vx, rtype=self.rtype)
        converted_rv=randvars.rv(rtype=self.rtype, args=[converted[0],converted[1]], name=self.name)
        return(converted_rv)
    def info(self):
        print("name    type      mx       vx     ")
        print([self.name,self.rtype,self.mx, self.vx])    
        return("--INFO--")




class col_attr(object):
    '''
    collection of attributes
    
    input: attributes in a vector
    
    properties:
    .all              all attributes
    .rvs              correpsonding random variables


    methods:
    .get_rvs()
    .info(): prints all attribute infos for each attribute
            name - type - mx - vx 
    '''
    def __init__(self,attrs):
        self.all=np.array(attrs)
        self.rvs=randvars.col_rv(rvs=self.get_rvs())  #,corr=corr
    def get_rvs(self):
        rvs=[]
        for r in range(len(self.all)):
            converts=funcs.convert_inputs(mx=self.all[r].mx, vx=self.all[r].vx, rtype=self.all[r].rtype)   
            rvs.append(randvars.rv(rtype=self.all[r].rtype, args=[converts[0],converts[1]], name=self.all[r].name))
        return(rvs)      
    def info(self):
        print("name    type      mx       vx  ")
        for i in range(len(self.all)):
            print([self.all[i].name,self.all[i].rtype,self.all[i].mx, self.all[i].vx])
        return("--INFO--")

#####################################################################




