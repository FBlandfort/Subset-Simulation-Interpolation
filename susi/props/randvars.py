


from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import gumbel_r
from scipy.stats import expon
from scipy.stats import uniform as uniform_r
import math
import numpy as np

from susi.props import nataf

class rv(object): #types: n, ln, g, u , e
    def __init__(self, rtype, args,name="default"):
        '''
        rv(rtype*, args*)
        rtype* is the distribution type "n", "ln", "u", "g" and
        args* is vector with distribution parameters
        in the following way:
        ------------------------------------------------------------
        normal (n): [m, s**2]    !variance!
        lognormal (ln): [m(n), s**2(n)]       !variance!
        uniform (u): [lower border, upper border]
        gumbel (g): [loc, scale]
        ------------------------------------------------------------
        letter in brackets is the corresponding shortcut for "rtype"
        ------------------------------------------------------------
        ------------------------------------------------------------
        rv
        properties: "args", "rtype", "sd", "m"
        methods: "sample"
        '''
        self.rtype=rtype
        self.args=args
        self.name=name
        if rtype=="n":
            self.mean=args[0]
            self.sd=args[1]
        elif rtype=="ln":
            self.mean=lognorm(s=args[1],scale=math.exp(args[0])).mean()
            self.sd=lognorm(s=args[1],scale=math.exp(args[0])).std()
        elif rtype=="g": #updated gumbel by scipy.stats. std/mean
            self.mean=gumbel_r(loc=args[0],scale=args[1]).mean()
            self.sd=gumbel_r(loc=args[0],scale=args[1]).std()
        elif rtype=="u":
            self.mean=uniform_r(loc=args[0],scale=args[1]).mean()
            self.sd=uniform_r(loc=args[0],scale=args[1]).std()
        elif rtype=="e":
            self.mean=expon(loc=args[0],scale=args[1]).mean()
            self.sd=expon(loc=args[0],scale=args[1]).std()
        else:
            print("distribution {0} not found" .format(rtype))
            return("error - distribution")
    def sample(self,n):
        if self.rtype=="n":
            return(norm(loc=self.args[0],scale=self.args[1]).rvs(n))
        elif self.rtype=="ln":
            return(lognorm(s=self.args[1],scale=math.exp(self.args[0])).rvs(n))
        elif self.rtype=="g":
            return(gumbel_r(loc=self.args[0],scale=self.args[1]).rvs(n))
        elif self.rtype=="e":
            return(expon(loc=self.args[0],scale=self.args[1]).rvs(n))
            pass
        elif self.rtype=="u":
            return(uniform_r(loc=self.args[0],scale=self.args[1]).rvs(n))
        else:
            print("distribution {0} not found" .format(rtype))
            return("error - distribution")
        return(samples)
    def cdf_b(self,b1,b2):
        if b1>b2:
            saveb=b2
            b2=b1
            b1=saveb
        if self.rtype=="n":
            return(norm(loc=self.args[0],scale=self.args[1]).cdf(b2)-norm(loc=self.args[0],scale=self.args[1]).cdf(b1))
        elif self.rtype=="ln":
            return(lognorm(s=self.args[1],scale=math.exp(self.args[0])).cdf(b2)-lognorm(s=self.args[1],scale=math.exp(self.args[0])).cdf(b1))
        elif self.rtype=="g":
            return(gumbel_r(loc=self.args[0],scale=self.args[1]).cdf(b2)-gumbel_r(loc=self.args[0],scale=self.args[1]).cdf(b1))
        elif self.rtype=="e":
            return(expon(loc=self.args[0],scale=self.args[1]).cdf(b2)-expon(loc=self.args[0],scale=self.args[1]).cdf(b1))
        elif self.rtype=="u":
            return(uniform_r(loc=self.args[0],scale=self.args[1]).cdf(b2)-uniform_r(loc=self.args[0],scale=self.args[1]).cdf(b1))
        else:
            print("distribution {0} not found" .format(rtype))
            return("error - distribution")
    def ppf(self,p):
        if self.rtype=="n":
            return(norm(loc=self.args[0],scale=self.args[1]).ppf(p))
        elif self.rtype=="ln":
            return(lognorm(s=self.args[1],scale=math.exp(self.args[0])).ppf(p))
        elif self.rtype=="g":
            return(gumbel_r(loc=self.args[0],scale=self.args[1]).ppf(p))
        elif self.rtype=="e":
            return(expon(loc=self.args[0],scale=self.args[1]).ppf(p))
        elif self.rtype=="u":
            return(uniform_r(loc=self.args[0],scale=self.args[1]).ppf(p))
        else:
            print("distribution {0} not found" .format(rtype))
            return("error - distribution")     
    def pdf(self,x):  
        if self.rtype=="n":
            return(norm(loc=self.args[0],scale=self.args[1]).pdf(x))
        elif self.rtype=="ln":
            return(lognorm(s=self.args[1],scale=math.exp(self.args[0])).pdf(x))
        elif self.rtype=="g":
            return(gumbel_r(loc=self.args[0],scale=self.args[1]).pdf(x))
        elif self.rtype=="e":
            return(expon(loc=self.args[0],scale=self.args[1]).pdf(x))
        elif self.rtype=="u":
            return(uniform_r(loc=self.args[0],scale=self.args[1]).pdf(x))
        else:
            print("distribution {0} not found" .format(rtype))
            return("error - distribution")    
    def cdf(self,x):  
        if self.rtype=="n":
            return(norm(loc=self.args[0],scale=self.args[1]).cdf(x))
        elif self.rtype=="ln":
            return(lognorm(s=self.args[1],scale=math.exp(self.args[0])).cdf(x))
        elif self.rtype=="g":
            return(gumbel_r(loc=self.args[0],scale=self.args[1]).cdf(x))
        elif self.rtype=="e":
            return(expon(loc=self.args[0],scale=self.args[1]).cdf(x))
        elif self.rtype=="u":
            return(uniform_r(loc=self.args[0],scale=self.args[1]).cdf(x))
        else:
            print("distribution {0} not found" .format(rtype))
            return("error - distribution")                         
    def info(self):
        print("name    type      args    ")
        print([self.name,self.rtype,self.args])


class col_rv(object):
    '''
    rv(rvs,corr) - rv collection
    random variable vector rvs
    correlation matrix given in corrX
    '''
    def __init__(self,rvs,corrX=None):
        self.all=np.array(rvs)
        if corrX is None:
            self.corrX=np.matrix(np.eye(len(self.all)))
        else:
            self.corrX=corrX
    def get_cz_col(self):
        cz=np.array(np.eye(N=len(self.all),M=len(self.all)))
        for n in range(1,len(self.all)):
            for m in range(0,n):
                cz[n,m]=nataf.get_cz(xi=self.all[n],xj=self.all[m],cx=self.corrX[n,m])
                cz[m,n]=cz[n,m]
        return(cz)
    def info(self):
        print("name    type      args    ")
        for i in range(len(self.all)):
            print([self.all[i].name,self.all[i].rtype,self.all[i].args])
        return("--INFO--")        


