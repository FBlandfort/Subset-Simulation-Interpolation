

from .. import props
from .. import main
from scipy.stats import norm
from scipy.stats import gamma

#################################################################################
#Example 1:
'''
sum of standard normally distributed random variables
with probability of failure pf= 1e-06 (-> norm.cdf(-4.7534243088229)=1e-06)
'''


def create_ex1(dim_d,pf):
    '''
    mono="i"
    '''    
    str1=str(['x[{0}]'.format(i) for i in range(dim_d)]).replace("'","")
    str2=["x{0}".format(i) for i in range(dim_d)]
        
    ex1=main.strurel(

        ls=props.limitstate(
        name="ex1"+"_dim:{0}".format(dim_d)+"_pf:{0}".format(pf)

        ,f_str="-(np.sum({0}))/math.sqrt({1})+{2}".format(str1,dim_d,-norm.ppf(pf))    
        ,argslist=str2
        ,prob=None)

        ,attrs=props.col_attr([props.attr(name="x{0}".format(i),rtype="n",mx=1e-10,vx=1e10) for i in range(dim_d)])
     
        )

    return(ex1)


def create_ex2a(dim_d,pf):
    '''
    mono="i"
    '''
    str1=str(['x[{0}]'.format(i) for i in range(dim_d)]).replace("'","")
    str2=["x{0}".format(i) for i in range(dim_d)]
        
    ex2a=main.strurel(

        ls=props.limitstate(
        name="ex2a"+"_dim:{0}".format(dim_d)+"_pf:{0}".format(pf)

        ,f_str="-(np.sum({0}))+{1}".format(str1,gamma(dim_d,scale=1).ppf(1-pf))    
        ,argslist=str2
        ,prob=None)

        ,attrs=props.col_attr([props.attr(name="x{0}".format(i),rtype="e",mx=1,vx=1) for i in range(dim_d)])
        
        )

    return(ex2a)
    
    
def create_ex2b(dim_d,pf):
    '''
    mono="d"
    '''
    str1=str(['x[{0}]'.format(i) for i in range(dim_d)]).replace("'","")
    str2=["x{0}".format(i) for i in range(dim_d)]
        
    ex2b=main.strurel(

        ls=props.limitstate(
        name="ex2b"+"_dim:{0}".format(dim_d)+"_pf:{0}".format(pf)

        ,f_str="(np.sum({0}))-{1}".format(str1,gamma(dim_d,scale=1).ppf(pf))    
        ,argslist=str2
        ,prob=None)

        ,attrs=props.col_attr([props.attr(name="x{0}".format(i),rtype="e",mx=1,vx=1) for i in range(dim_d)])
     
        )

    return(ex2b)

    

"""

##1
%autoreload
a1=susi.examples.coll2.create_ex1(5,1e-06)
a1.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="a")

a1.get_result(method="susi",xk="x0",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="a")
a1.susipf(1e-10,1e10,"n",ftype="interp",mono="i",stairb="n",selec=-1)[0]

susi.examples.coll1.example1.info()


##2a
a1=susi.examples.coll2.create_ex2a(5,1e-06)
a1.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="a")


##2b
a1=susi.examples.coll2.create_ex2b(5,1e-06)
a1.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="a")



"""





