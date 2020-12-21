


#################################################################################
#description:
'''
collection of examples for application of sus or susi;
these examples create strurel objects, which allow to apply sus or susi
with respect to the considered limit states and attributes
'''


from .. import props
from .. import main


#################################################################################
#Example 1:
'''
sum of standard normally distributed random variables
with probability of failure pf= 1e-06 (-> norm.cdf(-4.7534243088229)=1e-06)
'''
example1=main.strurel(


    ls=props.limitstate(
    name="ls1"
    ,f_str='-(np.sum([x[0],x[1],x[2]]))/math.sqrt(3)+4.7534243088229'    
    ,argslist=["x2","x0","x1"]
    ,prob=None)

    ,attrs=props.col_attr([props.attr(name="x{0}".format(i),rtype="n",mx=1e-10,vx=1e10) for i in range(3)])
 
    )



#################################################################################
#Example 2:
'''
sum of standard normally distributed random variables
with probability of failure pf= 5e-04  (-> norm.cdf(-3.2905267314918945)=5e-04)
'''
example2=main.strurel(

    ls=props.limitstate(
    name="ls1"
    ,f_str='-(np.sum([x[0],x[1],x[2]]))/math.sqrt(3)+3.2905267314918945'   
    ,argslist=["x2","x0","x1"]
    ,prob=None)

    ,attrs=props.col_attr([props.attr(name="x{0}".format(i),rtype="n",mx=1e-10,vx=1e10) for i in range(3)])
 
    )

    













