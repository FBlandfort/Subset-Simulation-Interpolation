
import numpy as np

from scipy.optimize import broyden1
import math
euler_mascheroni=0.5772156649015328606065120


##todo: add function to make input easier for creating a limit state function

#######attribute handling and limitstate match


def convert_inputs(mx,vx,rtype): 
    '''
    inputs:

    mx, vx, rtype: see attr class:
    mean, relative std, random variable type

    KEEP CONSISTENT UNITS (cm^2, m^2 , N, kN etc)
    --------------------------------------------
    outputs:

    (loc,scale) or (scale,s --for ln), see scipy.stats package

    converts the input variables according to their rtype
    to a parameter form so that they keep the same mx and vx
    if putting the converts as parameters in the scipy.stats package
    '''
    if rtype=="n":
        loc=mx
        scale=vx*abs(mx)
        return(loc,scale)
    elif rtype=="ln":
        def f(s):
            return(math.sqrt((math.exp(s**2)-1)*math.exp(2*(math.log(mx)-(s**2)/2.0)+s**2))-vx*mx)    
        s=float(broyden1(f, [0.5], f_tol=mx*1e-13))  ##was fixed 13 -> but relatively makes more sense
        scale=float(math.log(mx)-(s**2)/2.0)
        return(scale,s)
    elif rtype=="g":
        scale=float(math.sqrt(((((vx*mx)**2)*6)/(math.pi**2))))#beta
        loc=float(mx-(euler_mascheroni*scale))
        return(loc,scale)
    elif rtype=="e":
        loc=mx-mx*vx
        scale=mx*vx
        return(loc,scale)
    elif rtype=="u":
        loc=mx
        scale=vx
        return(loc,scale)
    else:
        return("rtype not found")
        





def limitstate_match(col_attr0,argslist):

    '''
    input:

    col_attr0:  col_attr(attrs=[attr(name="f_c",rtype="ln",mx=3.8, vx=0.13),attr(....])
    argslist: parameters of limit state function in their argument order appearance
    ----------------------------------------------------------------------------------
    output:
    
    matched: array with indices so that col_attr0[indices] has the same order as argslist
    '''
    matched=[]
    for r in range(len(col_attr0.all)):
        for a in range(len(argslist)):
            if col_attr0.all[r].name==argslist[a]:
                matched.append(a)
                break
    matched=np.array(matched)
    if len(matched)!=len(argslist):
        return("error matched length")
    else:
        return(matched)

























