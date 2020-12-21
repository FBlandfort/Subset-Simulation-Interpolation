


import unittest

import numpy as np
import random
import math

import susi


class main_test_attr(unittest.TestCase):

    #####################################################################
    ###### col1 						#################################

    print("###Note:\nDue to the stochastic nature of the tests," 
            +" errors might appear sometimes.\nRepetitive runs give best information \n###")

    ### susi ####

    def test_example1_susi(self):
        susi.examples.coll1.example1.get_result(method="susi",xk="x0",mono="i",max_error=1e-8)
        pfh=susi.examples.coll1.example1.susipf(1e-10,1e10,"n",ftype="interp",mono="i",stairb="n",selec=-1)[0]
        pfh_upper=susi.examples.coll1.example1.susipf(1e-10,1e10,"n",ftype="stair",mono="i",stairb="n",selec=-1)[0]
        pfh_lower=susi.examples.coll1.example1.susipf(1e-10,1e10,"n",ftype="stair",mono="d",stairb="y",selec=-1)[0]
        print([pfh_lower,pfh,pfh_upper,"True={0}".format(1e-06)])
        self.assertGreaterEqual(pfh, 5e-07)
        self.assertLessEqual(pfh, 5e-06)
        self.assertGreaterEqual(pfh, pfh_lower)
        self.assertLessEqual(pfh, pfh_upper)
        
    def test_example2_susi(self):
        susi.examples.coll1.example2.get_result(method="susi",xk="x0",mono="i",max_error=5e-6)
        pfh=susi.examples.coll1.example2.susipf(1e-10,1e10,"n",ftype="interp",mono="i",stairb="n",selec=-1)[0]
        pfh_upper=susi.examples.coll1.example2.susipf(1e-10,1e10,"n",ftype="stair",mono="i",stairb="n",selec=-1)[0]
        pfh_lower=susi.examples.coll1.example2.susipf(1e-10,1e10,"n",ftype="stair",mono="d",stairb="y",selec=-1)[0]
        print([pfh_lower,pfh,pfh_upper,"True={0}".format(5e-04)])
        self.assertGreaterEqual(pfh, 1e-04)
        self.assertLessEqual(pfh, 1e-03)
        self.assertGreaterEqual(pfh, pfh_lower)
        self.assertLessEqual(pfh, pfh_upper)



if __name__=="__main__":
    unittest.main()
















