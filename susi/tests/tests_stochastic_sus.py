


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

    ### sus  ####

    def test_example1_sus(self):
        susi.examples.coll1.example1.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="a")
        pfh=susi.examples.coll1.example1.result.pfi[-1]
        self.assertGreaterEqual(pfh, 5e-07)
        self.assertLessEqual(pfh, 5e-06)
        
    def test_example2_sus(self):
        susi.examples.coll1.example2.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="b")
        pfh=susi.examples.coll1.example2.result.pfi[-1]
        self.assertGreaterEqual(pfh, 1e-04)
        self.assertLessEqual(pfh, 1e-03)






if __name__=="__main__":
    unittest.main()
















