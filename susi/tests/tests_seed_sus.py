


import unittest

import numpy as np
import random
import math

import susi


class main_test_attr(unittest.TestCase):

    #####################################################################
    ###### col1 						#################################
 
    ### sus ####

    def test_example1_sus(self):
        random.seed(1)
        np.random.seed(1)
        susi.examples.coll1.example1.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="a")
        pfh=susi.examples.coll1.example1.result.pfi[-1]
        self.assertAlmostEqual(pfh, 6e-07)
        
    def test_example2_sus(self):
        random.seed(1)
        np.random.seed(1)
        susi.examples.coll1.example2.get_result(method="sus",plist=[0.1],Nlist=[200],palist=[0.2],seedplist=[1.0],reuse=1,vers="b")
        pfh=susi.examples.coll1.example2.result.pfi[-1]
        self.assertAlmostEqual(pfh, 4.5e-04)






if __name__=="__main__":
    unittest.main()
















