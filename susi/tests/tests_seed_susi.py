


import unittest

import numpy as np
import random
import math

import susi


class main_test_attr(unittest.TestCase):

    #####################################################################
    ###### col1 						#################################

    ### susi ####

    def test_example1_susi(self):
        random.seed(1)
        np.random.seed(1)
        susi.examples.coll1.example1.get_result(method="susi",xk="x0",mono="i",max_error=1e-8)
        pfh=susi.examples.coll1.example1.susipf(1e-10,1e10,"n",ftype="interp",mono="i",stairb="n",selec=-1)[0]
        pfh_upper=susi.examples.coll1.example1.susipf(1e-10,1e10,"n",ftype="stair",mono="i",stairb="n",selec=-1)[0]
        pfh_lower=susi.examples.coll1.example1.susipf(1e-10,1e10,"n",ftype="stair",mono="d",stairb="y",selec=-1)[0]
        self.assertAlmostEqual(pfh,1.1809263613787379e-06)
        self.assertAlmostEqual(pfh_lower,7.237347672145986e-07)
        self.assertAlmostEqual(pfh_upper,2.0851093313958585e-06)
        
    def test_example2_susi(self):
        random.seed(1)
        np.random.seed(1)
        susi.examples.coll1.example2.get_result(method="susi",xk="x0",mono="i",max_error=5e-6)
        pfh=susi.examples.coll1.example2.susipf(1e-10,1e10,"n",ftype="interp",mono="i",stairb="n",selec=-1)[0]
        pfh_upper=susi.examples.coll1.example2.susipf(1e-10,1e10,"n",ftype="stair",mono="i",stairb="n",selec=-1)[0]
        pfh_lower=susi.examples.coll1.example2.susipf(1e-10,1e10,"n",ftype="stair",mono="d",stairb="y",selec=-1)[0]
        [0.0003684254485114842, 0.0005632616075469215, 0.0009334464746366321]
        self.assertAlmostEqual(pfh,0.0005632616075469215)
        self.assertAlmostEqual(pfh_lower,0.0003684254485114842)
        self.assertAlmostEqual(pfh_upper,0.0009334464746366321)


if __name__=="__main__":
    unittest.main()



#to create new add at first with verified version:
#print([pfh_lower,pfh,pfh_upper])











