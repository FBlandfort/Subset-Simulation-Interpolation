


#assertion tests
import unittest

import numpy as np
import random
import math

from susi import props
from susi import sampling


#could add more testings of exception handling 


class main_test_attr(unittest.TestCase):

    #####################################################################
    ###### module props    ##############################################       
 
    def test_convert_normal(self):
        self.assertAlmostEqual(props.rv(rtype="n",args=props.attr(name="test",rtype="n",mx=1e-10,vx=1e10).get_convert().args).mean,1e-10) 
        self.assertAlmostEqual(props.rv(rtype="n",args=props.attr(name="test",rtype="n",mx=1e-10,vx=1e10).get_convert().args).sd,1e-10*1e10)
        self.assertAlmostEqual(props.rv(rtype="n",args=props.attr(name="test",rtype="n",mx=-5,vx=4.3).get_convert().args).mean,-5) 

    def test_convert_lognormal(self):
        self.assertAlmostEqual(props.rv(rtype="ln",args=props.attr(name="test",rtype="ln",mx=4.5,vx=1.2).get_convert().args).mean,4.5)
        self.assertAlmostEqual(props.rv(rtype="ln",args=props.attr(name="test",rtype="ln",mx=4.5,vx=1.2).get_convert().args).sd,4.5*1.2)
        self.assertAlmostEqual(props.rv(rtype="ln",args=props.attr(name="test",rtype="ln",mx=0.05,vx=0.1).get_convert().args).mean,0.05) 
        self.assertAlmostEqual(props.rv(rtype="ln",args=props.attr(name="test",rtype="ln",mx=0.05,vx=0.1).get_convert().args).sd,0.05*0.1)

    def test_convert_gumbel(self):
        self.assertAlmostEqual(props.rv(rtype="g",args=props.attr(name="test",rtype="g",mx=4.5,vx=1.2).get_convert().args).mean,4.5)
        self.assertAlmostEqual(props.rv(rtype="g",args=props.attr(name="test",rtype="g",mx=4.5,vx=1.2).get_convert().args).sd,4.5*1.2)
        self.assertAlmostEqual(props.rv(rtype="g",args=props.attr(name="test",rtype="g",mx=0.05,vx=0.1).get_convert().args).mean,0.05) 
        self.assertAlmostEqual(props.rv(rtype="g",args=props.attr(name="test",rtype="g",mx=0.05,vx=0.1).get_convert().args).sd,0.05*0.1)

    def test_convert_exponential(self):
        self.assertAlmostEqual(props.rv(rtype="e",args=props.attr(name="test",rtype="e",mx=4.5,vx=1.2).get_convert().args).mean,4.5)
        self.assertAlmostEqual(props.rv(rtype="e",args=props.attr(name="test",rtype="e",mx=4.5,vx=1.2).get_convert().args).sd,4.5*1.2)
        self.assertAlmostEqual(props.rv(rtype="e",args=props.attr(name="test",rtype="e",mx=0.05,vx=0.1).get_convert().args).mean,0.05) 
        self.assertAlmostEqual(props.rv(rtype="e",args=props.attr(name="test",rtype="e",mx=0.05,vx=0.1).get_convert().args).sd,0.05*0.1)

    def test_convert_uniform(self):
        self.assertAlmostEqual(props.rv(rtype="u",args=props.attr(name="test",rtype="u",mx=0,vx=5).get_convert().args).mean,2.5)
        self.assertAlmostEqual(props.rv(rtype="u",args=props.attr(name="test",rtype="u",mx=0,vx=5).get_convert().args).sd,math.sqrt((5**2)/12))
        self.assertAlmostEqual(props.rv(rtype="u",args=props.attr(name="test",rtype="u",mx=6,vx=4).get_convert().args).mean,8) 
        self.assertAlmostEqual(props.rv(rtype="u",args=props.attr(name="test",rtype="u",mx=6,vx=4).get_convert().args).sd,math.sqrt((4**2)/12))


    #####################################################################
    ###### module sampling ##############################################


    #######
    #sortbi

    def test_sortbi_p_given(self):
        #p=0.2
        [U_samples_ls_sorted, U_fail, U_no_fail, bi, p] = sampling.sortbi(U_samples_ls=np.column_stack(([random.sample(range(1,11),3) for i in range(10)],random.sample(range(1, 11), 10))), p=0.2,bi=10**6,bstar=0)
        self.assertEqual(p,0.2)
        self.assertEqual(bi,2)
        self.assertEqual(np.testing.assert_array_equal(U_fail[:,-1],np.array([1,2])),None)
        self.assertEqual(np.testing.assert_array_equal(U_no_fail[:,-1],np.arange(3,11)),None)
        self.assertEqual(np.testing.assert_array_equal(U_samples_ls_sorted[:,-1],np.arange(1,11)),None)

    def test_sortbi_p_given_bstar_reached(self):
        #bstar=6.5, note that case bstar equals a limitstate should be a null set (Probability=0)
        [U_samples_ls_sorted, U_fail, U_no_fail, bi, p] = sampling.sortbi(U_samples_ls=np.column_stack(([random.sample(range(1,11),3) for i in range(10)],random.sample(range(1, 11), 10))), p=0.2,bi=10**6,bstar=6.5)
        self.assertEqual(p,0.6)
        self.assertEqual(bi,6.5)
        self.assertEqual(np.testing.assert_array_equal(U_fail[:,-1],np.arange(1,7)),None)
        self.assertEqual(np.testing.assert_array_equal(U_no_fail[:,-1],np.arange(7,11)),None)
        self.assertEqual(np.testing.assert_array_equal(U_samples_ls_sorted[:,-1],np.arange(1,11)),None)

    def test_sortbi_p_not_given_bi_given(self):
        #bstar=6.5, note that case bstar equals a limitstate should be a null set (Probability=0)
        [U_samples_ls_sorted, U_fail, U_no_fail, bi, p] = sampling.sortbi(U_samples_ls=np.column_stack(([random.sample(range(1,11),3) for i in range(10)],random.sample(range(1, 11), 10))), p=0.0,bi=6.5,bstar=6.5)
        self.assertEqual(p,0.6)
        self.assertEqual(bi,6.5)
        self.assertEqual(np.testing.assert_array_equal(U_fail[:,-1],np.arange(1,7)),None)
        self.assertEqual(np.testing.assert_array_equal(U_no_fail[:,-1],np.arange(7,11)),None)
        self.assertEqual(np.testing.assert_array_equal(U_samples_ls_sorted[:,-1],np.arange(1,11)),None)
        [U_samples_ls_sorted, U_fail, U_no_fail, bi, p] = sampling.sortbi(U_samples_ls=np.column_stack(([random.sample(range(1,11),3) for i in range(10)],random.sample(range(1, 11), 10))), p=0.0,bi=4.5,bstar=5.5)

    def test_sortbi_bi_given_smaller_than_bstar(self):
        self.assertRaises(Warning)

    def test_higham(self):
        highmatr=props.nearPD(np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]]),nit=10)
        self.assertRaises(AssertionError,np.testing.assert_almost_equal(highmatr, np.array([ [ 1.        , -0.80842467  ,0.19157533 , 0.10677227]
                                        ,[-0.80842467 , 1.    ,     -0.65626745 , 0.19157533]
                                        ,[ 0.19157533 ,-0.65626745  ,1.     ,    -0.80842467]
                                        ,[ 0.10677227 , 0.19157533 ,-0.80842467 , 1.   ]])))







#assert_allclose

if __name__=="__main__":
    unittest.main()


#ipython --pylab qt5


#to execute the test
#python -m unittest susi.tests.deterministic_tests

#ipython --pylab qt5


#higham
#print nearPD(np.matrix([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]]),nit=10)''

'''
[ 1.         -0.80842467  0.19157533  0.10677227]
 [-0.80842467  1.         -0.65626745  0.19157533]
 [ 0.19157533 -0.65626745  1.         -0.80842467]
 [ 0.10677227  0.19157533 -0.80842467  1.        ]]
'''



















