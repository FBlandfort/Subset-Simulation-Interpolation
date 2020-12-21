
################################################
                                             
# Subset-Simulation-Interpolation (SuSI)     
                                             
################################################

This package introduces the novel algorithm SuSI,
but does also allow for extensive examination of
properties of ordinary Subset Simulation (SuS)

### Beta Version:
This algorithm is in a state suitable for scientific use, but should not be considered for 
reliability estimation in reality by now. The main reason for this recommendation
is that there were not yet much efforts to make it easy applicable and handle all
relevant input errors in an appropriate way.
Instead, the implementation focuses on provision of many settings, some only relevant
for research. For practice, such settings should be removed to avoid confusion.



##################################################################
# SCOPE            			

The algorithms in this implementation is a sequential Monte Carlo algorithm 
for reliability evaluation of complex engineering structures
It aims at
- Scientific examination of ordinary SuS
- Introduction of a novel algorithm, SuSI


The development aims at scientific research. 
Thus there are many settings involved for scientific research related to specific
examinations. These are often not important to efficiently evaluate specific examples
and default values typically provide efficient calculation. 

However, this implementation aims to mainly support researches at testing many 
different settings to explore the properties of SuS
and SuSI to the fullest, deriving information about its general mechanics. 
This allows to isolate effects of specific parameter selections or to analyze
shortcomings or extend the algorithms in the future in a novel manner.
 



## Related Scientific Publications:
This project was created in proceeding with, and used for, the dissertation
'Subset Simulation and Interpolalation: Efficient Reliability Estimation under Model-Dynamics
for Complex Civil Engineering Structures' by Florian Blandfort

Note that this is the result of a collaboration with the department of Civil Engineering
and also relies on joint work, and previous contributions by others, as cited below.
More specifically, it extends or ties to the following papers:

### Subset Simulation, in general
- S.-K. Au and J. L. Beck. Estimation of small failure probabilities in high
dimensions by subset simulation. Probabilistic Engineering Mechanics, 16
(4):263–277, 2001b.
-S.-K. Au and Y. Wang. Engineering risk assessment with subset simulation.
John Wiley & Sons, 2014.
### Subset Simulation, in particular adaptive conditional sampling (/sampling/acs)
- I. Papaioannou, W. Betz, K. Zwirglmaier, and D. Straub. Mcmc algorithms
for subset simulation. Probabilistic Engineering Mechanics, 41:89–103, 2015.



### Parameter State Model like evaluation:
-F. Blandfort, C. Glock, J. Sass, S. Schwaar, and R. Sefrin. A parametric state
space model for time-dependent reliability. In D. Yurchenko and D. Proske,
editors, Proceedings of the 17th International Probabilistic Workshop (IPW
2019), pages 31–36, Edinburgh, UK, 2019b. Heriot Watt University.
- F. Blandfort, C. Glock, J. Sass, S. Schwaar, and R. Sefrin. Efficient and
comprehensive time-dependent reliability analysis of complex structures by
a parameter state model. Accepted for publication in:
Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering.
(to appear: 2021)

### Subset Simulation Interpolation:
- F. Blandfort, C. Glock, J. Sass, S. Schwaar, and R. Sefrin. Subset simulation
interpolation a new approach to compute effects of model-dynamics in
structural reliability. In E. Z. M. Beer, editor, Proceedings of the 29th Eu-
ropean safety and reliability conference (ESREL 2019), pages 1978–1986,
Hannover, Germany, 2019a. ESRA.






##################################################################
# USAGE           

### Set Up

- Clone the repository
- Make sure that Python is installed (tested with Python 3.7.1)
- Install the requirements: `pip install -r requirements.txt`
- Goto folder of repository (Subset-Simulation-Interpolation, same folder as README.md is located)
and start python in this folder (elsewise, you can also change the working directory accordingly)
- import susi
- Now you can either proceed with 1., 2. or 3. (see below)

The corresponding methodology is described in the following

### Methodology

- First, a strurel object is created, representing the structure to be analyzed
with its stochastic properties and corresponding limit state equation as well
as the correlations of the variables
Say, we call the strurel object strurel1


- Second, one can compute results by either SuS or SuSI, based on the given strurel object
In both cases, we call the function get_result: strurel1.get_result(choose method="sus" or "susi"
	,"choose parameters for calculation").
If a result is calculated by SuS, the failure probability is given by 
strurel1.result.pfi[-1] (the last calculated failure probability)

- Third (SuSI), if we compute the result by SuSI, then we need post processing to receive the
failure probability. These computations are cheap now, no need for additional evaluations of
the limit state function.
If a result is calculated by SuSI, then we have to interpolate afterwards to receive a result
strurel1.susipf("choose properties of dynamic variable xk", "choose interpolation method")



#### 1. Testing (/tests)
As the provided algorithms relies on stochastic methods, 
we also partially use stochastic tests and verify 
further developments by some deterministic tests by seed setting.


python -m unittest susi.tests.tests_basic
python -m unittest susi.tests.tests_stochastic_sus
python -m unittest susi.tests.tests_stochastic_susi
python -m unittest susi.tests.tests_seed_sus
python -m unittest susi.tests.tests_seed_susi


### 2. Examples (/examples)

- coll1: sum of Gaussians, fixed version
- coll2: Linear, convex and concave limit states with arbitrary dimension and failure probability (according to: I. Papaioannou, W. Betz, K. Zwirglmaier, and D. Straub. Mcmc algorithms for subset simulation. Probabilistic Engineering Mechanics, 41:89–103, 2015.)





##################################################################
# IMPLEMENTATION          


- /main contains the main object which controls all functionalities

- /props contains stochastic properties of the strurel object,
	attributes, limit state function and random variables

- /sampling contains all sampling related functions such as adaptive
	conditional sampling, Subset Simulation, crude Monte Carlo

- /examples contains examplary objects for reliability evaluation

- /tests contains unittests for validation of implementation







