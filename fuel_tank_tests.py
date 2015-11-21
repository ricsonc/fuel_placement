from fuel_tank import *

#helper functions:

def rand_perm_instance_gen((min_size, max_size), (min_reps, max_reps),
                           distances, rel_freqs = None):
    '''generates an instance of Fuel Tank Problem such that:
    number of tanks is between min_size and max_size
    a selected pattern is repeated between min_reps and max_reps times
    all distances are selected from the list distances
    with relative frequencies rel_freqs (a small integer list)
    also returns the period of each cyle
    '''
    size = randint(min_size, max_size)
    cycles = randint(min_reps, max_reps)
    period = length/cycles
    distances_p = []
    if not rel_freqs:
        rel_freqs = [1]*len(distances)
    for i, x in enumerate(distances):
        distances_p.extend([x]*rel_freqs[i])
    distance_order = []
    for i in xrange(size):
        distance_order.append(choice(distances_p))
    return perm_instance(distance_order*cycles), size

def perm_instance(distances, name = ''):
    '''returns an instance of Fuel Tank Problem
    where fuels are a permutation of the distances
    given a list of distances
    and an optional name'''
    fuels = distances[:]
    OPTsoln = Solution(distances[:],0)
    OPT = max(distances)
    return Fuel_Tank_Problem(distances, fuels, OPT, OPTsoln, name)
    
def find_bad_case(alg_name, rand_gen_parameters, name = '', check_fn_name = '',
                  verbose = 100):
    '''returns an instance of Fuel Tank Problem
    given rand_instance_gen parameters
    which will fail when checked with the function with name check_fn_name
    note: only generates and tests permutation cases
    prints number of cases tested if verbose'''
    i = 0
    while 1:
        i += 1
        if verbose and not i%verbose:
            print i,
        candidate = bad_case_gen(mode = m if type(m) == type(1) else m())
        soln_fn = getattr(candidate, alg_name)
        soln_a = (soln_fn() if not checkmode
                  else soln_fn(check_fn = getattr(cand_inst, checkmode)))
        if not soln_a.success:
            cand_inst.name = name if name else alg_name
            return cand_inst

##actual tests:

#specific instances:

def test_max_min_feas():
    '''tests and plots the result for the max_min algorithm
    max_min does not find a feasible solution here'''
    test_case = perm_instance(([20]*2+[1]*4+[10]*6)*2, "max_min_bad_bad")
    test_case.soln_attempt_plot(test_case.max_min_soln, 12)

def test_max_min_gt_feas():
    '''tests and plots the results for the max_min_gt algorithm
    max_min_gt does not find a feasible solution here'''
    test_case = perm_instance(([20]*2+[1]*4+[10]*6)*3, "max_min_gt_bad_bad")
    test_case.soln_attempt_plot(test_case.max_min_soln, 12)

def test_minover_max():
    '''tests and plots the results for the minover_max algorithm
    minover_max succeeds on this instance but max_min fails'''
    test_case = perm_instance((([10]+[20]*3)*2+[20,1]+[20]*3+[1]*3+[10,1,20,1])*7,
                                 "minover_max_bad")
    test_case.soln_attempt_plot(test_case.M3_soln, 20, scale = 0.5)
    test_case.soln_attempt_plot(test_case.max_min_soln, 20, scale = 0.5)

def test_greedy_nonopt():
    '''tests and plots the results for the greedy algorithm
    greedy does not find an optimal solution to this instance'''
    #nonpermutation instance
    pass

#random searches:

def search_max_min_UB():
    '''searches for and plots a case where
    max_min goes over 3 times optimal'''
    pass #mode 0

def search_greedy_UB():
    '''searches for and plots a case where
    greedy goes over 2 time optimal or fails'''
    pass #mode 1

def search_greedy_nonopt():
    '''searches for and plots a case where
    greedy gives a nonoptimal solution'''
    pass #mode 1?

def search_minover_max_feas():
    '''searches for and plots a case where
    minover_max finds no solutions'''
    pass #mode 0

return rand_instance_gen((50, 500), (2, 7), [1, 20, 10])
return rand_instance_gen((50, 3000), (2, 11), [1, 20, 10])
