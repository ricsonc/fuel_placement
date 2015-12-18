from fuel_placement import *

#helper functions:

def rand_perm_instance_gen((min_size, max_size), (min_reps, max_reps),
                           distances, rel_freqs = None):
    '''generates an instance of Fuel Placement Problem such that:
    number of tanks is between min_size and max_size
    a selected pattern is repeated between min_reps and max_reps times
    all distances are selected from the list distances
    with relative frequencies rel_freqs (a small integer list)
    '''
    size = randint(min_size, max_size)
    cycles = randint(min_reps, max_reps)
    period = size/cycles
    distances_p = []
    if not rel_freqs:
        rel_freqs = [1]*len(distances)
    for i, x in enumerate(distances):
        distances_p.extend([x]*rel_freqs[i])
    distance_order = []
    for i in xrange(size):
        distance_order.append(choice(distances_p))
    return perm_instance(distance_order*cycles)

def perm_instance(distances, name = '', starts = 0):
    '''returns an instance of Fuel Placement Problem
    where fuels are a permutation of the distances
    given a list of distances
    and an optional name'''
    fuels = distances[:]
    OPTsoln = Solution(distances[:],0)
    OPT = max(distances)
    return Fuel_Placement_Problem(distances, fuels, OPT, OPTsoln, name,
                                  starts = starts)
    
def find_bad_case(alg_name, rand_gen_params, name = '', check_fn_name = None,
                  verbose = 100):
    '''returns an instance of Fuel Placement Problem
    given rand_instance_gen parameters
    which will fail when checked with the function with name check_fn_name
    note: only generates and tests permutation cases
    prints number of cases tested if verbose'''
    i = 0
    while 1:
        i += 1
        if verbose and not i%verbose:
            print i,
        candidate = rand_perm_instance_gen(*rand_gen_params)
        soln_fn = getattr(candidate, alg_name)
        soln_a = (soln_fn() if not check_fn_name
                  else soln_fn(check_fn = getattr(candidate, check_fn_name)))
        if not soln_a.success:
            candidate.name = name if name else alg_name
            return candidate

def find_and_plot(algname, search_params, name = '', check_fn = ''):
    '''searches for a bad case using find_bad_case
    then plots the test case with soln_attempt_plot
    then saves the test case to disk'''
    test_case = find_bad_case(algname, search_params, name, check_fn)
    test_case.soln_attempt_plot(getattr(test_case,algname), scale = .5)
    test_case.save()

##tests:

#specific instances:

def test_max_min_feas():
    '''tests and plots the result for the max_min algorithm
    max_min does not find a feasible solution here'''
    test_case = perm_instance(([20]*2+[1]*4+[10]*6)*2, "max_min_bad")
    test_case.soln_attempt_plot(test_case.max_min, 12)
    return test_case

def test_max_min_gt_feas():
    '''tests and plots the results for the max_min_gt algorithm
    max_min_gt does not find a feasible solution here'''
    test_case = perm_instance(([20]*2+[1]*4+[10]*6)*3, "max_min_gt_bad")
    test_case.soln_attempt_plot(test_case.max_min_gt, 12)
    return test_case

def test_greedy_OPT():
    '''tests and plots the results for the greedy algorithm
    greedy does not find an optimal solution to this instance'''
    distances = [1,1,1,24,1,1,24]
    fuels = [4, 6, 8, 8, 8, 9, 10]
    OPT = 24
    OPTsoln = Solution([4, 6, 8, 9, 8, 8, 10],0)
    test_case = Fuel_Placement_Problem(fuels, distances, OPT, OPTsoln,
                                       'greedy_bad')
    test_case.soln_attempt_plot(test_case.greedy, **{'ratio':1})
    return test_case

def test_greedy_fixed_2OPT():
    '''tests and plots the results for the greedy algorithm
    greedy_fixed does not find an 2 optimal solution to this instance'''
    distances = [10]+[3]*13+[10]*3
    fuels = [10]*3+[3]*7+[4]*7
    OPT = 10
    OPTsoln = Solution([10]+[4]*7+[3]*7+[10]*2,0)
    test_case = Fuel_Placement_Problem(fuels, distances, OPT, OPTsoln,
                                       'greedy_fixed_bad')
    test_case.soln_attempt_plot(test_case.greedy_fixed)
    return test_case

def test_greedy_2OPT():
    '''tests and plots the results for the greedy algorithm
    greedy does not find an 2 optimal solution to this instance'''
    distances = ([2]*5+[5]*4)*5
    fuels = ([5]*3+[2]*3+[3]*3)*5
    OPT = 5
    OPTsoln = Solution(([3]*3+[2]*3+[5]*3)*5,0)
    test_case = Fuel_Placement_Problem(fuels, distances, OPT, OPTsoln,
                                       'greedy_bad', 9)
    test_case.soln_attempt_plot(test_case.greedy)
    return test_case

def test_minover_min():
    '''tests and plots the results for the minover_min algorithm'''
    distances = ([2]*5+[5]*4)*20
    fuels = ([5]*3+[2]*3+[3]*3)*20
    OPT = 5
    OPTsoln = Solution(([3]*3+[2]*3+[5]*3)*20,0)
    test_case = Fuel_Placement_Problem(fuels, distances, OPT, OPTsoln,
                                       'mom_bad', 9)
    test_case.soln_attempt_plot(test_case.minover_min)
    return test_case

def test_min_next():
    '''tests and plots the results for the min_next algorithm'''
    distances = ([2]*5+[5]*4)*20
    fuels = ([5]*3+[2]*3+[3]*3)*20
    OPT = 5
    OPTsoln = Solution(([3]*3+[2]*3+[5]*3)*20,0)
    test_case = Fuel_Placement_Problem(fuels, distances, OPT, OPTsoln,
                                       'min_next_bad', 9)
    test_case.soln_attempt_plot(test_case.min_next)
    return test_case

def test_greedy_AOPT(do_plot = False):
    '''tests the results for the greedy algorithm,
    attempts to cause an arbitrarily bad approximation
    greedy ends up succeeding on this case
    '''
    r = 10
    e = 1
    k = 20
    F = 1001
    A = 1000
    distances = ([e]*(A+49)+[F]*2)*r
    fuels = ([e]*A+[e+k]*50+[F])*r
    OPT = F
    OPTsoln = Solution(([e+k]*50+[e]*A+[F])*r,0)
    test_case = Fuel_Placement_Problem(fuels, distances, OPT, OPTsoln,
                                       'greedy_aopt', A+51)
    print test_case.approx_ratio(test_case.greedy)
    if do_plot:
        test_case.soln_attempt_plot(test_case.greedy)
    return test_case

#random searches:

def search_max_min_UB():
    '''searches for and plots a case where
    max_min goes over 3 times optimal'''
    find_and_plot('max_min', ((5,50),(2,11),[1,20,10]), 'max_min_UB',
                  'check_soln_UB')

def search_minover_max_feas():
    '''searches for and plots a case where
    minover_max finds no solutions'''
    find_and_plot('minover_min', ((5,50),(2,11),[1,2,10]), 'minover_min')
    
def search_greedy_2OPT():
    '''searches for and plots a case where
    greedy goes over 2 time optimal or fails'''
    find_and_plot('greedy', ((5,50),(2,11),[1,20,10]), 'greedy_2OPT')

def search_greedy_OPT():
    '''searches for and plots a case where
    greedy gives a nonoptimal solution'''
    find_and_plot('greedy', ((5,50),(2,11),[1,20,10]), 'greedy_OPT')

def search_greedy_fixed_OPT():
    '''searches for and plots a case where greedy 
    gives a nonoptimal solution starting at p_0'''
    find_and_plot('greedy_fixed', ((1,100),(2,5),[1,20,10]), 'greedy_fixed_opt')
    
def main():
    #uncomment an example below
    #test_max_min_feas()
    #test_max_min_gt_feas()
    #test_greedy_OPT()
    #test_greedy_fixed_2OPT()
    #test_greedy_AOPT()
    #test_minover_min()
    #test_min_next()
    #searcho_max_min_UB()
    #search_minover_max_feas()
    #search_greedy_2OPT()
    #search_greedy_OPT()
    #search_greedy_fixed_OPT()
    pass
    
if __name__ == '__main__':
    main()

