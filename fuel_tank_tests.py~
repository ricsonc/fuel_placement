from fuel_tank import *

def rand_instance_gen((min_size, max_size), (min_reps, max_reps),
                           distances, rel_freqs = None):
    '''generates an instance of Fuel Tank Problem such that:
    number of tanks is between min_size and max_size
    a selected pattern is repeated between min_reps and max_reps times
    all distances are selected from the list distances
    with relative frequencies rel_freqs (a small integer list)
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
    return distance_order*cycles

def perm_instance(distances, name = ''):
    '''returns an instance of Fuel Tank Problem
    where fuels are a permutation of the distances
    given a list of distances
    and an optional name'''
    pass

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
        cand_inst = FT_instance(candidate[:], candidate, max(candidate),
                                Solution(candidate[:],0),
                                name if name else alg_name)
        soln_fn = getattr(cand_inst, alg_name)
        soln_a = (soln_fn() if not checkmode
                  else soln_fn(check_fn = getattr(cand_inst, checkmode)))
        if not soln_a.success:
            return cand_inst

def find_and_plot_case(alg_name, rand_gen_params,
                       name = '', check_fn_name = ''):
    '''finds a bad case with find_bad_case
    plots the case and saves it to file'''

    bad_case = get_bad_case_mm()
    bad_case.soln_attempt_plot(bad_case.max_min_soln, scale = 0.5)
    bad_case.save()

def test_max_min_fail():
    
    '''tests a specific counter-example to the max_min algorithm
    plots the results'''
    
    name = 'mm_bad_instance'
    epsilon = 1
    full = 20
    half = 10

    fulls = 2
    epsilons = 4
    halves = 6
    repetitions = 3
    
    distances = ([full]*fulls+[epsilon]*epsilons+[half]*halves)*repetitions
    fuels = distances[:]
    OPT = 20
    OPTsoln = Solution(distances[:], 0)

    bad_instance = FT_instance(fuels, distances, OPT, OPTsoln, name)
    bad_instance.soln_attempt_plot(bad_instance.max_min_soln, 12)

def test_minover_max_fail():

    '''tests a specific example to the minover_max algorithm and plots the results
    minover_max succeeds on this instance
    but max_min fails on this instance'''
    
    name = 'minover_max_instance'
    distances = ([10, 20, 20, 20, 10, 20, 20, 20, 20, 1,
                  20, 20, 20, 1, 1, 1, 10, 1, 20, 10]* 7)
    fuels = distances[:]
    OPT = 20
    OPTsoln = Solution(distances[:], 0)
    bad_instance = FT_instance(fuels, distances, OPT, OPTsoln, name)
    bad_instance.soln_attempt_plot(bad_instance.M3_soln, 20, scale = 0.5)
    bad_instance.soln_attempt_plot(bad_instance.max_min_soln, 20, scale = 0.5)

###################################
##remove the following eventually##
###################################

def bad_case_gen(mode = 1):
    if mode == 1:
        return rand_instance_gen((50, 500), (2, 7), [1, 20, 10])
    elif mode == 2:
        return rand_instance_gen((10,1000), (2, 10), range(20))
    elif mode == 3:
        return rand_instance_gen((10,1000), (2, 10), range(10,20))
    elif mode == 4:
        return rand_instance_gen((10,1000), (2, 10), [10,20])
    else:
        return rand_instance_gen((50, 3000), (2, 11), [1, 20, 10])

def find_bad_case_mm():
    #looks for a bad case for the max min algorithm
    return get_bad_case('max_min_soln', 'bad_case_mm_ub', m = 0,
                        checkmode = 'check_soln_UB')

def get_bad_case_greedy():
    #looks for a bad case for the greedy algorithm
    return get_bad_case('greedy_soln', 'bad_case_greedy', m = 1)

def get_bad_case_minover_max():
    #looks for bad case for the minover max algorithm
    return get_bad_case('M3_soln', 'bad_case_M3', m = 0)


def gen_bad_mm_ub_cases():
    bad_case = get_bad_case_mm()
    print bad_case.L
    print bad_case.distances    
    bad_case.soln_attempt_plot(bad_case.max_min_soln, scale = 0.5)
    bad_case.save()

def gen_bad_greedy_cases():
    bad_case = get_bad_case_greedy()
    print bad_case.L
    print bad_case.distances    
    bad_case.soln_attempt_plot(bad_case.greedy_soln, scale = 0.5)
    bad_case.save()

def gen_bad_M3_cases():
    bad_case = get_bad_case_M3()
    print bad_case.L
    print bad_case.distances    
    bad_case.soln_attempt_plot(bad_case.M3_soln, scale = 0.5)
    bad_case.save()

#test cases:

#greedy bad instance search
#minover max bad instance search
#max min bad instance over search
#greedy 1 OPT search

#max min bad instance * 
#max min bad gt instance *
