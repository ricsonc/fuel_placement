from itertools import chain
from collections import Counter, namedtuple
from itertools import permutations, combinations
from copy import copy
from random import choice, randint
from os import path, makedirs
from cPickle import load, dump
from pkgutil import find_loader
from importlib import import_module
from math import log, exp

#test if matplotlib is available
if find_loader('matplotlib') is not None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams as rcp
    MPL = True
else:
    print "matplotlib not found"
    print "plotting will not be possible"
    MPL = False

#helper functions:
    
def rotate(lst, i):
    '''rotates a list forward by i elements'''
    return lst[i:] + lst[:i]

def midpoint(p1, p2):
    '''returns midpoint of two points -- used for plotting'''
    return [(p1[0]+p2[0])/2., (p1[1]+p2[1])/2.]

def preparedir(file):
    '''if directory which contains file does not exist, make it.'''
    directory = file.rpartition('/')[0]
    if directory and not path.exists(directory):
        makedirs(directory)

def softmax(X, base, OPT):
    return sum((base**(float(x)/OPT) for x in X))

#class definitions:

class Solution(namedtuple("Solution", ['tank_order', 'start'])):
    '''A solution to an instance of the problem consists of:
    1. The list of fuel tanks f_1, f_2, .... f_n placed at positions
    p_k, p_k+1, ... p_k-1, where p_k is the start position
    2. The starting point, which is some integer from 1 to n'''

    
class Soln_Attempt(namedtuple("Soln_Attempt", ['success', 'solns', 'fails'])):
    '''A solution attempt consists of
    1. a boolean which is true iff successful solutions found
    2. A list of solutions (feasible and stays under alpha times OPT)
    3. A list of solutions which fail the previous constraint'''

    
class Fuel_Placement_Problem:
    '''instance of the fuel placement problem'''
    
    def __init__(self, fuels, distances, OPT = 0, OPTsoln = None,
                 name = 'fuel_placement_problem', starts = 0):
        '''Initialize a new instance of the fuel placement problem
        fuels: a list of available fuel tanks
        distances: a list of distances between possible tank positions
        OPT: the minimum fuel tank size possible
        OPTsoln: an optimal Solution which is solves this problem
        name: specifies the location that plots will be stored
        starts: specifies the number of unique starting points
        '''
        assert sum(fuels) == sum(distances)
        self.fuels = Counter(fuels)
        self.distances = distances
        self.n = len(fuels)
        self.OPT = OPT
        self.L = sum(fuels)
        self.OPTsoln = OPTsoln
        self.name = name
        self.starts = starts
        
    #helper functions:
            
    def valid_soln(self, soln):
        '''checks that all fuel tanks are used exactly once'''
        assert Counter(soln.tank_order) == self.fuels

    def fuel_levels(self, soln):
        '''returns the amount of fuel in the tank at each p_i 
        before and after tank is filled
        soln: a Solution to this instance
        '''
        rot_distances = rotate(self.distances, soln.start)
        fuels = [0]
        current_fuel = 0
        for i, x in enumerate(soln.tank_order):
            current_fuel += x
            fuels.append(current_fuel)
            current_fuel -= rot_distances[i]
            fuels.append(current_fuel)
        return fuels

    def check_soln_general(self, soln, ratio, fuel_fn):
        self.valid_soln(soln)
        fuels = self.fuel_levels(soln)
        return not(fuel_fn(fuels))
    
    def check_soln(self, soln, ratio):
        '''checks that at all times, the fuel in the tank is:
        1. above 0
        2. below ratio times opt'''
        fuel_fn = lambda fuels: min(fuels) < 0 or max(fuels) > ratio*self.OPT
        return self.check_soln_general(soln, ratio, fuel_fn)

    def check_soln_UB(self, soln, ratio):
        '''checks that at all times, the fuel in tank is:
        1. below ratio times opt
        does not check for above 0'''
        fuel_fn = lambda fuels: max(fuels) > ratio*self.OPT
        return self.check_soln_general(soln, ratio, fuel_fn)

    def check_soln_UBLB(self, soln, ratio):
        '''check that the difference between the upper and lower bound is
        less thn ratio times opt'''
        fuel_fn = lambda fuels: (max(fuels)-min(fuels)) > ratio*self.OPT
        return self.check_soln_general(soln, ratio, fuel_fn)
    
    def general_soln(self, ratio, soln_p, check_fn = None, lazy = True):
        '''
        returns solutions to the instance
        soln_p is a function that returns a solution starting at point p
        check_fn is the function used to check the validity of a solution
        this function returns a Soln_Attempt which stores:
        1. whether the algorithm succeeded at any starting point
        2. cases on which the algorithm succeeded
        3. cases on which the algorithm failed
        '''
        check_fn = check_fn if check_fn else self.check_soln
        good_solns = []
        bad_solns = []
        for x in xrange(self.starts if self.starts else self.n):
            print x
            soln = soln_p(x, ratio)
            if check_fn(soln, ratio):
                good_solns.append(soln)
                if lazy:
                    break
            else:
                bad_solns.append(soln)
        return Soln_Attempt(bool(good_solns), good_solns, bad_solns)

    def general_soln_p(self, start, selection_fn):
        '''returns a solution to the instance, not necessarily valid
        start: the starting point of the solution
        selection_fn: takes the current fuel in tank, and available tanks left
                      returns the tank which is selected next'''
        rot_distances = rotate(self.distances, start)
        current_fuel = 0
        tank_order = []
        t_fuels = copy(self.fuels)
        for distance in rot_distances:
            pick_tank = selection_fn(current_fuel, list(t_fuels.elements()),
                                     distance)
            t_fuels[pick_tank] -= 1
            tank_order.append(pick_tank)
            current_fuel += pick_tank
            current_fuel -= distance
        return Solution(tank_order, start)

    def general_local_search(self, cost_fn, neighbor_fn,
                             ratio, solution, verbose = True, debug = True):
        '''runs local search given a potential_fn, a neighbor_fn, 
        a ratio at which to stop, a starting solution'''
        potential_fn = lambda solution: cost_fn(self.fuel_levels(solution))
        if solution is None:
            solution = Solution(list(self.fuels.elements()),0)
        current_cost = potential_fn(solution)
        if current_cost < self.OPT*ratio:
            return Soln_Attempt(True, [solution], [])
        i = 0
        while True:
            if debug:
                self.plot_soln(solution, 'debug/'+str(i)) ###
            i+=1
            if verbose:
                print 'iteration', current_cost, current_cost/self.OPT
            good_neighbors = [neighbor for neighbor in neighbor_fn(solution)
                              if potential_fn(neighbor) < current_cost]
            if not good_neighbors:
                return Soln_Attempt(False, [], [solution])
            solution = min(good_neighbors, key = lambda x: potential_fn(x))
            current_cost = potential_fn(solution)
            if current_cost < self.OPT*ratio:
                if debug:
                    self.plot_soln(solution, 'debug/'+str(i)) ###
                return Soln_Attempt(True, [solution], [])

    def swap_2_neighbors(self, solution):
        '''returns all neighbors from a solution from swapping two fuels'''
        neighbors = []
        torder = solution.tank_order
        for i, x in enumerate(torder):
            for j, y in enumerate(torder):
                if i < j:
                    norder = torder[:]
                    norder[i], norder[j] = norder[j], norder[i]
                    neighbors.append(Solution(norder, solution.start))
        return neighbors

    def move_start_neighbors(self, solution):
        '''returns all neighbors of a solution from moving the start point'''
        neighbors = []
        for i in xrange(self.n):
            if i != solution.start:
                neighbors.append(Solution(solution.tank_order[:], i))
        return neighbors

    def all_neighbors(self, solution):
        '''returns all neighbors of a solution from
        swapping 2 fuels or the start point'''
        return (self.swap_2_neighbors(solution)+
                self.move_start_neighbors(solution))

    def positive_neighbors(self, solution):
        '''returns the subset of all_neighbors(solution) which do not 
        go below 0 fuel'''
        return [soln for soln in self.all_neighbors(solution) 
                if self.fuel_levels(soln) >= 0]

    def double_swap_neighbors(self, solution):
        neighbors = []
        for (a1,a2,b2,b1) in combinations(range(self.n),4):
            norder = solution.tank_order[:]
            #if norder[a1] != norder[b1] and norder[a2] != norder[b2]:
            if norder[a1] != norder[b1] or norder[a2] != norder[b2]:            
                norder[a1],norder[b1] = norder[b1],norder[a1]
                norder[a2],norder[b2] = norder[b2],norder[a2]
                neighbors.append(Solution(norder, solution.start))
        return neighbors

    def double_swap_positive_neighbors(self, solution):
        return [soln for soln in self.double_swap_neighbors(solution)
                if all(f >= 0 for f in self.fuel_levels(soln))]
    
    #implementation of algorithms:
    
    def greedy_p(self, start, ratio):
        '''the greedy algorithm applied to this instance at some start'''
        def greedy_selection(current_fuel, tfuels, _):
            '''select the largest which does not exceed the bound'''
            smallest_tank = min(tfuels)
            tank_size = self.OPT*ratio
            if current_fuel + smallest_tank > tank_size:
                return smallest_tank
            else:
                return max((fuel for fuel in tfuels
                            if current_fuel+fuel <= tank_size))
        return self.general_soln_p(start, greedy_selection)
        
    def greedy(self, ratio = 2, check_fn = None):
        '''greedy algorithm applied to this instance
        assumes that our guess of the min tank size, W = OPT*ratio'''
        return self.general_soln(ratio, self.greedy_p, check_fn)
    
    def greedy_fixed(self, ratio = 2, check_fn = None):
        '''greedy algorithm only at starting point'''
        start_soln = self.greedy_p(0, ratio)
        check_fn = check_fn if check_fn else self.check_soln
        if check_fn(start_soln, ratio):
            return Soln_Attempt(True, [start_soln], [])
        return Soln_Attempt(False, [], [start_soln])

    def max_min_p(self, start, ratio = 1):
        '''the max min algorithm applied to this instance at some start'''
        def max_min_selection(current_fuel, tfuels, _):
            '''select the smallest tank if under or at OPT
            else select largest'''
            return min(tfuels) if current_fuel >= self.OPT else max(tfuels)
        return self.general_soln_p(start, max_min_selection)
    
    def max_min(self, ratio = 3, check_fn = None):
        '''max min algorithm applied to this instance
        assumes that our guess for the min tank size is correct'''
        return self.general_soln(ratio, self.max_min_p, check_fn)

    def max_min_p_gt(self, start, ratio = 1):
        '''max min gt algorithm applied to this instance at some start'''
        def max_min_selection(current_fuel, tfuels, _):
            '''same as max_min_p, but inequality is exclusive'''
            return min(tfuels) if current_fuel > self.OPT else max(tfuels)
        return self.general_soln_p(start, max_min_selection)
    
    def max_min_gt(self, ratio = 3, check_fn = None):
        '''max min gt algorithm:
        same as max_min, but inequality in selection_fn is exclusive'''
        return self.general_soln(ratio, self.max_min_p_gt, check_fn)

    def minover_min_p(self, start, ratio = 3):
        '''minover min algorithm applied to this instance at some start'''
        def minover_min_selection(current_fuel, tfuels, _):
            '''if the largest tank does not get us over W, pick it
            else, pick the smallest tank which gets us over W'''
            return (min(tfuels) if current_fuel >= self.OPT
                    else (max(tfuels) if max(tfuels)+current_fuel < self.OPT
                          else min([fuel for fuel in tfuels
                                    if fuel+current_fuel >= self.OPT])))
        return self.general_soln_p(start, minover_min_selection)

    def minover_min(self, ratio = 3, check_fn = None):
        '''minover min algorithm
        assumes the guess for minimum tank size is OPT
        variant of max_min algorithm'''
        return self.general_soln(ratio, self.minover_min_p, check_fn)

    def min_next_p(self, start, ratio = 3):
        '''min next algorithm'''
        def min_next_selection(current_fuel, tfuels, dist):
            '''min next algorithm:
            pick the smallest fuel tank which gets to the next stop, 
            or the largest tank if none will'''
            return (max(tfuels) if max(tfuels)+current_fuel < dist
                    else (min([fuel for fuel in tfuels
                               if fuel+current_fuel >= dist])))
        return self.general_soln_p(start, min_next_selection)

    def max_local_search(self, ratio = 1, solution = None):
        '''runs local search with standard cost function
        and swap_2_neighbors as the neighbor function'''
        def cost(fuel_levels):
            return max(fuel_levels)-min(fuel_levels)
        return self.general_local_search(cost, self.swap_2_neighbors, ratio, 
                                         solution)

    def max2_local_search(self, ratio = 1.0001, solution = None):
        '''runs local search with max 2 cost function
        and swap_2_neighbors as the neighbor function'''        
        def cost(fuel_levels):
            min_level = min(fuel_levels)
            max_level = max(fuel_levels)
            return (max_level-min_level+
                    (fuel_levels.count(min_level)+
                     fuel_levels.count(max_level))*1E-6)
        return self.general_local_search(cost, self.swap_2_neighbors, ratio, 
                                         solution)
    
    def softmax_center_local_search(self, ratio = 1, solution = None):
        '''runs local search with the softmax cost function 
        with all values mapped to abs(val-center)
        and the swap_2_neighbors as the neighbor function'''
        def cost(fuel_levels):
            center = (max(fuel_levels)+min(fuel_levels))/2
            return log(sum((exp(abs(val-center)) for val in fuel_levels)))
        return self.general_local_search(cost, self.swap_2_neighbors, ratio, 
                                         solution)
    
    def softmax_abs_local_search(self, ratio = 1, solution = None):
        '''runs local search with the softmax cost function 
        with all values mapped to abs(val)
        and the all_neighbors as the neighbor function'''
        def cost(fuel_levels):
            return log(sum((exp(abs(val)) for val in fuel_levels)))
        return self.general_local_search(cost, self.all_neighbors, ratio, 
                                         solution)

    def softmax_positive_local_search(self, ratio = 1):
        '''runs local search with the softmax cost function 
        and the positive_neighbors as the neighbor function'''
        #construct positive solution
        solution = Solution() 
        def cost(fuel_levels):
            center = (max(fuel_levels)+min(fuel_levels))/2
            return log(sum((exp(val) for val in fuel_levels)))
        return self.general_local_search(cost, self.positive_neighbors, ratio, 
                                         solution)

    def softmax_rotate_local_search(self, base = 2., ratio = 1, solution = None):
        '''runs local search with softmax cost function
        with values mapped to val-min(vals)
        and swap_2_neighbors as the neighbor function'''
        #UNTESTED!
        def cost(fuel_levels):
            return softmax([val-min(fuel_levels) for val in fuel_levels],
                           base, self.OPT)
        return self.general_local_search(cost, self.swap_2_neighbors, ratio,
                                         solution)
    
    def doubleswap_max2_LS(self, ratio = 1.0001, solution = None):
        '''runs local search with max 2 cost function
        and double_swap_neighbors as the neighbor function'''        
        def cost(fuel_levels):
            min_level = min(fuel_levels)
            max_level = max(fuel_levels)
            return (max_level-min_level+
                    (fuel_levels.count(min_level)+
                     fuel_levels.count(max_level))*1E-10)
        return self.general_local_search(cost, self.double_swap_neighbors, ratio, 
                                         solution)

    def doubleswap_softmax_rotate_LS(self, base = 2, ratio = 1.0001, solution = None):
        def cost(fuel_levels):
            return softmax([val-min(fuel_levels) for val in fuel_levels],
                           base, self.OPT)
        return self.general_local_search(cost, self.double_swap_neighbors, ratio,
                                         solution)

    def doubleswap_softmax_center_LS(self, base = 2, ratio = 1.0001, solution = None):
        def cost(fuel_levels):
            center = (min(fuel_levels)+max(fuel_levels))/2.
            return softmax([abs(val-center) for val in fuel_levels],
                           base, self.OPT)
        return self.general_local_search(cost, self.double_swap_neighbors, ratio,
                                         solution)

    def doubleswap_softmax_abs_LS(self, base = 2, ratio = 1.0001, solution = None):
        def cost(fuel_levels):
            return softmax([abs(val) for val in fuel_levels], base, self.OPT)
        return self.general_local_search(cost, self.double_swap_neighbors, ratio,
                                         solution)

    def doubleswap_softmax_positive_LS(self, base = 2, ratio = 1.0001, solution = None):
        def cost(fuel_levels):
            return softmax(fuel_levels, base, self.OPT)
        return self.general_local_search(cost, self.double_swap_positive_neighbors,
                                         ratio, solution)
    
    def min_next(self, ratio = 3, check_fn = None):
        #check both upper and lower
        '''min next algorithm'''
        check_fn = check_fn if check_fn else self.check_soln_UBLB
        return self.general_soln(ratio, self.min_next_p, check_fn)
    
    def general_perm_follow(self, perm_soln):
        '''returns a solution attempt which follows the permutation solution'''
        #sort the tanks
        #pair the tanks
        #return constructed instance
        pass
    
    def perm_follow_aselection(self):
        '''follows the after-selected permutation solution of this instance'''
        #make after-selected permutation solution
        #return call general_perm_follow
        pass

    def perm_follow_bselection(self):
        '''follows the before-selected permutation solution of this instance'''
        #make before-selected permutation solution
        #return call general_perm_follow
        pass
    
    def min_level(self, tanks):
        '''return point at which fuel in tank is least'''
        fuels = self.fuel_levels(Solution(tanks, 0))
        return fuels.index(min(fuels))

    def brute_force(self):
        '''brute forces an optimal strategy in n! time
        does not return all failed attempts
        this function is untested!'''
        for perm in permutations(self.fuels.elements()):
            min_fuel_p = self.min_level(perm)
            soln = Solution(permp, min_fuel_p)
            if(check_soln(soln, 1)):
                return Soln_Attempt(true, [soln], [])
        return Soln_Attempt(false, [], [soln])

    #I/O functions and analysis:

    def approx_ratio(self, alg, **kwargs):
        '''computes the approximation ratio given an algorithm'''
        ratio = float('inf')
        attempt = alg(**kwargs)
        for soln in attempt.solns+attempt.fails:
            ratio = min(ratio, ((lambda x: max(x)-min(x))
                                (self.fuel_levels(soln)))/float(self.OPT))
        return ratio
        
    def plot_soln(self, soln, name = '', hbars = [-1,0,1,2,3], aspectr = 1,
                  scale = 1, verbose = True, annotations = False):
        '''plots a Solution:
        name is optional
        hbars are horizontal bars plotted at certain multiples of OPT
        scale determines thickness of lines and size of font
        verbose prints filename once it has been saved to disk
        if python crashes in this function, try reducing scale
        '''
        #check if matplotlib available
        global MPL
        if not MPL:
            print "unable to plot -- matplotlib not installed"
            return
        
        #compute font size and line thickness etc
        fontsize = scale*1000./self.L
        width = scale*100./self.L
        margin = scale*1.*self.OPT
        dpi = scale*2*self.L
        
        rcp['font.size'] = fontsize
        rcp['axes.linewidth'] = width
        rcp['xtick.major.width'] = width
        rcp['ytick.major.width'] = width
        rcp['xtick.major.size'] = width*10
        rcp['ytick.major.size'] = width*10

        #plots the fuel graph and horizontal bars
        levels = self.fuel_levels(soln)
        rot_dists = rotate(self.distances, soln.start)
        cum_dist_npf = reduce(lambda c, x: c + [c[-1] + x], rot_dists, [0])
        cum_dist = list(chain(*zip(*([cum_dist_npf]*2))))
        cum_dist.pop()

        for bar in hbars:
            plt.axhline(bar*self.OPT, 0, self.L, linewidth = width,
                        c = ('r' if bar else 'k'))
            if self.OPTsoln:
                olevels = self.fuel_levels(self.OPTsoln)[:-1]
                opt_fuels = rotate(olevels, 2*soln.start)
                optline, = plt.plot(cum_dist, opt_fuels+[opt_fuels[0]],
                                    linewidth = width/3, linestyle = ':', c='b')
                optline.set_dashes([.5,.5])
        plt.plot(cum_dist, levels, linewidth = width, c = 'b')

        #labels points and fuel tanks
        points = zip(cum_dist, levels)
        ax = plt.axes()
        if annotations:
            for i, p in enumerate(points):
                if i%2:
                    ax.annotate(str(p[0])+','+str(p[1]), (p[0], p[1]+margin/4.),
                                fontsize = fontsize, color = 'g')
                if i and i%2 and i < len(points)-2:
                    lastpoint = points[i-1]
                    mp = midpoint(p, lastpoint)
                    ax.annotate(p[1]-lastpoint[1], (mp[0]+margin/10, mp[1]),
                                fontsize = fontsize, color = 'm')

        #adjust axes
        ax.set_xlabel('cumulative distance', fontsize = fontsize*2)
        ax.set_ylabel('fuel in tank', fontsize = fontsize*2)
        ax.set_aspect(aspectr)
        ax.set_xlim(0, self.L)
        ax.set_ylim(min(levels)-margin, max(levels)+margin)
        if annotations:
            plt.xticks(cum_dist_npf)
        else:
            plt.xticks([])
            
        #save figure
        writeto = self.name+name+'.png'
        preparedir(writeto)
        plt.savefig(writeto, dpi = dpi, bbox_inches = "tight")
        plt.close()

        if verbose:
            print "drawn " + writeto

    def soln_attempt_plot(self, alg, scale = 1, verbose = True,
                          annot = False, **kwargs):
        '''plots the result of an algorithm which returns a Soln_Attempt'''
        attempt = alg(**kwargs)
        if attempt.success:
            if verbose:
                print "success"
            self.plot_soln(attempt.solns[0], '/'+alg.__name__+'-success',
                           scale = scale, annotations = annot)
        else:
            if verbose:
                print "failed"
            for i, x in enumerate(attempt.fails):
                self.plot_soln(x, '/'+alg.__name__+'-fail-'+str(i),
                               scale = scale, annotations = annot)

    def save(self):
        '''save the instance to disk'''
        with open(self.name, 'r') as ouf:
            dump(ouf)
            
