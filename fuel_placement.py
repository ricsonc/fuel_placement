from itertools import chain
from collections import Counter, namedtuple
from itertools import permutations
from copy import copy
from random import choice, randint
from os import path, makedirs
from cPickle import load, dump
from pkgutil import find_loader
from importlib import import_module

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
                 name = 'fuel_placement_problem'):
        '''Initialize a new instance of the fuel placement problem
        fuels: a list of available fuel tanks
        distances: a list of distances between possible tank positions
        OPT: the minimum fuel tank size possible
        OPTsoln: an optimal Solution which is solves this problem
        name: specifies the location that plots will be stored
        '''
        assert sum(fuels) == sum(distances)
        self.fuels = Counter(fuels)
        self.distances = distances
        self.n = len(fuels)
        self.OPT = OPT
        self.L = sum(fuels)
        self.OPTsoln = OPTsoln
        self.name = name
        
    #helper functions:
            
    def valid_soln(self, soln):
        '''checks that all fuel tanks are used exactly once'''
        assert Counter(soln.tank_order) == self.fuels

    def fuel_levels(self, soln):
        '''returns the amount of fuel in the tank at each p_i
        soln: a Solution to this instance
        postfuel: a boolean value
        if postfuel is true, the list contains also the amount of fuel the tank 
        has after it is filled at each p_i but before it has moved to p_i+1
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
    
    def check_soln(self, soln, ratio):
        '''checks that at all times, the fuel in the tank is:
        1. above 0
        2. below ratio times opt'''
        self.valid_soln(soln)
        fuels = self.fuel_levels(soln)
        if min(fuels) < 0 or max(fuels) > ratio*self.OPT:
            return False
        return True

    def check_soln_UB(self, soln, ratio):
        '''checks that at all times, the fuel in tank is:
        1. below ratio times opt
        does not check for above 0'''
        self.valid_soln(soln)
        fuels = self.fuel_levels(soln)
        if max(fuels) > ratio*self.OPT:
            return False
        return True
    
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
        for x in xrange(self.n):
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
            pick_tank = selection_fn(current_fuel, list(t_fuels.elements()))
            t_fuels[pick_tank] -= 1
            tank_order.append(pick_tank)
            current_fuel += pick_tank
            current_fuel -= distance
        return Solution(tank_order, start)

    #implementation of algorithms:
    
    def greedy_p(self, start, ratio):
        '''the greedy algorithm applied to this instance at some start'''
        def greedy_selection(current_fuel, tfuels):
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
        def max_min_selection(current_fuel, tfuels):
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
        def max_min_selection(current_fuel, tfuels):
            '''same as max_min_p, but inequality is exclusive'''
            return min(tfuels) if current_fuel > self.OPT else max(tfuels)
        return self.general_soln_p(start, max_min_selection)
    
    def max_min_gt(self, ratio = 3, check_fn = None):
        '''max min gt algorithm:
        same as max_min, but inequality in selection_fn is exclusive'''
        return self.general_soln(ratio, self.max_min_p_gt, check_fn)

    def minover_min_p(self, start, ratio = 3):
        '''minover min algorithm applied to this instance at some start'''
        def minover_min_selection(current_fuel, tfuels):
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

    def general_perm_follow(self):
        '''returns a solution attempt which follows the permutation solution'''
        pass
    
    def perm_follow_aselection(self):
        '''follows the after-selected permutation solution of this instance'''
        pass

    def perm_follow_bselection(self):
        '''follows the before-selected permutation solution of this instance'''
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

    #I/O functions:
    
    def plot_soln(self, soln, name = '', hbars = [-1,0,1,2,3], aspectr = 1,
                  scale = 1, verbose = True, annotations = True):
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
        margin = scale*100./self.OPT
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

    def soln_attempt_plot(self, alg, starts = 0, scale = 1, verbose = True,
                          annot = True, **kwargs):
        '''plots the result of an algorithm which returns a Soln_Attempt
        starts specifies the number of starting points which should be
        plotted, starting from the first'''
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
                if not starts or i < starts:
                    self.plot_soln(x, '/'+alg.__name__+'-fail-'+str(i),
                                   scale = scale, annotations = annot)

    def save(self):
        '''save the instance to disk'''
        with open(self.name, 'r') as ouf:
            dump(ouf)
            
