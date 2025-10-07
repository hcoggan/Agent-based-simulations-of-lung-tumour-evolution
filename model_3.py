# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:19:04 2024

@author: 44749
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:05:37 2024

@author: 44749
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:54 2024

@author: 44749
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:05:37 2024

@author: 44749
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:54 2024

@author: 44749
"""

import numpy as np
from matplotlib import pyplot as plt
import time as time
from scipy.spatial.distance import cdist
import argparse

from numba import jit, njit, prange

#scratch_limit = 2000

rng = np.random.default_rng()

coords = [-1, 0, 1]
neighbour_coords = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1,3)
neighbour_coords = np.delete(neighbour_coords, 13, axis=0)


#Given a matrix of distances and a set of N entries in that matrix, calculate the sum of the distances between all points
@njit()
def distance_between_all_points(dists, entries):
    total_distance = 0
    for e in range(len(entries)):
        for e_ in range(e):
            total_distance += dists[entries[e]][entries[e_]]
    return total_distance

#Given a set of P coordinates and a number N, choose the set of N points that are furthest apart.
def choose_furthest_points(coords, dists, no_samples, num_attempts=20000):
    #assume coords is of form (P, 3)
    #dists = cdist(coords, coords)
    #this gives a PxP matrix
    #500 choose 8, approximately our number of samples in the 5mn cell case, is of the order of 10^16 so we will not choose all samples
    #we will choose, let's say 10000 sets of samples
    min_dist, current_choice =  0, None
    for attempt in range(num_attempts):
        entries_to_test = np.random.choice(len(coords), no_samples, replace=False)
        distance = distance_between_all_points(dists, entries_to_test)
        if distance > min_dist:
            min_dist = distance
            current_choice = entries_to_test
    return current_choice


#to save the arrays when they get too large
def put_or_save(scratch, arr, minind, maxind, vals, arrlength, arrname, arrno, expno, scratch_limit): #we need simulation number, array number, array length, and which array this is
    #if arrname == "subclone_parentage":
        #print(vals)
        #assert(np.min(vals) > 0) #we're not trying to save any illegal types, right
    if arrno > scratch_limit:
        raise Exception("Too much saved into memory.")
    #bear in mind that minind and maxind are the 'hypothetical' index this would have if this were one long array
    act_min_ind, act_max_ind = minind-arrno*arrlength, maxind-arrno*arrlength #what are their indices in THIS array?
    overspill = act_max_ind - arrlength
    #if arrname == "subclone_parentage":
        #print("pre-check", arrno, np.min(arr[:act_min_ind]), arr[:act_min_ind]) #are there illegal types already in the dataset?
    #print("act_min_ind", act_min_ind, "act_max_ind", act_max_ind, "overspill", overspill, "len(vals)", len(vals))
    if overspill > 0: #too long!
        #then fill it up to the limit:
        arr[act_min_ind:] = vals[:len(vals)-overspill]
        #if arrname == "subclone_parentage":
            #print("saving check", arrno, np.min(arr)) #are we trying to save illegal datatypes?
        np.save(scratch+"/"+str(expno)+"_"+arrname+"_"+str(arrno)+".npy", arr) #save array
        newarr = np.zeros(arrlength, dtype=np.int64)
        newarr[:overspill] = vals[len(vals)-overspill:] #put in new values- note this will throw ERRORS if more than arrlength is added at once, which is good anyway
        print(arrname + " " +str(arrno) + " saved")
        return newarr, arrno+1
    else:
        arr[act_min_ind:act_max_ind] = vals
        return arr, arrno

@njit(parallel=True)
def update_random_numbers(ur_nos, ur_tracker, nr, thresh):
    if nr - ur_tracker < thresh:
        ur_nos, ur_tracker = np.random.rand(nr), 0
    return ur_nos, ur_tracker

@njit(parallel=True)
def update_random_muts(mut_rand_nos, mut_tracker, nr, thresh, exp_muts):
    if nr - mut_tracker < thresh:
        mut_rand_nos, mut_tracker = np.random.poisson(size=nr, lam=exp_muts), 0
    return mut_rand_nos, mut_tracker
    


#simplified version of the above, non-parallelised but using numpy
#force cells in a full deme to compete based on fitness
def cull_cells_in_one_deme(x, y, z, cell_types, cell_driv_muts, cell_epig_alts, positions, deme_pops, deme_size, s):
    pos, pop = positions[x][y][z], deme_pops[x][y][z]
    #find properties
    if pop > deme_size:
        types_here, driv_muts_here, epig_alts_here = cell_types[pos][:pop], cell_driv_muts[pos][:pop], cell_epig_alts[pos][:pop]
        survival_fitnesses = np.power(1+s, driv_muts_here+epig_alts_here)
        total_fit = np.sum(survival_fitnesses)
        survival_probs = deme_size*survival_fitnesses/total_fit
        #now decide which cells live
        surviving = np.where(np.random.rand(pop) <= survival_probs)[0] #indices of surviving cells
        types_here, driv_muts_here, epig_alts_here = types_here[surviving], driv_muts_here[surviving], epig_alts_here[surviving] #properties of surviving cells
        new_pop = len(surviving)
        #wipe properties of existing deme
        cell_types[pos], cell_driv_muts[pos], cell_epig_alts[pos] = 0, 0, 0
        cell_types[pos][:new_pop] = types_here
        cell_driv_muts[pos][:new_pop] = driv_muts_here
        cell_epig_alts[pos][:new_pop] = epig_alts_here
        #now update population
        deme_pops[x][y][z] = new_pop
    return cell_types, cell_driv_muts, cell_epig_alts, deme_pops


#binomial inheritance of nongenetic alterations
#takes a list of numbers of alterations (assume possessed by cells just dividing) and asks whether or not they will be inherited
@njit()
def binomial_inheritance(div_epig_alts, herit):
    for n, alt in enumerate(div_epig_alts):
        #if alt == 0, do nothing
        if alt == 1:
            div_epig_alts[n] = 1 if np.random.rand() <= herit else 0
        else:
            if alt > 1:
                div_epig_alts[n] = np.random.binomial(alt, herit)
    return div_epig_alts
        

#simplified version of the above, that copies one deme at a time
#I am no longer going to bother pre-generating random numbers, let's not overcomplicate things here
def single_deme_division_np(x, y, z, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, exp_muts, deme_size, s, herit, epig_rate, division_prob, driv_prob):
    pos, pop = positions[x][y][z], deme_pops[x][y][z]
    #find properties
    types_here, driv_muts_here, epig_alts_here = cell_types[pos][:pop], cell_driv_muts[pos][:pop], cell_epig_alts[pos][:pop]

    #okay so the error doesn't come in in this function, it's somewhere else- between division
    #assert(np.min(types_here) >= 1)
    #calculate division fitness
    division_probs_here = division_prob + (1-division_prob)*(1-np.power((1-s), driv_muts_here+epig_alts_here))
    dividing_decisions = np.random.rand(pop)
    dividing, not_dividing = np.where(dividing_decisions <= division_probs_here)[0], np.where(dividing_decisions > division_probs_here)[0]
    num_new_cells = 2*len(dividing) #by this we mean the number of cells which will be 'freshly divided'
    num_not_dividing = len(not_dividing)
    
    #make two copies of all dividing cells
    div_types, div_driv_muts, div_epig_alts = np.hstack((types_here[dividing], types_here[dividing])), np.hstack((driv_muts_here[dividing], driv_muts_here[dividing])), np.hstack((epig_alts_here[dividing], epig_alts_here[dividing]))
    #keep not-dividing cells
    non_div_types, non_div_driv_muts, non_div_epig_alts = types_here[not_dividing], driv_muts_here[not_dividing], epig_alts_here[not_dividing] 

    #print("div types", div_types)
    
    #if num_new_cells > 0:
        #print(len(div_types))
        #assert(np.min(div_types) >= 1)
        
    #if num_not_dividing > 0:
        #print(len(non_div_types))
        #assert(np.min(non_div_types) >= 1)
    
    #OK, so now these cells are dividing- decide which are mutating:
    muts_per_div = np.random.poisson(size=len(div_types), lam=exp_muts) #how many mutations do you expect?
    type_changing = np.where(muts_per_div > 0)[0] #we have at least one mutation
    num_new_types = len(type_changing) #we have this many new types

    #record the parentages of the new types
    new_subclone_parentages = div_types[type_changing]
    #then allocate these new types
    div_types[type_changing] = np.arange(num_types+1, num_types+num_new_types+1)

    #if num_new_cells > 0:
        #print(len(div_types))
        #assert(np.min(div_types) >= 1)
        
    #if num_not_dividing > 0:
        #print(len(non_div_types))
        #assert(np.min(non_div_types) >= 1)
    
    #how many of these are driver mutations? we assume we incur at most one new driver mut per division
    new_mut_nos = muts_per_div[type_changing] 
    new_driv_mut_here_probs = driv_prob*new_mut_nos #probability scales with number of mutations
    new_driv_nos = np.zeros_like(new_mut_nos) #zero by default
    actual_new_driver_mut = np.where(np.random.rand(num_new_types) <= new_driv_mut_here_probs)[0] #only of length num_new_types, so
    new_driv_nos[actual_new_driver_mut] = 1 #record one new driver in the relevant place
    cells_gaining_drivers = type_changing[actual_new_driver_mut] #find indices of cells gaining a driver mutation
    div_driv_muts[cells_gaining_drivers] += 1 #update the number of drivers in these cells
    
    #now deal with epigenetic alterations. firstly, decide which dividing cells are keeping their epigenetic alterations- each is kept with probability herit
    div_epig_alts = binomial_inheritance(div_epig_alts, herit)
    #now decide which are getting new alterations, at rate epig_rate
    new_epig_alts = np.where(np.random.rand(num_new_cells) <= epig_rate)[0]
    div_epig_alts[new_epig_alts] += 1 #update by one

    #now all dividing information has been updated, update the deme
    new_cell_pop = num_new_cells + len(not_dividing)
    cell_types[pos], cell_driv_muts[pos], cell_epig_alts[pos] = 0, 0, 0
    cell_types[pos][:new_cell_pop], cell_driv_muts[pos][:new_cell_pop], cell_epig_alts[pos][:new_cell_pop] = np.hstack((div_types, non_div_types)), np.hstack((div_driv_muts, non_div_driv_muts)), np.hstack((div_epig_alts, non_div_epig_alts))
    deme_pops[x][y][z] = new_cell_pop

    #assert(np.min(cell_types[pos][:new_cell_pop]) >= 1)

    #now return all of this
    return cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types+num_new_types, num_new_types, new_subclone_parentages, new_mut_nos, new_driv_nos
    

    
    
        
@njit(parallel=True)
def get_average_deme_fitnesses(fits, epig_alts, pop, s, division_prob):
    return np.average(np.power((1+s), fits[:pop]+epig_alts[:pop]))


#update this to cull cells in demes one by one
def all_deme_division(xs, ys, zs, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, ur_nos, ur_tracker, deme_size, n_demes, nr, exp_muts, s, herit, epig_rate, division_prob, driv_prob, spno, mno, dno, expno, scratch, arrlength, mut_rand_nos, mut_tracker, scratch_limit):
    deme_fits = np.zeros(n_demes)
    for deme in range(n_demes):
        x, y, z = xs[deme], ys[deme], zs[deme]
        pop, pos_here = deme_pops[x][y][z], positions[x][y][z]
        cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, num_new_types, new_subclone_parentages, new_mut_nos, new_driv_nos =  single_deme_division_np(x, y, z, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, exp_muts, deme_size, s, herit, epig_rate, division_prob, driv_prob)
        #assert(np.min(new_subclone_parentages) > 0)
        #save new stuff
        if num_new_types > 0:
            #assert(np.min(new_subclone_parentages) > 0)
            subclone_parentage, spno = put_or_save(scratch, subclone_parentage, num_types-num_new_types+1, num_types+1, new_subclone_parentages, arrlength, "subclone_parentage", spno, expno, scratch_limit)
            muts_per_clone, mno = put_or_save(scratch, muts_per_clone, num_types-num_new_types+1, num_types+1, new_mut_nos, arrlength, "muts_per_clone", mno, expno, scratch_limit)
            #drivs_per_clone, dno = put_or_save(scratch, drivs_per_clone, num_types-num_new_types+1, num_types+1, new_driv_nos, arrlength, "drivs_per_clone", dno, expno, scratch_limit)
        #now cull the cells in THIS deme, based on fitness-- the function finds population again within the step (necessarily, as it will have changed during division step) and position (unnecessarily)
        cell_types, cell_driv_muts, cell_epig_alts, deme_pops = cull_cells_in_one_deme(x, y, z, cell_types, cell_driv_muts, cell_epig_alts, positions, deme_pops, deme_size, s)
        #now update average fitness of this deme
        pop = deme_pops[x][y][z]
        deme_fits[pos_here] = get_average_deme_fitnesses(cell_driv_muts[pos_here], cell_epig_alts[pos_here], pop, s, division_prob)
    return cell_types, cell_driv_muts, cell_epig_alts, deme_pops, deme_fits, num_types, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, ur_nos, ur_tracker, mut_rand_nos, mut_tracker




#take a list of coordinates and decide which of them are neighbours
def create_neighbour_matrix(xs, ys, zs, num_demes):
    #a matrix where diff_xs[i][j] = abs(xs[i]-xs[j])
    diff_xs, diff_ys, diff_zs = np.absolute(xs[:, np.newaxis] - xs), np.absolute(ys[:, np.newaxis]-ys), np.absolute(zs[:, np.newaxis]-zs)
    max_diffs = np.maximum(np.maximum(diff_xs, diff_ys), diff_zs) #record the maximum difference
    #two cells are neighbours if and only if their maximum difference is one (if it's less it's the same point, if it's more they're further apart)
    is_neighbour = np.zeros((num_demes, num_demes), dtype=np.int8)
    neigh_1, neigh_2 = np.where(max_diffs==1)
    is_neighbour[neigh_1, neigh_2] = 1
    return is_neighbour

#move cells between demes during homeostatic phase
#DO NOT PARALLELISE, movement has to be done in order to avoid copying to the same deme
@njit()
def move_cells(cell_types, cell_driv_muts, cell_epig_alts, deme_pops, xs, ys, zs, positions, movement_matrix, n_demes, ur_nos, ur_tracker, order, nr):
    for deme in order: #go through in a random order
        x, y, z = xs[deme], ys[deme], zs[deme]
        pos = positions[x][y][z] #just to make sure, you know
        available = np.nonzero(movement_matrix[pos])[0] #does this deme have anywhere it can move to?
        num_available = len(available)
        if num_available > 0:
            pop = deme_pops[x][y][z]
            ur_nos, ur_tracker = update_random_numbers(ur_nos, ur_tracker, nr, pop+1)
            to_move_to = available[int(num_available*ur_nos[ur_tracker])] #this is the chosen POSITION
            ur_tracker += 1
            #choose one at random
            move_markers = ur_nos[ur_tracker:ur_tracker+pop]
            moving, staying = np.nonzero(move_markers < 0.5)[0], np.where(move_markers >= 0.5)[0]
            num_moving, num_staying = len(moving), len(staying)
            ur_tracker += pop
            cell_types[to_move_to][:num_moving], cell_driv_muts[to_move_to][:num_moving], cell_epig_alts[to_move_to][:num_moving] = cell_types[pos][moving], cell_driv_muts[pos][moving], cell_epig_alts[pos][moving]
            cell_types[to_move_to][num_moving:], cell_driv_muts[to_move_to][num_moving:], cell_epig_alts[to_move_to][num_moving:]= 0, 0, 0 #wipe existing cells
            types_staying, muts_staying, epig_staying =  cell_types[pos][staying], cell_driv_muts[pos][staying], cell_epig_alts[pos][staying]  #copy existing cells
            cell_types[pos], cell_driv_muts[pos], cell_epig_alts[pos] = 0,0,0
            cell_types[pos][:num_staying], cell_driv_muts[pos][:num_staying], cell_epig_alts[pos][:num_staying] = types_staying, muts_staying, epig_staying
            deme_pops[x][y][z], deme_pops[xs[to_move_to]][ys[to_move_to]][zs[to_move_to]] = num_staying, num_moving #reset populations
            movement_matrix[:, to_move_to] = 0 #make sure nothing else can expand into this
            movement_matrix[to_move_to, :] = 0 #make sure that a deme that has just been expanded into cannot then expand again-- these cells now stay where they are
    return cell_types, cell_driv_muts, cell_epig_alts, deme_pops, ur_nos, ur_tracker



#we consider only non-necrotic rim. two demes cannot appear in the same deme at once
def homeostat_growth(cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, xs, ys, zs, positions, n_demes, neighbour_matrix, order, arrlength, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, expno, s, herit, epig_rate, division_prob, driv_prob, deme_size, exp_muts, scratch, ur_nos, ur_tracker, mut_rand_nos, mut_tracker, scratch_limit, no_demes_total=10000, nr=1000000):
    #allow to divide and cull them
    cell_types, cell_driv_muts, cell_epig_alts, deme_pops, deme_fits, num_types, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, ur_nos, ur_tracker, mut_rand_nos, mut_tracker = all_deme_division(xs, ys, zs, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, ur_nos, ur_tracker, deme_size, n_demes, nr, exp_muts, s, herit, epig_rate, division_prob, driv_prob, spno, mno, dno, expno, scratch, arrlength, mut_rand_nos, mut_tracker, scratch_limit)
    #having allowed this, so that everything is on the same timestep, go through them again and cull/move them if there's a power differential
    movement_prob = (1-deme_fits/deme_fits[:, np.newaxis])*neighbour_matrix #prob is 0 if they're not neighbours
    movement_matrix = np.heaviside(movement_prob - rng.random((n_demes, n_demes)), 0) #1 if can move, 0 otherwise- assigned to recorded position, not in the order x, y, z are provided (though they SHOULD be the same thing)
    cell_types, cell_driv_muts, cell_epig_alts, deme_pops, ur_nos, ur_tracker = move_cells(cell_types, cell_driv_muts, cell_epig_alts, deme_pops, xs, ys, zs, positions, movement_matrix, n_demes, ur_nos, ur_tracker, order, nr)
    #the only order that matters is in the function above, and that is randomised within move_cells
    return cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, ur_nos, ur_tracker, mut_rand_nos, mut_tracker

#return a list of all neighbours and the locations of empty neighbours

def get_empty_neighbours(x, y, z, deme_pops, neighbour_coords):
    #check if it still has empty neighbour demes
    neighbours_here = np.add([x, y, z], neighbour_coords)
    xs_e, ys_e, zs_e = neighbours_here[:, 0], neighbours_here[:, 1], neighbours_here[:, 2]
    neighbour_pops = np.array([deme_pops[xs_e[r]][ys_e[r]][zs_e[r]] for r in range(len(neighbours_here))])
    empty_neighbours = np.where(neighbour_pops==0)[0]
    return xs_e, ys_e, zs_e, empty_neighbours


#modified and cheaper version of above- demes only expand if room
#keep an ongoing list of occupied positions, iterate over them in the order they were added
def initial_growth(cell_types, cell_driv_muts, cell_epig_alts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, necrotic_marker, ur_nos, ur_tracker, s, herit, epig_rate, division_prob, driv_prob, exp_muts, arrlength, spno, mno, dno, expno, deme_size, scratch, mut_rand_nos, mut_tracker, xs, ys, zs, scratch_limit, no_demes_total=10000, nr=1000000000):
    
    #find occupied demes
    #check populations here 
    no_occupied_demes = len(xs)
    
    for n in range(no_occupied_demes):
        x, y, z = xs[n], ys[n], zs[n]
        pos, pop = positions[x][y][z], deme_pops[x][y][z]
        #assert(np.min(cell_types[pos][:pop]) >= 1)
        #assert(len(np.where(cell_types[pos] != 0)[0]) == pop)
    #DO NOT FILTER OUT NECROTIC DEMES, it will mess with the number of occupied demes. just ignore them.
        
    order = np.arange(no_occupied_demes)
    #print("order before shuffling", order)
    np.random.shuffle(order) 
    #print("order after shuffling", order)
    #iterate over non-surrounded demes at start of simulation
    #allow cells to divide, with no culling
    for n in order:
        x, y, z = xs[n], ys[n], zs[n]
        pos = positions[x][y][z]
        no_cells_here = deme_pops[x][y][z]
        #print("pop check", cell_types[pos][:no_cells_here])
        #assert(np.min(cell_types[pos][:no_cells_here]) >= 1)

        #check if it still has empty neighbour demes
        xs_e, ys_e, zs_e, empty_neighbours = get_empty_neighbours(x, y, z, deme_pops, neighbour_coords)
        if len(empty_neighbours) == 0:
            necrotic_marker[x][y][z] = 1 #mark as necrotic, even if already necrotic
        else: #there are empty neighbours! so things can divide
            ur_nos, ur_tracker = update_random_numbers(ur_nos, ur_tracker, nr, 4*deme_size)
            #allow to divide on a per deme level
            cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, num_new_types, new_subclone_parentages, new_mut_nos, new_driv_nos =  single_deme_division_np(x, y, z, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, exp_muts, deme_size, s, herit, epig_rate, division_prob, driv_prob)
            #save everything that just came out of this
            if num_new_types > 0:
                subclone_parentage, spno = put_or_save(scratch, subclone_parentage, num_types-num_new_types+1, num_types+1, new_subclone_parentages, arrlength, "subclone_parentage", spno, expno, scratch_limit)
                muts_per_clone, mno = put_or_save(scratch, muts_per_clone, num_types-num_new_types+1, num_types+1, new_mut_nos, arrlength, "muts_per_clone", mno, expno, scratch_limit)
                drivs_per_clone, dno = put_or_save(scratch, drivs_per_clone, num_types-num_new_types+1, num_types+1, new_driv_nos, arrlength, "drivs_per_clone", dno, expno, scratch_limit)
            #now deal with overspill
            no_cells_here = deme_pops[x][y][z]
            #assert(np.min(cell_types[pos][:no_cells_here]) >= 1)
            #print("pre-movement population check", np.sum(deme_pops))
            if no_cells_here > deme_size: #overspill
                ur_nos, ur_tracker = update_random_numbers(ur_nos, ur_tracker, nr, 1)
                index_ov = empty_neighbours[int(len(empty_neighbours)*ur_nos[ur_tracker])] #choose an index from the list
                ur_tracker += 1
                x_ov, y_ov, z_ov = xs_e[index_ov], ys_e[index_ov], zs_e[index_ov]
                pos_ov = no_occupied_demes #assign it next position in list
                print("Last position", pos_ov, "cell_type length", len(cell_types))
                xs = np.append(xs, x_ov)
                ys = np.append(ys, y_ov)
                zs = np.append(zs, z_ov) # add position to list
                positions[x_ov][y_ov][z_ov] = pos_ov #record position
                #now move overspill at random
                ur_nos, ur_tracker = update_random_numbers(ur_nos, ur_tracker, nr, no_cells_here)
                mov_markers = ur_nos[ur_tracker:ur_tracker+no_cells_here] #split in half
                ur_tracker += no_cells_here
                moving, staying = np.where(mov_markers >= 0.5)[0], np.where(mov_markers<0.5)[0]
                no_moving = len(moving)
                no_staying = no_cells_here - len(moving)
                while no_moving > deme_size or no_staying > deme_size: #check we don't have overflow
                    ur_nos, ur_tracker = update_random_numbers(ur_nos, ur_tracker, nr, no_cells_here)
                    mov_markers = ur_nos[ur_tracker:ur_tracker+no_cells_here] #split in half
                    ur_tracker += no_cells_here
                    moving, staying = np.where(mov_markers >= 0.5)[0], np.where(mov_markers<0.5)[0]
                    no_moving = len(moving)
                    no_staying = no_cells_here - len(moving)
                #having decided which are moving, move them
                cell_types[pos_ov][:no_moving], cell_driv_muts[pos_ov][:no_moving], cell_epig_alts[pos_ov][:no_moving] = cell_types[pos][moving], cell_driv_muts[pos][moving], cell_epig_alts[pos][moving]
                deme_pops[x_ov][y_ov][z_ov] = no_moving
                cell_types[pos][:no_staying], cell_driv_muts[pos][:no_staying], cell_epig_alts[pos][:no_staying] = cell_types[pos][staying], cell_driv_muts[pos][staying], cell_epig_alts[pos][staying] 
                cell_types[pos][no_staying:], cell_driv_muts[pos][no_staying:], cell_epig_alts[pos][no_staying:] = 0, 0, 0 #wipe the rest of the existing ones
                deme_pops[x][y][z] = no_staying
                no_occupied_demes += 1 #it will be recalculated at the start of each step, this is to keep it up to date within it
                #assert(np.min(cell_types[pos][:no_staying]) >= 1)
                #assert(np.min(cell_types[pos_ov][:no_moving]) >= 1)
                #print("post-movement population check", np.sum(deme_pops))
                #run until the first timestep where this passes
    #one last sweep to check how many neighbours everything now has
    for n in range(len(xs)):
        x, y, z = xs[n], ys[n], zs[n]
        pos = positions[x][y][z]
        pop = deme_pops[x][y][z]
        #print("pop check after", cell_types[pos][:pop])
        #assert(np.min(cell_types[pos][:pop]) >= 1)
        if necrotic_marker[x][y][z]==0: #if it's not been marked as necrotic yet
            xs_e, ys_e, zs_e, empty_neighbours = get_empty_neighbours(x, y, z, deme_pops, neighbour_coords)
            if len(empty_neighbours) == 0:
                necrotic_marker[x][y][z] = 1 #mark as necrotic, discard
    return cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, necrotic_marker, ur_nos, ur_tracker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs
                
                
                
                
                
def run_to_full_size(expno, scratch, s, herit, epig_rate, division_prob, driv_prob, exp_muts, no_demes_total, no_cells_total, gridlength, deme_size, scratch_limit, nr=100000000, arrlength=1000000):
    num_types = 1
    c = int(gridlength/2)
    xs, ys, zs = np.array([c]), np.array([c]), np.array([c]) #initial list of positions
    subclone_parentage, muts_per_clone, drivs_per_clone = np.zeros(arrlength, dtype=np.int64), np.zeros(arrlength, dtype=np.int64), np.zeros(arrlength, dtype=np.int64)
    necrotic_marker = np.zeros((gridlength, gridlength, gridlength), dtype=np.int8)
    cell_types = np.zeros((int(2.5*no_demes_total), 2*deme_size), dtype=np.int64) #we may need more than 1000 demes
    cell_driv_muts = np.zeros((int(2.5*no_demes_total), 2*deme_size), dtype=np.int64)
    cell_epig_alts = np.zeros((int(2.5*no_demes_total), 2*deme_size), dtype=np.int64)
    print("Size in memory, roughly", 3*cell_types.nbytes)
    deme_pops = np.zeros((gridlength, gridlength, gridlength), dtype=np.int64)
    positions = np.zeros((gridlength, gridlength, gridlength), dtype=np.int64) #record position in list cell_types and cell_driv_muts
    deme_pops[c][c][c] = 1 #one cell in the middle, type 1, no driver mutations, that deme has position 0
    subclone_parentage[1] = 1 #type 1 is its own parent
    cell_types[0][0] = 1 #all starts at 1
    total_cells = 1
    t=0
    num_occupied_demes = 1
    spno, mno, dno = 0, 0, 0
    ur_nos, ur_tracker = rng.random(nr), 0
    mut_rand_nos, mut_tracker = rng.poisson(lam=exp_muts, size=nr), 0
    while total_cells < no_cells_total:
        cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, necrotic_marker, ur_nos, ur_tracker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs = initial_growth(cell_types, cell_driv_muts, cell_epig_alts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, necrotic_marker, ur_nos, ur_tracker, s, herit, epig_rate, division_prob, driv_prob, exp_muts, arrlength, spno, mno, dno, expno, deme_size, scratch, mut_rand_nos, mut_tracker, xs, ys, zs, scratch_limit, no_demes_total=no_demes_total, nr=nr)
    #find occupied deme
        for n in range(len(xs)):
            x, y, z = xs[n], ys[n], zs[n]
            pos, pop = positions[x][y][z], deme_pops[x][y][z]
            #assert(np.min(cell_types[pos][:pop]) >= 1)
        total_cells = np.sum(deme_pops)
        num_occupied_demes = len(np.where(deme_pops > 0)[0])
        print(total_cells, "cells", num_types, "types", num_occupied_demes, "occupied demes")
        t+= 1
    return t, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, necrotic_marker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs
    
 #now narrow this down to only those above necrotic core
#take existing necrosis markers and discard all those below surface
def keep_surface(cell_types, cell_driv_muts, cell_epig_alts, positions, deme_pops, necrotic_marker, gridlength, xs, ys, zs):
    surface = np.where(necrotic_marker[xs, ys, zs]==0)[0] #is on surface- we know this is accurate, we kept it
    surf_xs, surf_ys, surf_zs = xs[surface], ys[surface], zs[surface]
    kept_positions = positions[surf_xs, surf_ys, surf_zs]
    new_positions = np.zeros((gridlength, gridlength, gridlength), dtype=np.int64) #record positions of non-necrotic demes in new list
    new_positions[surf_xs, surf_ys, surf_zs] = np.arange(len(surface)) #once we condense list down, this is where the surface demes
    return cell_types[kept_positions], cell_driv_muts[kept_positions], cell_epig_alts[kept_positions], new_positions, surf_xs, surf_ys, surf_zs


#now run homeostatic growth
def run_surface_growth(t, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, necrotic_marker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, expno, nr, arrlength, deme_size, scratch, exp_muts, s, herit, epig_rate, division_prob, driv_prob, no_demes_total, total_time, gridlength, xs, ys, zs, scratch_limit):
    #identify surface- return locations and cell data in matched order
    cell_types, cell_driv_muts, cell_epig_alts, positions, surf_xs, surf_ys, surf_zs = keep_surface(cell_types, cell_driv_muts, cell_epig_alts, positions, deme_pops, necrotic_marker, gridlength, xs, ys, zs)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(surf_xs, surf_ys, surf_zs, c=deme_pops[surf_xs, surf_ys, surf_zs])
    plt.show()
    n_demes = len(cell_types) #how many demes?
    print("surface gathered", n_demes, "'active' demes")
    neighbour_matrix = create_neighbour_matrix(surf_xs, surf_ys, surf_zs, n_demes)
    ur_nos, ur_tracker = rng.random(nr), 0 #set up random numbers
    mut_rand_nos, mut_tracker = rng.poisson(lam=exp_muts, size=nr), 0
    print("neighbour matrix created")
    #decide orders in advance
    order_lists = rng.permuted(np.tile(np.arange(n_demes), total_time).reshape(total_time, n_demes), axis=1)
    while t < total_time:
        cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, ur_nos, ur_tracker, mut_rand_nos, mut_tracker = homeostat_growth(cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, surf_xs, surf_ys, surf_zs, positions, n_demes, neighbour_matrix, order_lists[t], arrlength, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, expno, s, herit, epig_rate, division_prob, driv_prob, deme_size, exp_muts, scratch, ur_nos, ur_tracker, mut_rand_nos, mut_tracker, scratch_limit, no_demes_total=no_demes_total, nr=nr)
        total_cells = np.sum(deme_pops)
        print(total_cells, "cells", num_types, "types", t+1, "days")
        t += 1
    return t, cell_types, deme_pops, num_types, positions, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, surf_xs, surf_ys, surf_zs

def grow_tumour(expno, scratch, division_prob, driv_prob, exp_muts, gridlength, deme_size, s, herit, epig_rate, no_demes_total, no_cells_total, arrlength, scratch_limit, nr=10000000, total_time=365):
    t, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, necrotic_marker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs  = run_to_full_size(expno, scratch, s, herit, epig_rate, division_prob, driv_prob, exp_muts, no_demes_total, no_cells_total, gridlength, deme_size, scratch_limit, nr=nr, arrlength=arrlength)
    t, cell_types, deme_pops, num_types, positions, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, surf_xs, surf_ys, surf_zs = run_surface_growth(t, cell_types, cell_driv_muts, cell_epig_alts, deme_pops, num_types, positions, necrotic_marker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, expno, nr, arrlength, deme_size, scratch, exp_muts, s, herit, epig_rate, division_prob, driv_prob, no_demes_total, total_time, gridlength, xs, ys, zs, scratch_limit)
    return t, cell_types, deme_pops, num_types, positions, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, surf_xs, surf_ys, surf_zs
 
#return THE INDICES OF 8 non-overlapping samples, each comprising 1% of active demes
def divide_samples_into_collections(surf_xs, surf_ys, surf_zs, deme_pops, no_samples):
    active_deme_locs = np.vstack((surf_xs, surf_ys, surf_zs)).T
    dists_between_demes = cdist(active_deme_locs, active_deme_locs)
    no_active_demes = len(active_deme_locs)
    no_demes_in_sample = int(0.01*no_active_demes) #about 1% of active demes and thus (we hope) of active cells
    samples_too_close = True
    while samples_too_close:
        test_sample_collections = np.zeros((no_samples, no_demes_in_sample), dtype=np.int64) #get a matrix of samples
        centres = choose_furthest_points(active_deme_locs, dists_between_demes, no_samples) #make 20k attempts to find the points furthest apart
        for m, centre in enumerate(centres):
            demes_in_order = np.argsort(dists_between_demes[centre]) #returns deme indices in ascending order of distance from m
            test_sample_collections[m] = demes_in_order[:no_demes_in_sample] #return the closest few demes, doesn't matter if there are ties
        overlap = (len(np.unique(test_sample_collections)) < no_demes_in_sample*no_samples) #check if overlap
        samples_too_close = overlap #or (min_dist_between_centres - sample_diameter < 3)) #implement separation between sample edges
        #print(overlap, min_dist_between_centres - sample_diameter, sample_diameter)
    else:
        return test_sample_collections, active_deme_locs, active_deme_locs[centres] #we have achieved the correct number of samples

#now choose a number of samples and return pops, clones, counts
def get_sample_clones(active_deme_locs, positions, sample_collections, cell_types, deme_pops, no_samples, cells_per_sample=50000):
    chosen_sample_clone_dict, chosen_sample_pops = [], np.zeros(no_samples, dtype=int) #store each here
    all_clones_seen = np.array([], dtype=np.int64)
    for s in range(no_samples):
        deme_inds = sample_collections[s] #choose deme indices from broader list here
        sample_clones = np.array([], dtype=np.int64)
        sample_pop = 0
        for deme_ind in deme_inds:
            [i, j, k] = active_deme_locs[deme_ind]
            pop, pos= deme_pops[i][j][k], positions[i][j][k]
            sample_pop += pop
            sample_clones = np.concatenate((sample_clones, cell_types[pos][:pop])) #append all types to a sample-wide list
        if sample_pop >= cells_per_sample: #sample roughly 50k of them if we have too many
            print(cells_per_sample, sample_pop)
            cells_sampled = np.where(cells_per_sample/sample_pop > np.random.rand(sample_pop))[0]
            sample_clones = sample_clones[cells_sampled]
            sample_pop = len(cells_sampled)
        sample_clones, sample_counts = np.unique(sample_clones, return_counts=True)
        #assert(np.sum(sample_counts)==sample_pop)
        chosen_sample_clone_dict.append(dict(zip(list(sample_clones), list(sample_counts)))) #create a dictionary of types and their counts
        chosen_sample_pops[s] = sample_pop
        all_clones_seen = np.concatenate((all_clones_seen, sample_clones)) #add unique clones to overall list
    #now get a list of all clones in this sample
    all_clones_seen = np.unique(all_clones_seen)
    return chosen_sample_clone_dict, chosen_sample_pops, all_clones_seen

#a function to construct the tree, given:
#a list of dictionaries, one per samples, linking a type to a count in that sample
#a list of all unique types
def construct_tree(scratch, all_clones_seen, subclone_parentage, muts_per_clone, drivs_per_clone, arrlength, arrno, expno):
    #we do not want to allow all clones to appear in memory at once
    #subclone_parentage records the parents of each type
    #the nth array corresponds to types n*arrlength + (n+1)*arrlength
    #the initial array is n=0; the parentage of type 1 is 1 and is stored there
    #we start with n=arrno; there have been arrno arrays saved into memory before this
    array_n = arrno
    parentage_dict, muts_per_type_dict = {}, {}
    #parentage dict has keys of types and values of parent types
    #muts per type dict has types of keys and values of number of mutations- this is the number lying between each tye and its parent
    to_find_now = all_clones_seen #we want to deprecate this list until it is a list of 1s
    while array_n >= 0:
        range_min, range_max = array_n*arrlength, (array_n+1)*arrlength #this is the range of types for which subclone_parentage currently corresponds
        #to_find_now corresponds to 'candidate types', which are still within range
        #assert(np.max(to_find_now) < range_max) #range is [), so we should always have found descendants before this 
        to_find_here = np.where(to_find_now >= range_min)[0] #these are the indices of types in range
        for index in to_find_here:
            type = to_find_now[index] #the actual type
            anc = type #we want to follow this up the tree until we find something out of range, pulling recorded parentages where we can and recording where we must
            while anc >= range_min and anc > 1: #accept that we might find the root, and if we have, do not attempt to find its parentage
                if anc not in parentage_dict: #if the parent of this type has not already been found and record
                    index_in_list = anc - range_min
                    parent, num_muts = subclone_parentage[index_in_list], muts_per_clone[index_in_list]
                    parentage_dict[anc] = parent
                    muts_per_type_dict[anc] = num_muts
                    anc = parent
                else: #the parent of this type has already been found
                    parent = parentage_dict[anc]
                    anc = parent #replace this with its parent in the list
            else:
                to_find_now[index] = anc
        #at the end of this loop, all of these should be out of range, OR we should have a list of 1s
        #assert(np.max(to_find_now) < range_min or (len(np.unique(to_find_now))==1 and np.unique(to_find_now)[0]==1))
        #assert(np.min(to_find_now) > 0)
        to_find_now = np.unique(to_find_now)  #no need to be tracing the same type many times
        array_n -= 1
        #print(array_n)
        if array_n >= 0:
            subclone_parentage, muts_per_clone= np.load(scratch+"/"+str(expno)+"_subclone_parentage_"+str(array_n)+".npy", allow_pickle=True), np.load(scratch+"/"+str(expno)+"_muts_per_clone_"+str(array_n)+".npy", allow_pickle=True)
    #at the end of this we should have a list of 1s
    #assert(np.min(to_find_now)==1)
    #assert(np.max(to_find_now)==1)
    return parentage_dict, muts_per_type_dict


#get a list of all types in the tree, with a list of instantiating muts per type
def get_mutations_per_clone(parentage_dict, muts_per_type_dict):
    #get a list of mutations for each clone
    all_types = list(parentage_dict.keys()) #all relevant types
    num_types = len(all_types)
    num_muts_seen = 0
    muts_per_type = np.zeros((num_types, 2), dtype=np.int64) #record the first and last indices of the instantiating mutations, in [start, stop) form
    types_to_index_dict = {}
    for type_index, type in enumerate(all_types):
        num_muts = muts_per_type_dict[type]
        muts_per_type[type_index] = [num_muts_seen, num_muts_seen + num_muts]
        num_muts_seen += num_muts
        types_to_index_dict[type] = type_index #records the position of each type in the list, and thus the index you should look at to get the instantiating mutations
    return types_to_index_dict, muts_per_type

#get a dictionary of raw prevalences, to sequence later
def get_prevalences_per_sample(chosen_sample_clone_dict, parentage_dict, types_to_index_dict, muts_per_type, chosen_sample_pops):
    muts_per_sample_dicts = [] #a list of mutation to prevalence dicts, one for each type
    for dict, pop in zip(chosen_sample_clone_dict, chosen_sample_pops):
        muts_per_sample = {} #how many counts are there of each mutation here?
        for clone, count in dict.items():
            #print("clone", clone, "count", count)
            #list_of_muts = []
            #list_of_ancs = []
            anc = clone
            while anc != 1:
                position = types_to_index_dict[anc] #where are the instantiating mutations stored?
                [mut_start, mut_end] = muts_per_type[position] #get the instantiating mutations
                #list_of_muts += list(range(mut_start, mut_end))
                #list_of_ancs.append(anc)
                for mut in range(mut_start, mut_end):
                    if mut in muts_per_sample: #if this mutation is already here
                        muts_per_sample[mut] += count #add the cells
                    else:
                        muts_per_sample[mut] = count
                anc = parentage_dict[anc]
                #print(anc)
        for mut, count in muts_per_sample.items():
            muts_per_sample[mut] = count/pop #divide through to get raw prevalence
        muts_per_sample_dicts.append(muts_per_sample)
    return muts_per_sample_dicts

def sequence(mut_prevalences, read_depth=160, detect_thresh=0.01, min_reads=4, error=0.0001):
    sequenced_mut_dicts = []
    for dict in mut_prevalences:
        sequenced_dict = {}
        for (mut, prev) in dict.items(): #look at the raw prevalences
            if prev >= detect_thresh:
                r = np.random.binomial(read_depth, prev*(1-error))
                if r >= min_reads: #if it passes detection threshold
                    sequenced_dict[mut] = r/read_depth
        sequenced_mut_dicts.append(sequenced_dict)
    return sequenced_mut_dicts
    

#fix this bit, set up full pipeline, start running it AT INTERVALS
#drivers no longer saved into memory

def run(savepath, expname, expno, scratch, s, herit, no_cells_total, total_time, scratch_limit, division_prob=1-np.exp(-1), driv_prob=0.00001, exp_muts=0.6, gridlength=100, deme_size=10000, arrlength=1000000, nr=1000000, no_samples=8, read_depth=160, detect_thresh=0.01, min_reads=4, error=0.0001, cells_per_sample=50000, gen_drivers=1, nongen_drivers=1):
    no_demes_total = int(no_cells_total/deme_size) #by assumption
    epig_rate = driv_prob*exp_muts if nongen_drivers==1 else 0 #by assumption- same as probability of drivers, if using
    #if we are using genetic evolution, turn drivers on; if not, turn them off
    adjusted_driv_prob = driv_prob if gen_drivers==1 else 0
    start = time.time()
    t, cell_types, deme_pops, num_types, positions, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, surf_xs, surf_ys, surf_zs = grow_tumour(expno, scratch, division_prob, adjusted_driv_prob, exp_muts, gridlength, deme_size, s, herit, epig_rate, no_demes_total, no_cells_total,  arrlength, scratch_limit, nr=nr, total_time=total_time)
    end = time.time()
    print(end-start, "seconds to simulate tumour")
    test_sample_collections, active_deme_locs, chosen_sample_locs = divide_samples_into_collections(surf_xs, surf_ys, surf_zs, deme_pops, no_samples=no_samples)
    chosen_sample_clone_dict, chosen_sample_pops, all_clones_seen = get_sample_clones(active_deme_locs, positions, test_sample_collections, cell_types, deme_pops, no_samples, cells_per_sample=cells_per_sample)
    parentage_dict, muts_per_type_dict =  construct_tree(scratch, all_clones_seen, subclone_parentage, muts_per_clone, drivs_per_clone, arrlength, spno, expno)
    types_to_index_dict, muts_per_type = get_mutations_per_clone(parentage_dict, muts_per_type_dict)
    muts = get_prevalences_per_sample(chosen_sample_clone_dict, parentage_dict, types_to_index_dict, muts_per_type, chosen_sample_pops)
    sequenced_muts = sequence(muts, read_depth=read_depth, detect_thresh=detect_thresh, min_reads=min_reads, error=error)
    #now save what you need-- these are raw prevalences
    np.save(savepath+"/"+expname+"_"+str(expno)+"_mutdict.npy", sequenced_muts)
    np.save(savepath+"/"+expname+"_"+str(expno)+"_sample_locs.npy", chosen_sample_locs)



def main():
    parser = argparse.ArgumentParser(description="VPM with replacement.")
    parser.add_argument('--name', type=str, required=True, help='Identifier of experiment name.')
    parser.add_argument('--num', type=int, required=True, help='Number of experiment.')
    parser.add_argument('--savepath', type=str, required=True, help='Directory to save final results.')
    parser.add_argument('--sel', type=float, required=True, help='Value of driver mutation.')
    parser.add_argument('--herit', type=float, required=True, help='Heritability of epigenetic alterations.')
    parser.add_argument('--exp_muts', type=float, required=True, help='Mutations expected per division.')
    parser.add_argument('--no_cells_total', type=float, required=True, help='Total number of cells.')
    parser.add_argument('--total_time', type=float, required=True, help='Total number of days the simulation is run for.')
    parser.add_argument('--scratch', type=str, required=True, help='Filepath to save intermediate results.')
    parser.add_argument('--scratch_limit', type=int, required=True, help='Number of arrays to save into memory.')
    parser.add_argument('--gen_drivers', type=int, required=True, help='Whether or not to use genetic drivers.')
    parser.add_argument('--nongen_drivers', type=int, required=True, help='Whether or not to use nongenetic drivers')
    parser.add_argument('--batch_size', type=int, required=True, help='Number of jobs in a batch')
    args = parser.parse_args()
    for b in range(args.batch_size):
      exp_num = args.batch_size*(args.num-1) + b + 1 #everything is saved into the same scratch space and then loaded back into memory in order-- everything necessary has been overwritten
      try:
        np.load(args.savepath+"/"+args.name+"_"+str(exp_num)+"_mutdict.npy", allow_pickle=True) #to avoid repeating runs
      except:
        run(args.savepath, args.name, exp_num, args.scratch, args.sel, args.herit, int(args.no_cells_total), int(args.total_time), args.scratch_limit, gen_drivers=args.gen_drivers, nongen_drivers=args.nongen_drivers, exp_muts=args.exp_muts)

if __name__ == '__main__':
    main()

