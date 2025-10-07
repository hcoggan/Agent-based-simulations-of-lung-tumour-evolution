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
import gc

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
        np.save(scratch+"_"+arrname+"_"+str(arrno)+".npy", arr) #save array
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
#cells die at a death rate modulated by fitness

#force cells in a full deme to compete based on fitness
def cull_cells_in_one_deme_with_death_rate(x, y, z, cell_types, cell_driv_muts,  positions, deme_pops, deme_size, s, death_rate):
    pos, pop = positions[x][y][z], deme_pops[x][y][z]
    #find properties
    if pop > deme_size:
        types_here, driv_muts_here = cell_types[pos][:pop], cell_driv_muts[pos][:pop]
        survival_fitnesses = np.power(1+s, driv_muts_here)
        av_fit = np.average(survival_fitnesses)
        survival_probs = (1-death_rate)*survival_fitnesses/av_fit
        #now decide which cells live
        surviving = np.where(np.random.rand(pop) <= survival_probs)[0] #indices of surviving cells
        types_here, driv_muts_here = types_here[surviving], driv_muts_here[surviving] #properties of surviving cells
        new_pop = len(surviving)
        #wipe properties of existing deme
        cell_types[pos], cell_driv_muts[pos] = 0, 0
        cell_types[pos][:new_pop] = types_here
        cell_driv_muts[pos][:new_pop] = driv_muts_here
        #now update population
        deme_pops[x][y][z] = new_pop
    return cell_types, cell_driv_muts, deme_pops




#simplified version of the above, that copies one deme at a time
#I am no longer going to bother pre-generating random numbers, let's not overcomplicate things here
def single_deme_division_np(x, y, z, cell_types, cell_driv_muts, deme_pops, num_types, positions, exp_muts, deme_size, s, division_prob, driv_prob):
    pos, pop = positions[x][y][z], deme_pops[x][y][z]
    #find properties
    types_here, driv_muts_here = cell_types[pos][:pop], cell_driv_muts[pos][:pop]

    #okay so the error doesn't come in in this function, it's somewhere else- between division
    #assert(np.min(types_here) >= 1)
    #calculate division fitness
    division_probs_here = division_prob + (1-division_prob)*(1-np.power((1-s), driv_muts_here))
    dividing_decisions = np.random.rand(pop)
    dividing, not_dividing = np.where(dividing_decisions <= division_probs_here)[0], np.where(dividing_decisions > division_probs_here)[0]
    num_new_cells = 2*len(dividing) #by this we mean the number of cells which will be 'freshly divided'
    num_not_dividing = len(not_dividing)
    
    #make two copies of all dividing cells
    div_types, div_driv_muts= np.hstack((types_here[dividing], types_here[dividing])), np.hstack((driv_muts_here[dividing], driv_muts_here[dividing]))
    #keep not-dividing cells 
    non_div_types, non_div_driv_muts = types_here[not_dividing], driv_muts_here[not_dividing]

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
    

    #now all dividing information has been updated, update the deme
    new_cell_pop = num_new_cells + len(not_dividing)
    cell_types[pos], cell_driv_muts[pos] = 0, 0
    cell_types[pos][:new_cell_pop], cell_driv_muts[pos][:new_cell_pop] = np.hstack((div_types, non_div_types)), np.hstack((div_driv_muts, non_div_driv_muts))
    deme_pops[x][y][z] = new_cell_pop

    #assert(np.min(cell_types[pos][:new_cell_pop]) >= 1)

    #now return all of this
    return cell_types, cell_driv_muts, deme_pops, num_types+num_new_types, num_new_types, new_subclone_parentages, new_mut_nos, new_driv_nos
    

    
    

#return a list of all neighbours and the locations of empty neighbours

def get_empty_neighbours(x, y, z, deme_pops, neighbour_coords):
    #check if it still has empty neighbour demes
    neighbours_here = np.add([x, y, z], neighbour_coords)
    xs_e, ys_e, zs_e = neighbours_here[:, 0], neighbours_here[:, 1], neighbours_here[:, 2]
    neighbour_pops = np.array([deme_pops[xs_e[r]][ys_e[r]][zs_e[r]] for r in range(len(neighbours_here))])
    empty_neighbours = np.where(neighbour_pops==0)[0]
    return xs_e, ys_e, zs_e, empty_neighbours

#return a list of all neighbours and the locations of occupied neighbours OVER CAPACITY
def get_overspilling_neighbours(x, y, z, deme_pops, neighbour_coords, deme_size):
    #check if it still has empty neighbour demes
    neighbours_here = np.add([x, y, z], neighbour_coords)
    xs_e, ys_e, zs_e = neighbours_here[:, 0], neighbours_here[:, 1], neighbours_here[:, 2]
    neighbour_pops = np.array([deme_pops[xs_e[r]][ys_e[r]][zs_e[r]] for r in range(len(neighbours_here))])
    overspilling_neighbours = np.where(neighbour_pops>=deme_size)[0] #we're assuming that competition might have reduced population, but not by this much- this is necessary to prevent dual expansion
    return xs_e, ys_e, zs_e, overspilling_neighbours





#split this into two parts
#this function checks if any demes have empty neighbours and are under capacity; if they are, the cells within them divide and die
#it returns all information and a boolean indicating whether all demes with empty neighbours are full
def initial_growth_divide_and_death(cell_types, cell_driv_muts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, necrotic_marker, ur_nos, ur_tracker, s, death_rate, division_prob, driv_prob, exp_muts, arrlength, spno, mno, dno, expno, deme_size, scratch, mut_rand_nos, mut_tracker, xs, ys, zs, scratch_limit, no_demes_total=10000, nr=1000000000):
    #find occupied demes
    all_demes_full = True #records if any demes with empty neighbours are over capacity
    
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
            #RULES: IF DEME IS UNDER CAPACITY, IT DIVIDES AND THEN 
            if no_cells_here < deme_size: 
                #CELLS CAN ONLY DIVIDE IF ENOUGH SPACE
                cell_types, cell_driv_muts, deme_pops, num_types, num_new_types, new_subclone_parentages, new_mut_nos, new_driv_nos =  single_deme_division_np(x, y, z, cell_types, cell_driv_muts, deme_pops, num_types, positions, exp_muts, deme_size, s, division_prob, driv_prob)
                #now cull on a per deme level
                if num_new_types > 0: #save things into memory
                    subclone_parentage, spno = put_or_save(scratch, subclone_parentage, num_types-num_new_types+1, num_types+1, new_subclone_parentages, arrlength, "subclone_parentage", spno, expno, scratch_limit)
                    muts_per_clone, mno = put_or_save(scratch, muts_per_clone, num_types-num_new_types+1, num_types+1, new_mut_nos, arrlength, "muts_per_clone", mno, expno, scratch_limit)
                    #drivs_per_clone, dno = put_or_save(scratch, drivs_per_clone, num_types-num_new_types+1, num_types+1, new_driv_nos, arrlength, "drivs_per_clone", dno, expno, scratch_limit)
                no_cells_here = deme_pops[x][y][z]
                #COMPETE FOR SURVIVAL IF FULL
                if no_cells_here > 0.1*deme_size: #to mimic resource-dependent competition
                    cell_types, cell_driv_muts, deme_pops = cull_cells_in_one_deme_with_death_rate(x, y, z, cell_types, cell_driv_muts, positions, deme_pops, deme_size, s, death_rate)
                    no_cells_here = deme_pops[x][y][z]    
                if no_cells_here < deme_size:
                    all_demes_full = False #if it is still under capacity at the end of a timestep, even after division and death, then we can't have a splitting event yet
            #if it is over capacity, nothing happens
    return cell_types, cell_driv_muts, deme_pops, num_types, positions, necrotic_marker, ur_nos, ur_tracker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs, all_demes_full

#split this into two parts
#this function accepts that all demes are over-capacity
#it makes a list of the empty neighbours of all demes which have them
#then demes compete on fitness for their empty neighbouring demes, and expand into them
def expansion_divide_and_death(cell_types, cell_driv_muts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, necrotic_marker, ur_nos, ur_tracker, s, division_prob, driv_prob, exp_muts, arrlength, spno, mno, dno, expno, deme_size, scratch, mut_rand_nos, mut_tracker, xs, ys, zs, scratch_limit, no_demes_total=10000, nr=1000000000):

    print("Expansion")
    #find occupied demes
    #check populations here 
    no_occupied_demes = len(xs)
    print(no_occupied_demes, "before expansion")
    
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
    all_empty_neighbours = []
    for n in order:
        x, y, z = xs[n], ys[n], zs[n]
        pos = positions[x][y][z]
        no_cells_here = deme_pops[x][y][z]
        #print("pop check", cell_types[pos][:no_cells_here])
        #assert(np.min(cell_types[pos][:no_cells_here]) >= 1)
        #check if it still has empty neighbour demes
        xs_e, ys_e, zs_e, empty_neighbours = get_empty_neighbours(x, y, z, deme_pops, neighbour_coords)
        if len(empty_neighbours) == 0:
            necrotic_marker[x][y][z] = 1 #mark as necrotic, even if already necrotic- this is definitely unnecessary but can't hurt
        else: #IF THERE ARE ANY EMPTY NEIGHBOURS, ADD THESE- by definition this will not include the neighbours of necrotic cells
            for index in empty_neighbours:
                all_empty_neighbours.append([xs_e[index], ys_e[index], zs_e[index]])
    all_empty_neighbours = np.array(all_empty_neighbours)
    #now get a unique list of all empty demes neighbouring full demes
    all_empty_neighbours = np.unique(all_empty_neighbours, axis=0)
    #shuffle these
    reordered_indices = np.arange(len(all_empty_neighbours))
    np.random.shuffle(reordered_indices)
    all_empty_neighbours = all_empty_neighbours[reordered_indices]
    for [x_ov, y_ov, z_ov] in all_empty_neighbours: #iterate over all neighbours
        #get all full neighbours (i.e. those over capacity)
        xs_e, ys_e, zs_e, overspilling_neighbours = get_overspilling_neighbours(x_ov, y_ov, z_ov, deme_pops, neighbour_coords, deme_size)
        if len(overspilling_neighbours) > 0: #choose one to expand into this deme, proportionally to fitness
            av_deme_fitnesses = []
            for index in overspilling_neighbours: #for each overspilling neighbour
                x, y, z = xs_e[index], ys_e[index], zs_e[index]
                pos, pop = positions[x][y][z], deme_pops[x][y][z]
                av_deme_fitnesses.append(np.average(np.power(1+s, cell_driv_muts[pos][:pop]))) 
            av_deme_fitnesses = np.array(av_deme_fitnesses)
            prob_of_choice = av_deme_fitnesses/np.sum(av_deme_fitnesses) #choose one proportionally to fitnesses
            chosen_index = np.random.choice(overspilling_neighbours, p=prob_of_choice) #choose an index from overspilling_neighbours, a list of indices of overspilling neighbours in xs_e
            x, y, z = xs_e[chosen_index], ys_e[chosen_index], zs_e[chosen_index]
            #print("movement from ", x, y, z, " to ", x_ov, y_ov, z_ov)
            #now move cells from x, y, z to x_ov, y_ov, z_ov 
            ur_nos, ur_tracker = update_random_numbers(ur_nos, ur_tracker, nr, 4*deme_size)
            no_cells_here, pos = deme_pops[x][y][z], positions[x][y][z]
            pos_ov = no_occupied_demes #assign it next position in list
            #print("Last position", pos_ov, "cell_type length", len(cell_types))
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
            cell_types[pos_ov][:no_moving], cell_driv_muts[pos_ov][:no_moving] = cell_types[pos][moving], cell_driv_muts[pos][moving]
            deme_pops[x_ov][y_ov][z_ov] = no_moving
            cell_types[pos][:no_staying], cell_driv_muts[pos][:no_staying] = cell_types[pos][staying], cell_driv_muts[pos][staying]
            cell_types[pos][no_staying:], cell_driv_muts[pos][no_staying:]= 0, 0 #wipe the rest of the existing ones
            deme_pops[x][y][z] = no_staying
            no_occupied_demes += 1 #it will be recalculated at the start of each step, this is to keep it up to date within it
            #assert(np.min(cell_types[pos][:no_staying]) >= 1)
            #assert(np.min(cell_types[pos_ov][:no_moving]) >= 1)
            #print("post-movement population check", np.sum(deme_pops))
            #run until the first timestep where this passes   
    #OVERALL
    #every empty deme is only iterated over once 
    #we check whether demes are still full before allowing them to expand, so nothing can expand more than once or be expanded into more than once
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
    print(no_occupied_demes, "after expansion")
    return cell_types, cell_driv_muts, deme_pops, num_types, positions, necrotic_marker, ur_nos, ur_tracker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs
                
                
                                
                
                
def run_to_full_size(expno, scratch, s, death_rate, division_prob, driv_prob, exp_muts, no_demes_total, no_cells_total, gridlength, deme_size, scratch_limit, nr=100000000, arrlength=1000000):
    num_types = 1
    c = int(gridlength/2)
    xs, ys, zs = np.array([c]), np.array([c]), np.array([c]) #initial list of positions
    subclone_parentage, muts_per_clone, drivs_per_clone = np.zeros(arrlength, dtype=np.int64), np.zeros(arrlength, dtype=np.int64), np.zeros(arrlength, dtype=np.int64)
    necrotic_marker = np.zeros((gridlength, gridlength, gridlength), dtype=np.int8)
    cell_types = np.zeros((int(2*no_demes_total), 2*deme_size), dtype=np.int64) #we may need more than 1000 demes
    cell_driv_muts = np.zeros((int(2*no_demes_total), 2*deme_size), dtype=np.int64)
    print("Size in memory, roughly", 2*cell_types.nbytes)
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
        #allow cells to divide and die
        cell_types, cell_driv_muts, deme_pops, num_types, positions, necrotic_marker, ur_nos, ur_tracker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs, all_demes_full = initial_growth_divide_and_death(cell_types, cell_driv_muts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, necrotic_marker, ur_nos, ur_tracker, s, death_rate, division_prob, driv_prob, exp_muts, arrlength, spno, mno, dno, expno, deme_size, scratch, mut_rand_nos, mut_tracker, xs, ys, zs, scratch_limit, no_demes_total=no_demes_total, nr=nr)
        #if all full, allow expansion
        if all_demes_full:
            cell_types, cell_driv_muts, deme_pops, num_types, positions, necrotic_marker, ur_nos, ur_tracker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs = expansion_divide_and_death(cell_types, cell_driv_muts, deme_pops, subclone_parentage, muts_per_clone, drivs_per_clone, num_types, positions, necrotic_marker, ur_nos, ur_tracker, s, division_prob, driv_prob, exp_muts, arrlength, spno, mno, dno, expno, deme_size, scratch, mut_rand_nos, mut_tracker, xs, ys, zs, scratch_limit, no_demes_total=no_demes_total, nr=nr)
        total_cells = np.sum(deme_pops)
        if total_cells == 0:
          raise Exception("Tumour died off.")
        num_occupied_demes = len(np.where(deme_pops > 0)[0])
        print(total_cells, "cells", num_types, "types", num_occupied_demes, "occupied demes")
        t+= 1
    return t, cell_types, cell_driv_muts, deme_pops, num_types, positions, necrotic_marker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs
        


 #now narrow this down to only those above necrotic core
#take existing necrosis markers and discard all those below surface
def keep_surface(cell_types, cell_driv_muts, positions, deme_pops, necrotic_marker, gridlength, xs, ys, zs):
    surface = np.where(necrotic_marker[xs, ys, zs]==0)[0] #is on surface- we know this is accurate, we kept it
    surf_xs, surf_ys, surf_zs = xs[surface], ys[surface], zs[surface]
    kept_positions = positions[surf_xs, surf_ys, surf_zs]
    new_positions = np.zeros((gridlength, gridlength, gridlength), dtype=np.int64) #record positions of non-necrotic demes in new list
    new_positions[surf_xs, surf_ys, surf_zs] = np.arange(len(surface)) #once we condense list down, this is where the surface demes
    return cell_types[kept_positions], cell_driv_muts[kept_positions], new_positions, surf_xs, surf_ys, surf_zs


def grow_tumour(expno, scratch, division_prob, driv_prob, exp_muts, gridlength, deme_size, s, death_rate, no_demes_total, no_cells_total, arrlength, scratch_limit, nr=10000000, total_time=365):
    t, cell_types, cell_driv_muts, deme_pops, num_types, positions, necrotic_marker, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, xs, ys, zs  = run_to_full_size(expno, scratch, s, death_rate, division_prob, driv_prob, exp_muts, no_demes_total, no_cells_total, gridlength, deme_size, scratch_limit, nr=nr, arrlength=arrlength)
    cell_types, cell_driv_muts, positions, surf_xs, surf_ys, surf_zs = keep_surface(cell_types, cell_driv_muts, positions, deme_pops, necrotic_marker, gridlength, xs, ys, zs)
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
            subclone_parentage, muts_per_clone= np.load(scratch+"_subclone_parentage_"+str(array_n)+".npy", allow_pickle=True), np.load(scratch+"_muts_per_clone_"+str(array_n)+".npy", allow_pickle=True)
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


def run(savepath, expname, expno, scratch, s, death_rate, no_cells_total, total_time, scratch_limit, division_prob=1-np.exp(-1), driv_prob=0.00001, exp_muts=0.6, gridlength=100, deme_size=10000, arrlength=1000000, nr=1000000, no_samples=8, read_depth=160, detect_thresh=0.01, min_reads=4, error=0.0001, cells_per_sample=50000, gen_drivers=1):
    no_demes_total = int(no_cells_total/deme_size) #by assumptio
    #if we are using genetic evolution, turn drivers on; if not, turn them off
    adjusted_driv_prob = driv_prob if gen_drivers==1 else 0
    start = time.time()
    t, cell_types, deme_pops, num_types, positions, subclone_parentage, muts_per_clone, drivs_per_clone, spno, mno, dno, surf_xs, surf_ys, surf_zs = grow_tumour(expno, scratch, division_prob, adjusted_driv_prob, exp_muts, gridlength, deme_size, s, death_rate, no_demes_total, no_cells_total,  arrlength, scratch_limit, nr=nr, total_time=total_time)
    end = time.time()
    print(end-start, "seconds to simulate tumour")
    test_sample_collections, active_deme_locs, chosen_sample_locs = divide_samples_into_collections(surf_xs, surf_ys, surf_zs, deme_pops, no_samples=no_samples)
    chosen_sample_clone_dict, chosen_sample_pops, all_clones_seen = get_sample_clones(active_deme_locs, positions, test_sample_collections, cell_types, deme_pops, no_samples, cells_per_sample=cells_per_sample)
    parentage_dict, muts_per_type_dict =  construct_tree(scratch, all_clones_seen, subclone_parentage, muts_per_clone, drivs_per_clone, arrlength, spno, expno)
    types_to_index_dict, muts_per_type = get_mutations_per_clone(parentage_dict, muts_per_type_dict) #no np.array is modified except all_clones_seen, and that is not reused
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
    parser.add_argument('--death_rate', type=float, required=True, help='Death rate.')
    parser.add_argument('--exp_muts', type=float, required=True, help='Mutations expected per division.')
    parser.add_argument('--no_cells_total', type=float, required=True, help='Total number of cells.')
    parser.add_argument('--total_time', type=float, required=True, help='Total number of days the simulation is run for.')
    parser.add_argument('--scratch', type=str, required=True, help='Filepath to save intermediate results.')
    parser.add_argument('--scratch_limit', type=int, required=True, help='Number of arrays to save into memory.')
    parser.add_argument('--batch_size', type=int, required=True, help='Size of batch.')
    args = parser.parse_args()
    scratch = args.scratch + "/"+ str(args.num)
    for b in range(args.batch_size):
        exp_num=(args.num-1)*args.batch_size + b + 1
        try:
          np.load(args.savepath+"/"+args.name+"_"+str(exp_num)+"_mutdict.npy", allow_pickle=True) #to avoid repeating runs
        except:
          run(args.savepath, args.name, exp_num, scratch, args.sel, args.death_rate, int(args.no_cells_total), int(args.total_time), args.scratch_limit, exp_muts=args.exp_muts)
          gc.collect()
if __name__ == '__main__':
    main()


