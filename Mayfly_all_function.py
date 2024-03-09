import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols_old import *
import itertools
import pandas as pd
import math
random.seed(3124)
np.random.seed(3124)
def read_input(name_path_input):
    # อ่านไฟล์ CSV เพียงครั้งเดียว
    df_item_sas_random = pd.read_csv('df_item_sas_random.csv')

    # อ่านไฟล์อื่นๆ
    duedate_path = f'{name_path_input}\\duedate_{name_path_input}.csv'
    input_location_path = f'{name_path_input}\\input_location_item_{name_path_input}.csv'
    df_duedate = pd.read_csv(duedate_path, header=None)
    df_item_oder = pd.read_csv(input_location_path, header=None)

    # แปลงเป็น list
    list_duedate = df_duedate[0].tolist()
    num_order = df_item_oder.shape[1]

    # ประมวลผลแต่ละ order และสร้าง DataFrame
    order_items = [df_item_oder[order][df_item_oder[order] != 0] for order in range(num_order)]
    df_item_pools = [
        df_item_sas_random[df_item_sas_random['location'].isin(order_item)].assign(duedate=list_duedate[order],order=order) for order, order_item in
        enumerate(order_items)]

    # รวม DataFrame
    df_item_pool = pd.concat(df_item_pools, ignore_index=True)

    # สร้าง list_order และ list_total_item (ถ้าจำเป็น)
    list_order = [order for order in range(num_order) for _ in range(len(order_items[order]))]
    list_total_item = [item for order_item in order_items for item in order_item.tolist()]

    return df_item_pool, df_item_sas_random

def init_population_and_eval(pop_size, num_item):
    population = [[i for i in range(num_item)] for _ in range(pop_size)]
    evaluations = []
    for individual in population:
        random.shuffle(individual)
        evaluations.append(evaluate_all_sols(individual, df_item_pool, name_path_input))
    return population, evaluations

def process_evaluations(evaluations, pop_size):
    cur_sols, cur_sols_value, cur_arc_sols, arc_sols_cut, velocity_dict = [], [], [], [], []
    for evaluation in evaluations:
        cur_sols.append(sum(evaluation[2], []))
        cur_sols_value.append(evaluation[1])
    for index in range(pop_size):
        cur_arc_sols.append(sol_from_list_to_arc(cur_sols[index]))
        arc_sols_cut.append(cut_arc_sol(cur_arc_sols[index]))
        velocity_dict.append(init_velocity_sol(arc_sols_cut[index]))
    return cur_sols, cur_sols_value, cur_arc_sols, arc_sols_cut, velocity_dict

def sol_from_list_to_arc(sol):
    return [(sol[i], sol[i+1]) for i in range(len(sol) - 1)]


def all_sol_from_list_to_arc(all_sols):
    num_sol = len(all_sols)
    num_item = len(all_sols[0])
    all_arc_sols = [[(all_sols[i][j], all_sols[i][j+1]) for j in range(num_item-1)] for i in range(num_sol)]
    return all_arc_sols

def cut_arc_sol(arc_sol):
    num_item = len(arc_sol) + 1
    arc_sol_list = [[] for _ in range(num_item)]

    for arc in arc_sol:
        arc_sol_list[arc[0]].append(arc)
        arc_sol_list[arc[1]].append(arc)

    arc_sol_cut = [arc_sol_list[item] for item in range(num_item)]

    return arc_sol_cut

def init_velocity_sol(arc_sol_cut, vmax, vmin):
    num_item = len(arc_sol_cut)
    arc_sol_velocity_dict = [{} for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut[item]:
            # Random factor for the velocity
            rand = random.random()
            # Calculate the velocity based on the formula given
            velocity = rand * (vmax - vmin) + vmin
            # Ensure the velocity is within the specified limits
            velocity = min(vmax, max(vmin, velocity))
            arc_sol_velocity_dict[item][arc] = round(velocity, 4)

    return arc_sol_velocity_dict

def coef_times_velocity(coef,arc_sol_velocity_dict):
    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_velocity_dict[item].keys():
            if coef*arc_sol_velocity_dict[item][arc] >1:
                coef_times_velocity_dict[item][arc] = 1
            else:
                coef_times_velocity_dict[item][arc] = round(coef*arc_sol_velocity_dict[item][arc],4)
    return coef_times_velocity_dict

def position_minus_position(arc_first, arc_second):
    num_item = len(arc_first)
    pos_minus_pos = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                pos_minus_pos[item].append(arc)
    return pos_minus_pos

def coef_times_position(c_value, arc_diff):
    import random
    num_item = len(arc_diff)
    coef_times_position_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_diff[item]:
            coef = c_value*random.random()
            if coef > 1:
                coef = 1
            coef_times_position_dict[item][arc] = round(coef,3)
    return coef_times_position_dict

def add_velocity(velocity_first, velocity_second):
    added_velocity_dict = []
    for vf, vs in zip(velocity_first, velocity_second):
        combined_dict = {arc: max(vf.get(arc, 0), vs.get(arc, 0)) for arc in set(vf) | set(vs)}
        added_velocity_dict.append(combined_dict)
    return added_velocity_dict


def check_velocity_inconsistency(added_velocity_dict):
    # Directly iterate over items and their arcs without creating a deep copy.
    for item, arcs in enumerate(added_velocity_dict):
        for arc, velocity in arcs.items():
            # Check and update based on velocity in the arc's originating and ending items
            origin_velocity = added_velocity_dict[arc[0]].get(arc, 0)
            end_velocity = added_velocity_dict[arc[1]].get(arc, 0)
            max_velocity = max(velocity, origin_velocity, end_velocity)

            # Update the current arc's velocity to the maximum found
            added_velocity_dict[item][arc] = max_velocity

    return added_velocity_dict


def creat_cut_set(added_velocity_dict, alpha):
    cut_set = [
        [arc for arc, velocity in item_dict.items() if velocity >= alpha]
        for item_dict in added_velocity_dict
    ]

    return cut_set


def select_dest_from_source(source, picked_list, *sets):
    import random
    # Flatten the sets and filter directly using list comprehension
    eligible_arcs = [
        arc for s in sets for arc in s[source]
        if arc[1] not in picked_list and arc[0] == source
    ]

    if eligible_arcs:
        dest = random.choice(eligible_arcs)[1]
        arc_source_dest = (source, dest)
        return dest, arc_source_dest
    else:
        # Handle the case where no eligible destination is found
        return None, (source, None)

def sol_position_update(cut_set, previous_x, sub_E_list, start_previous_x, start_pbest, start_gbest):
    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []

    # Streamlined random source selection
    all_starts = [start_previous_x, start_pbest, start_gbest] + list(range(num_item))
    source = random.choice(all_starts)
    picked_list.append(source)

    for _ in range(num_item - 1):
        dest, arc_source_dest = select_dest_from_source(source, picked_list, cut_set, previous_x, sub_E_list)
        if dest is None:  # Handle case where no destination is found
            break
        source = dest
        picked_list.append(dest)
        picked_list_arc.append(arc_source_dest)

    return picked_list, picked_list_arc

import random

def mutShuffleIndexes(individual, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 1)
            individual[i], individual[swap_indx] = individual[swap_indx], individual[i]
    return individual


def cxPartialyMatched(ind1, ind2):
    """Executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place. This crossover expects
    :term:`sequence` individuals of indices, the result for any other type of
    individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values in a certain range of the two parents and swapping the values
    of those indexes. For more details see [Goldberg1985]_.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

def compare_and_replace(parents, offspring):
    new_solutions = []
    for parent, child in zip(parents, offspring):
        # เปรียบเทียบค่า tardiness
        if child[1] <= parent[1]:
            new_solutions.append(child)
        else:
            new_solutions.append(parent)
    return new_solutions

def gravity_calculation(gmax, gmin, gen, num_gen):
    gravity = gmax - (((gmax - gmin) / (num_gen)) * gen)
    return gravity

def coef_times_position_mayfly_version(a, B, arc_diff, randomness_scale=0.1):
    num_item = len(arc_diff)
    coef_times_position_dict = [{} for _ in range(num_item)]

    for item, arcs in enumerate(arc_diff):
        r = len(arcs)
        if r > 0:
            coef = gaussian_coefficient(a, B, r)
            # Add randomness to the coefficient
            random_adjustment = (random.random() - 0.5) * 2 * randomness_scale
            coef_with_randomness = coef * (1 + random_adjustment)
            for arc in arcs:
                coef_times_position_dict[item][arc] = round(coef_with_randomness, 3)

    return coef_times_position_dict

def exponential_decay(initial_value, iteration, delta):
    return initial_value * (delta ** iteration)

def partial_shuffle(sol, fraction):
    num_elements_to_shuffle = int(len(sol) * fraction)
    indices = list(range(len(sol)))
    random.shuffle(indices)
    indices_to_shuffle = indices[:num_elements_to_shuffle]

    # Directly swap elements without creating a separate list of selected elements
    for i in range(num_elements_to_shuffle):
        swap_with_idx = random.choice(indices_to_shuffle)
        # Swap elements
        sol[indices_to_shuffle[i]], sol[swap_with_idx] = sol[swap_with_idx], sol[indices_to_shuffle[i]]

def calculate_tardiness_behavior(tardiness_scores):
    # Calculate best tardiness
    best_tardiness = min(tardiness_scores)

    return best_tardiness

# Function to calculate Gaussian coefficient
def gaussian_coefficient(a, B, r):
    return a * (1 - math.exp(-B * r ** 2))


def coef_times_position_with_gaussian(arc_first, arc_second, a, beta):
    # Convert arc_second to a list of sets for fast lookups
    arc_second_sets = [set(arcs) for arcs in arc_second]

    # Pre-calculate random factors if they are independent of arc values
    random_factors = [random.random() for _ in range(len(arc_first))]

    coef_times_position_dict = []

    for item, arcs in enumerate(arc_first):
        arcs_difference = set(arcs) - arc_second_sets[item]
        rp = len(arcs_difference)
        c_value = a * math.exp(-beta * rp ** 2)

        # Construct the dict for this item using dict comprehension
        velocities = {arc: min(1, round(c_value * random_factors[item], 4)) for arc in arcs_difference}
        coef_times_position_dict.append(velocities)

    return coef_times_position_dict


def dynamic_inertia_weight(g_max, g_min, iter, iter_max):
    return g_max - ((g_max - g_min) / iter_max) * iter

def coef_times_velocity_mayfly(g_max, g_min, iter, iter_max, arc_sol_velocity_dict):

    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)]
    g = dynamic_inertia_weight(g_max, g_min, iter, iter_max)
    for item in range(num_item):
        for arc, velocity in arc_sol_velocity_dict[item].items():
            updated_velocity = g * velocity
            coef_times_velocity_dict[item][arc] = 1 if updated_velocity > 1 else round(updated_velocity, 4)
    return coef_times_velocity_dict