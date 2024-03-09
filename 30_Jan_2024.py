import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols_old import *
import itertools
import pandas as pd
from Mayfly_all_function import *
import concurrent.futures
import copy
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
    num_item = len(sol)
    arc_sol = []
    for i in range(num_item-1):
        arc_sol.append((sol[i],sol[i+1]))
    return arc_sol

def all_sol_from_list_to_arc(all_sols):
    num_sol = len(all_sols)
    num_item = len(all_sols[0])
    all_arc_sols = [[(all_sols[i][j], all_sols[i][j+1]) for j in range(num_item-1)] for i in range(num_sol)]
    return all_arc_sols

def cut_arc_sol(arc_sol): #argument เป็น arc_sol ของคำตอบเดียว
    num_item = len(arc_sol)+1
    arc_sol_cut = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol:
            if item == arc[0] or item == arc[1]:
                arc_sol_cut[item].append(arc)
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
    #[{}, {}, {}, ...]
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
    num_item = len(velocity_first)
    added_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in velocity_first[item]:
            added_velocity_dict[item][arc] = velocity_first[item][arc]
        for arc in velocity_second[item]:
            if arc in added_velocity_dict[item].keys():
                if velocity_second[item][arc] > added_velocity_dict[item][arc]:
                    added_velocity_dict[item][arc] = velocity_second[item][arc]
            else:
                added_velocity_dict[item][arc] = velocity_second[item][arc]
    return added_velocity_dict

def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    import copy
    new_added_velocity_dict = [{arc:prob for arc,prob in added_velocity_dict[item].items()}for item in range(num_item)]
    for item in range(num_item):
        for arc_first in added_velocity_dict[item].keys():
            if arc_first in added_velocity_dict[arc_first[0]].keys():
                if added_velocity_dict[item][arc_first]< added_velocity_dict[arc_first[0]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[0]][arc_first]
            if arc_first in added_velocity_dict[arc_first[1]].keys():
                if added_velocity_dict[item][arc_first]< added_velocity_dict[arc_first[1]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[1]][arc_first]
    return new_added_velocity_dict

def creat_cut_set(added_velocity_dict,alpha):
    num_item = len(added_velocity_dict)
    cut_set = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc] >= alpha:
                cut_set[item].append(arc)
    return cut_set

def select_dest_from_source(source, picked_list, *sets):
    import random
    for set in sets:
        new_set = []
        if len(set[source])>0:
            for arc in set[source]:
                if arc[1] not in picked_list and arc[0] == source:
                    new_set.append(arc)
        if len(new_set)>0:
            dest = random.choice(new_set)[1]
            break
    arc_source_dest = (source,dest)
    return dest, arc_source_dest

def sol_position_update(cut_set, previous_x, sub_E_list, start_previous_x, start_pbest, start_gbest):
    import random

    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []

    source = random.choice([start_previous_x, start_pbest, start_gbest, random.choice(range(num_item))])
    picked_list.append(source)

    for item_counter in range(num_item-1):
        dest, arc_source_dest = select_dest_from_source(source, picked_list, cut_set, previous_x, sub_E_list)
        source = dest
        picked_list.append(dest)
        picked_list_arc.append(arc_source_dest)

    return picked_list, picked_list_arc


def mutate(ind, max_mutation_strength, current_gen, total_gens):
    """Applies a dynamic mutation to the individual based on the current generation."""
    mutation_strength = max_mutation_strength * (1 - (current_gen / total_gens))
    num_elements_to_shuffle = int(len(ind) * mutation_strength)
    indices_to_shuffle = random.sample(range(len(ind)), num_elements_to_shuffle)
    selected_elements = [ind[i] for i in indices_to_shuffle]
    random.shuffle(selected_elements)
    for i, idx in enumerate(indices_to_shuffle):
        ind[idx] = selected_elements[i]


def cxPartialyMatched(ind1, ind2,max_mutation_strength,gbest,current_gen, total_gens):
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

    # Check for local optima by comparing with gbest
    if ind1 == gbest or ind2 == gbest:
        # Apply strong mutation to both individuals based on the dynamic strength
        mutate(ind1, max_mutation_strength, current_gen, total_gens)
        mutate(ind2, max_mutation_strength, current_gen, total_gens)

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

# Function to calculate Gaussian coefficient
def gaussian_coefficient(a, B, r):
    import math
    return a * (1 - math.exp(-B * r))

def coef_times_position_mayfly_version(a, B, arc_diff):

    num_item = len(arc_diff)
    coef_times_position_dict = [{} for _ in range(num_item)]

    for item, arcs in enumerate(arc_diff):
        # The distance 'r' is the number of differing arcs
        r = len(arcs)
        # If there are no differing arcs, we can skip the computation
        if r > 0:
            coef = gaussian_coefficient(a, B, r)
            for arc in arcs:
                coef_times_position_dict[item][arc] = round(coef, 3)

    return coef_times_position_dict

def exponential_decay(initial_value, iteration, delta):
    return initial_value * (delta ** iteration)

def partial_shuffle(sol, fraction):
    num_elements_to_shuffle = int(len(sol) * fraction)
    indices_to_shuffle = random.sample(range(len(sol)), num_elements_to_shuffle)
    selected_elements = [sol[i] for i in indices_to_shuffle]
    random.shuffle(selected_elements)
    for i, idx in enumerate(indices_to_shuffle):
        sol[idx] = selected_elements[i]

def calculate_tardiness_behavior(tardiness_scores):
    # Calculate best tardiness
    best_tardiness = min(tardiness_scores)

    return best_tardiness


def mayfly(name_path_input, num_gen, pop_size, *parameters):
    a1, a2,a3, beta, gmax,gmin,vmax,vmin,delta,nuptial_dance,random_flight,alpha,mutation_strength = parameters
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    half_pop_size = pop_size // 2
    E_all = list(itertools.permutations(range(num_item), 2))
    sub_E_list = [[arc for arc in E_all if arc[0] == item or arc[1] == item] for item in range(num_item)]

    # Initialize gbest and pbest values for males and females
    gbest_value, gbest_sol, gbest_arc_sol_cut = [100000], [], []
    male_pbest_value,female_pbest_value = [100000] * half_pop_size,[100000] * half_pop_size
    male_pbest_sol , female_pbest_sol = [[] for _ in range(half_pop_size)],[[] for _ in range(half_pop_size)]
    male_pbest_arc_sols , female_pbest_arc_sols = [[] for _ in range(half_pop_size)],[[] for _ in range(half_pop_size)]
    male_pbest_arc_sols_cut , female_pbest_arc_sols_cut = [[] for _ in range(half_pop_size)],[[] for _ in range(half_pop_size)]
    def process_mayfly_population(population):
        evaluations = []
        cur_sols = []
        cur_sols_value = []
        cur_arc_sols = []
        arc_sols_cut = []
        velocity_dict = []

        for mayfly in population:
            random.shuffle(mayfly)
            evaluation = evaluate_all_sols(mayfly, df_item_pool, name_path_input)
            evaluations.append(evaluation)
            cur_sols.append(sum(evaluation[2], []))
            cur_sols_value.append(evaluation[1])
            cur_arc_sols.append(sol_from_list_to_arc(cur_sols[-1]))
            arc_sols_cut.append(cut_arc_sol(cur_arc_sols[-1]))
            velocity_dict.append(init_velocity_sol(arc_sols_cut[-1], vmax, vmin))

        return evaluations, cur_sols, cur_sols_value, cur_arc_sols, arc_sols_cut, velocity_dict

    # Initialize male and female populations
    male_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]
    female_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]

    # Parallel processing for male and female populations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        male_future = executor.submit(process_mayfly_population, male_mayfly_population)
        female_future = executor.submit(process_mayfly_population, female_mayfly_population)

        male_results = male_future.result()
        female_results = female_future.result()
    gbest_per_gen = []
    male_value_history = []
    female_value_history  = []
    male_evaluations, male_cur_sols, male_cur_sols_value, male_cur_arc_sols, male_arc_sols_cut, male_velocity_dict = male_results
    female_evaluations, female_cur_sols, female_cur_sols_value, female_cur_arc_sols, female_arc_sols_cut, female_velocity_dict = female_results
    for gen in range(num_gen):
        # ---------------------- Mayfly Male Section ----------------------
        for sol in range(half_pop_size):
            current_value = male_cur_sols_value[sol]
            current_sol = male_cur_sols[sol]
            current_arc_sol = sol_from_list_to_arc(current_sol)
            current_arc_sol_cut = cut_arc_sol(current_arc_sol)

            # Update personal best if current solution is better or equal
            if current_value <= male_pbest_value[sol]:
                male_pbest_value[sol] = current_value
                male_pbest_sol[sol] = copy.deepcopy(current_sol)
                male_pbest_arc_sols[sol] = copy.deepcopy(current_arc_sol)
                male_pbest_arc_sols_cut[sol] = copy.deepcopy(current_arc_sol_cut)

                # Update global best if current personal best is better
                if current_value <= gbest_value:
                    gbest_value = current_value
                    gbest_sol = copy.deepcopy(current_sol)
                    gbest_arc_sol_cut = copy.deepcopy(current_arc_sol_cut)
                    #Additional checks and updates
                    if current_value == gbest_value:
                        # Calculate the fraction of the solution to shuffle based on the current iteration and exponential decay
                        nuptial_dance_fraction = exponential_decay(nuptial_dance, gen,delta)  # assuming initial value is 1 for full shuffle
                        partial_shuffle(male_cur_sols[sol], nuptial_dance_fraction)
                        male_cur_arc_sols[sol] = copy.deepcopy(sol_from_list_to_arc(male_cur_sols[sol]))
                        male_arc_sols_cut[sol] = copy.deepcopy(cut_arc_sol(male_cur_arc_sols[sol]))
                        male_velocity_dict[sol] = copy.deepcopy(init_velocity_sol(male_arc_sols_cut[sol], vmax, vmin))

        # Initialize lists before the loop
        male_coef_velocity = []
        male_pbest_diff = []
        male_gbest_diff = []
        male_added_pbest_gbest = []
        male_added_velocity = []
        male_velocity_check_incon = []
        male_cut_set = []
        male_new_pos = []
        male_evaluations_new_pos = []

        for sol in range(half_pop_size):
            coef_velocity = coef_times_velocity(gravity_calculation(gmax, gmin, gen, num_gen), male_velocity_dict[sol])
            pbest_diff = coef_times_position_mayfly_version(a1,beta,position_minus_position(male_pbest_arc_sols_cut[sol], male_arc_sols_cut[sol]))
            gbest_diff = coef_times_position_mayfly_version(a2,beta, position_minus_position(gbest_arc_sol_cut, male_arc_sols_cut[sol]))
            added_pbest_gbest = add_velocity(gbest_diff, pbest_diff)
            added_velocity = add_velocity(coef_velocity, added_pbest_gbest)
            velocity_check_incon = check_velocity_inconsistency(added_velocity)
            cut_set = creat_cut_set(velocity_check_incon, alpha )
            new_pos = sol_position_update(cut_set, male_arc_sols_cut[sol], sub_E_list, male_cur_sols[sol][0], male_pbest_sol[sol][0],gbest_sol[0])[0]

            # Append the results to their respective lists
            male_coef_velocity.append(coef_velocity)
            male_pbest_diff.append(pbest_diff)
            male_gbest_diff.append(gbest_diff)
            male_added_pbest_gbest.append(added_pbest_gbest)
            male_added_velocity.append(added_velocity)
            male_velocity_check_incon.append(velocity_check_incon)
            male_cut_set.append(cut_set)
            male_new_pos.append(new_pos)
            male_evaluations_new_pos.append(evaluate_all_sols(new_pos, df_item_pool, name_path_input))

        # # Update male mayfly new solutions
        # for sol in range(half_pop_size):
        #     # Update current solution values and solutions if the new solution is better
        #     if male_new_cur_sols_value[sol] <= male_cur_sols_value[sol]:
        #         male_cur_sols_value[sol] = male_new_cur_sols_value[sol]
        #         # Using deep copy only if necessary
        #         male_cur_sols[sol] = copy.deepcopy(male_new_cur_sols[sol])

        # ---------------------- Female Mayfly Section ----------------------
        for sol in range(half_pop_size):
            # Update personal best if the current solution is better
            if female_cur_sols_value[sol] <= female_pbest_value[sol]:
                female_pbest_value[sol] = female_cur_sols_value[sol]
                female_pbest_sol[sol] = copy.deepcopy(female_cur_sols[sol])
                female_pbest_arc_sols[sol] = copy.deepcopy(sol_from_list_to_arc(female_pbest_sol[sol]))
                female_pbest_arc_sols_cut[sol] = copy.deepcopy(cut_arc_sol(female_pbest_arc_sols[sol]))

                # Update global best if the current personal best is better
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    gbest_sol = copy.deepcopy(female_pbest_sol[sol])
                    gbest_arc_sol_cut = copy.deepcopy(female_pbest_arc_sols_cut[sol])

            # Additional checks and updates
            if female_cur_sols_value[sol] <= male_cur_sols_value[sol]:
                # Calculate the fraction of the solution to shuffle based on the current generation and exponential decay
                random_flight_fraction = exponential_decay(random_flight, gen, delta)  # Using 'gen' as the current generation count
                partial_shuffle(female_cur_sols[sol], random_flight_fraction)
                female_cur_arc_sols[sol] = copy.deepcopy(sol_from_list_to_arc(female_cur_sols[sol]))
                female_arc_sols_cut[sol] = copy.deepcopy(cut_arc_sol(female_cur_arc_sols[sol]))
                female_velocity_dict[sol] = copy.deepcopy(init_velocity_sol(female_arc_sols_cut[sol], vmax, vmin))

        # Initialize lists before the loop
        female_coef_velocity = []
        female_male_diff = []
        female_added_velocity = []
        female_velocity_check_incon = []
        female_cut_set = []
        female_new_pos = []
        female_evaluations_new_pos = []

        for sol in range(half_pop_size):
            coef_velocity = coef_times_velocity(gravity_calculation(gmax, gmin, gen, num_gen), female_velocity_dict[sol])
            male_diff = coef_times_position_mayfly_version(a3,beta,position_minus_position(male_arc_sols_cut[sol], female_arc_sols_cut[sol]))
            added_velocity = add_velocity(coef_velocity, male_diff)
            velocity_check_incon = check_velocity_inconsistency(added_velocity)
            cut_set = creat_cut_set(velocity_check_incon, alpha )
            new_pos = sol_position_update(cut_set, female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0]
            evaluation = evaluate_all_sols(new_pos, df_item_pool, name_path_input)

            female_coef_velocity.append(coef_velocity)
            female_male_diff.append(male_diff)
            female_added_velocity.append(added_velocity)
            female_velocity_check_incon.append(velocity_check_incon)
            female_cut_set.append(cut_set)
            female_new_pos.append(new_pos)
            female_evaluations_new_pos.append(evaluation)

        sorted_male_evaluations = sorted(male_evaluations_new_pos, key=lambda x: x[1])
        sorted_female_evaluations = sorted(female_evaluations_new_pos, key=lambda x: x[1])
        # Use list comprehensions for processing evaluations
        male_new_cur_sols = [sum(evaluation[2], []) for evaluation in sorted_male_evaluations]
        female_new_cur_sols = [sum(evaluation[2], []) for evaluation in sorted_female_evaluations]

        # Offspring generation and evaluation
        offspring_1 = []
        offspring_2 = []
        for sol in range(half_pop_size):
            offspring_pair = cxPartialyMatched(copy.deepcopy(male_new_cur_sols[sol]),copy.deepcopy(female_new_cur_sols[sol]), gbest_sol,mutation_strength,gen,num_gen)
            offspring_1.append(offspring_pair[0])
            offspring_2.append(offspring_pair[1])

        # Combining and evaluating offspring in one loop
        offspring_combine = offspring_1 + offspring_2
        offspring_evaluations = [evaluate_all_sols(offspring, df_item_pool, name_path_input) for offspring in offspring_combine]

        # Separate offspring to male and female randomly
        random.shuffle(offspring_evaluations)
        separate_offspring_male = offspring_evaluations[:half_pop_size]
        separate_offspring_female = offspring_evaluations[half_pop_size:]
        # sorted_separate_offspring_male = sorted(separate_offspring_male, key=lambda x: x[1])
        # sorted_separate_offspring_female = sorted(separate_offspring_female, key=lambda x: x[1])
        # Replace the worst solutions with the best new ones
        male_new_sols_replace = compare_and_replace(male_evaluations_new_pos, separate_offspring_male)
        female_new_sols_replace = compare_and_replace(female_evaluations_new_pos, separate_offspring_female)

        # Extracting new male and female solutions
        extrac_new_male_sols = [sum(sol[2], []) for sol in male_new_sols_replace]
        extrac_new_female_sols = [sum(sol[2], []) for sol in female_new_sols_replace]


        # Update current solutions with new ones if they are better
        for sol in range(half_pop_size):
            if female_new_sols_replace[sol][1] <= female_cur_sols_value[sol]:
                female_cur_sols_value[sol] = female_new_sols_replace[sol][1]
                female_cur_sols[sol] = extrac_new_female_sols[sol]
            female_velocity_dict[sol] = copy.deepcopy(female_velocity_check_incon[sol])
            if male_new_sols_replace[sol][1] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = male_new_sols_replace[sol][1]
                male_cur_sols[sol] = extrac_new_male_sols[sol]
            male_velocity_dict[sol] = copy.deepcopy(male_velocity_check_incon[sol])
            male_value_history.append(male_new_sols_replace[sol][1])
            female_value_history.append(female_new_sols_replace[sol][1])
        gbest_per_gen.append(gbest_value)
        progress_percent = (gen + 1) / num_gen * 100
        print(f'Progress: {progress_percent:.2f}%')
    print("----" * 50)
    print(f"Name of Input File: {name_path_input}")
    print(f"Number of Generations: {num_gen}")
    print(f"Population Size: {pop_size}")
    print(f"Final Best Solution (Tardiness): {gbest_value}")
    print(f"Best Male Solution: {min(male_value_history)}")
    print(f"Best Female Solution: {min(female_value_history)}")
    print("----" * 50)
    return gbest_per_gen,female_value_history,male_value_history


start_time = time.time()
num_gen = 50
pop_size = 20
a1 = 0
a2 = 0
a3 = 0.5
beta = 0
gmax = 1
gmin = 0
vmax = 1
vmin = 0
alpha = 0.5
delta = 0
nuptial_dance = 0
random_flight = 0
mutation_strength = 0
gbest_per_gen = []
male_value_history = []
female_value_history = []
name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
gbest_per_gen,male_value_history,female_value_history= mayfly(name_path_input, num_gen, pop_size, a1, a2,a3, beta, gmax,gmin,vmax,vmin,delta,nuptial_dance,random_flight,alpha,mutation_strength )

# End the timer
end_time = time.time()
time_taken = end_time - start_time

# Convert time_taken to hours, minutes, and seconds
hours = int(time_taken // 3600)
minutes = int((time_taken % 3600) // 60)
seconds = time_taken % 60
gbest_value = min(gbest_per_gen)
gbest_per_gen = gbest_per_gen[:num_gen]
male_value_history = male_value_history[:num_gen]
female_value_history = female_value_history[:num_gen]

# Display final results
hours, remainder = divmod(time_taken, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
print(f"Time Taken (second) : {time_taken:.2f}")
print("----" * 50)


# Plotting the graph with proper limits
plt.figure(figsize=(10, 5))
plt.plot(gbest_per_gen, label='Best Global Solution (gbest)', marker='_')
# plt.plot(male_value_history, label='Male Solution Movement', linestyle='--')
# plt.plot(female_value_history, label='Female Solution Movement', linestyle='-.')
plt.xlabel(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
plt.ylabel('Tardiness')
plt.title(f'(Mayfly Algorithm - {name_path_input}) - {pop_size} Population Size - {num_gen} Generations - Total tardiness: {gbest_value}')

# Set the x-axis limit from 0 to num_gen
plt.xlim(0, num_gen)

# Add the legend
plt.legend()

# Create an inset in the plot for parameter descriptions
param_descriptions = (
    "Parameters:\n"
    f"a1 = {a1}\n"
    f"a2 = {a2}\n"
    f"a3 = {a3}\n"
    f"beta = {beta}\n"
    f"gmax = {gmax}\n"
    f"gmin = {gmin}\n"
    f"vmax = {vmax}\n"
    f"vmin = {vmin}\n"
    f"delta = {delta}"
)
# Position the text box in figure coords, and set the box style
text_box = plt.text(0.16, 0.04, param_descriptions, transform=plt.gcf().transFigure, fontsize=11,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

# Show the plot with the parameter descriptions
plt.show()

