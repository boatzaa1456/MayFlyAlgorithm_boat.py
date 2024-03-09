import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols import *
import itertools
import pandas as pd
import concurrent.futures
import copy
import math
value_heavy = 40
# seed_all = 3124
# random.seed(3124)

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
        evaluations.append(evaluate_all_sols(individual, df_item_pool,heavy_item_set, name_path_input))
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


def cut_arc_sol(arc_sol):
    num_item = len(arc_sol) + 1
    arc_sol_list = [[] for _ in range(num_item)]

    for arc in arc_sol:
        arc_sol_list[arc[0]].append(arc)
        arc_sol_list[arc[1]].append(arc)

    arc_sol_cut = [arc_sol_list[item] for item in range(num_item)]

    return arc_sol_cut


import random


def init_velocity_sol(arc_sol_cut):
    # Use list comprehension to create a list of dictionaries, where each dictionary
    # contains arcs as keys and random velocities as values
    arc_sol_velocity_dict = [
        {arc: round(random.random(), 4) for arc in item} for item in arc_sol_cut
    ]

    return arc_sol_velocity_dict


def coef_times_velocity(coef, arc_sol_velocity_dict):
    # Use list comprehension to iterate over each item's velocity dictionary
    # Use dictionary comprehension to apply coefficient to each velocity
    # and enforce a maximum value of 1
    return [
        {arc: min(1, coef * velocity) for arc, velocity in item_dict.items()}
        for item_dict in arc_sol_velocity_dict
    ]



def gaussian_coefficient(a, B, r):
    return a * math.exp(-B * r ** 2)


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
    # import random
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

def nuptial_dance(solution):
    """Perform a random swap in the solution to mimic nuptial dance."""
    i, j = random.sample(range(len(solution)), 2)
    solution[i], solution[j] = solution[j], solution[i]
    return solution

def random_flight(solution, subset_size):
    """Randomly shuffle a subset of the solution to mimic random flight."""
    if subset_size > len(solution):
        subset_size = len(solution)
    indices = random.sample(range(len(solution)), subset_size)
    subset = [solution[i] for i in indices]
    random.shuffle(subset)
    for index, value in zip(indices, subset):
        solution[index] = value
    return solution

def replace_with_better_offspring(parents, offspring):
    """
    แทนที่พ่อหรือแม่ด้วยลูกที่ดีกว่าในกลุ่มเพศเดียวกัน

    :param parents: ลิสต์ของพ่อหรือแม่ (ตามเพศ) แต่ละคนเป็น tuple ที่มีค่า tardiness ที่ตำแหน่งที่ 1
    :param offspring: ลิสต์ของลูก ตามเพศเดียวกันกับ parents, แต่ละคนเป็น tuple ที่มีค่า tardiness ที่ตำแหน่งที่ 1
    :return: ลิสต์ใหม่ของประชากรที่อาจรวมถึงลูกที่ดีกว่าแทนที่พ่อหรือแม่
    """
    # สร้างลิสต์ใหม่จาก parents เพื่อไม่เปลี่ยนแปลงข้อมูลเดิม
    new_population = parents.copy()

    # ลูปเพื่อเปรียบเทียบแต่ละลูกกับพ่อหรือแม่
    for i, parent in enumerate(parents):
        # หากมีลูกน้อยกว่าพ่อหรือแม่, อาจไม่ต้องเปรียบเทียบทุกคู่
        if i < len(offspring):
            # เปรียบเทียบค่า tardiness
            if offspring[i][1] < parent[1]:
                # แทนที่พ่อหรือแม่ด้วยลูกที่ดีกว่า
                new_population[i] = offspring[i]

    return new_population


def mayfly(name_path_input, num_gen, pop_size, *parameters):
    global sol
    seed_all = 3124
    random.seed(seed_all)
    a1, a2,a3,g, beta,sub_size,alpha,mutation_rate  = parameters
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    half_pop_size = pop_size // 2
    E_all = list(itertools.permutations(range(num_item), 2))
    sub_E_list = [[arc for arc in E_all if arc[0] == item or arc[1] == item] for item in range(num_item)]
    # สร้าง set ของ item ที่ถือว่าเป็น item หนัก
    num_item = len(df_item_pool)
    heavy_item_set = set(df_item_pool[df_item_pool['weight'] >= value_heavy].index)
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
            evaluation = evaluate_all_sols(mayfly, df_item_pool,heavy_item_set,name_path_input)
            evaluations.append(evaluation)
            cur_sols.append(sum(evaluation[2], []))
            cur_sols_value.append(evaluation[1])
            cur_arc_sols.append(sol_from_list_to_arc(cur_sols[-1]))
            arc_sols_cut.append(cut_arc_sol(cur_arc_sols[-1]))
            velocity_dict.append(init_velocity_sol(arc_sols_cut[-1]))

        return evaluations, cur_sols, cur_sols_value, cur_arc_sols, arc_sols_cut, velocity_dict

    # Initialize male and female populations
    male_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]
    female_mayfly_population = [[i for i in range(num_item)] for _ in range(half_pop_size)]

    # Parallel processing for male and female populations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        male_future = executor.submit(process_mayfly_population, male_mayfly_population)
        male_results = male_future.result()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        female_future = executor.submit(process_mayfly_population, female_mayfly_population)
        female_results = female_future.result()

    gbest_per_gen = []
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
                # ตรวจสอบก่อนว่าจำเป็นต้องสร้างสำเนาหรือไม่
                if male_pbest_sol[sol] != current_sol:
                    male_pbest_sol[sol] = current_sol  # อาจไม่จำเป็นต้องใช้ deepcopy ถ้าไม่มีการแก้ไขข้อมูล
                if male_pbest_arc_sols[sol] != current_arc_sol:
                    male_pbest_arc_sols[sol] = current_arc_sol
                if male_pbest_arc_sols_cut[sol] != current_arc_sol_cut:
                    male_pbest_arc_sols_cut[sol] = current_arc_sol_cut

                # Update global best if current personal best is better
                if current_value <= gbest_value:
                    gbest_value = current_value
                    if gbest_sol != current_sol:
                        gbest_sol = current_sol  # อาจไม่จำเป็นต้องใช้ deepcopy
                    if gbest_arc_sol_cut != current_arc_sol_cut:
                        gbest_arc_sol_cut = current_arc_sol_cut
                    nuptial_dance(male_cur_sols[sol])
                    male_cur_arc_sols[sol] = sol_from_list_to_arc(male_cur_sols[sol])  # อัพเดทโดยไม่ใช้ deepcopy
                    male_arc_sols_cut[sol] = cut_arc_sol(male_cur_arc_sols[sol])
                    male_velocity_dict[sol] = init_velocity_sol(male_arc_sols_cut[sol])  # อัพเดทโดยไม่ใช้ deepcopy

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
        male_new_value = []


        for sol in range(half_pop_size):
            mcoef_velocity = coef_times_velocity(g,male_velocity_dict[sol])
            mpbest_diff = coef_times_position_with_gaussian(male_pbest_arc_sols_cut[sol],male_arc_sols_cut[sol],a1,beta)
            mgbest_diff =coef_times_position_with_gaussian(gbest_arc_sol_cut, male_arc_sols_cut[sol],a2, beta)
            madded_pbest_gbest = add_velocity(mgbest_diff, mpbest_diff)
            madded_velocity = add_velocity(mcoef_velocity, madded_pbest_gbest)
            mvelocity_check_incon = check_velocity_inconsistency(madded_velocity)
            mcut_set = creat_cut_set(mvelocity_check_incon, alpha )
            mnew_pos = sol_position_update(mcut_set, male_arc_sols_cut[sol], sub_E_list, male_cur_sols[sol][0], male_pbest_sol[sol][0],gbest_sol[0])[0]
            mevaluation = evaluate_all_sols(mnew_pos, df_item_pool,heavy_item_set ,name_path_input)

            # Append the results to their respective lists
            male_coef_velocity.append(mcoef_velocity)
            male_pbest_diff.append(mpbest_diff)
            male_gbest_diff.append(mgbest_diff)
            male_added_pbest_gbest.append(madded_pbest_gbest)
            male_added_velocity.append(madded_velocity)
            male_velocity_check_incon.append(mvelocity_check_incon)
            male_cut_set.append(mcut_set)
            male_new_pos.append(mnew_pos)
            male_evaluations_new_pos.append(mevaluation)

        # ---------------------- Female Mayfly Section ----------------------
        for sol in range(half_pop_size):
            # Update personal best if the current solution is better
            if female_cur_sols_value[sol] <= female_pbest_value[sol]:
                female_pbest_value[sol] = female_cur_sols_value[sol]
                # ตรวจสอบก่อนว่าจำเป็นต้องสร้างสำเนาหรือไม่
                if female_pbest_sol[sol] != female_cur_sols[sol]:
                    female_pbest_sol[sol] = female_cur_sols[sol]  # หลีกเลี่ยง deepcopy
                if female_pbest_arc_sols[sol] != sol_from_list_to_arc(female_cur_sols[sol]):
                    female_pbest_arc_sols[sol] = sol_from_list_to_arc(female_cur_sols[sol])  # สร้างใหม่โดยไม่ใช้ deepcopy
                if female_pbest_arc_sols_cut[sol] != cut_arc_sol(female_pbest_arc_sols[sol]):
                    female_pbest_arc_sols_cut[sol] = cut_arc_sol(female_pbest_arc_sols[sol])  # สร้างใหม่โดยไม่ใช้ deepcopy

                # Update global best if the current personal best is better
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    if gbest_sol != female_cur_sols[sol]:
                        gbest_sol = female_cur_sols[sol]  # หลีกเลี่ยง deepcopy
                    if gbest_arc_sol_cut != cut_arc_sol(female_pbest_arc_sols[sol]):
                        gbest_arc_sol_cut = cut_arc_sol(female_pbest_arc_sols[sol])  # สร้างใหม่โดยไม่ใช้ deepcopy
                    random_flight(female_cur_sols[sol], sub_size)
                    female_cur_arc_sols[sol] = sol_from_list_to_arc(female_cur_sols[sol])  # สร้างใหม่โดยไม่ใช้ deepcopy
                    female_arc_sols_cut[sol] = cut_arc_sol(female_cur_arc_sols[sol])  # สร้างใหม่โดยไม่ใช้ deepcopy
                    female_velocity_dict[sol] = init_velocity_sol(female_arc_sols_cut[sol])  # สร้างใหม่โดยไม่ใช้ deepcopy

        # Initialize lists before the loop
        female_coef_velocity = []
        female_male_diff = []
        female_added_velocity = []
        female_velocity_check_incon = []
        female_cut_set = []
        female_new_pos = []
        female_evaluations_new_pos = []

        for sol in range(half_pop_size):
            fcoef_velocity = coef_times_velocity(g, female_velocity_dict[sol])
            fmale_diff = coef_times_position_with_gaussian(male_arc_sols_cut[sol], female_arc_sols_cut[sol],a3,beta)
            fadded_velocity = add_velocity(fcoef_velocity, fmale_diff)
            fvelocity_check_incon = check_velocity_inconsistency(fadded_velocity)
            fcut_set = creat_cut_set(fvelocity_check_incon, alpha )
            fnew_pos = sol_position_update(fcut_set, female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0]
            fevaluation = evaluate_all_sols(fnew_pos, df_item_pool,heavy_item_set,name_path_input)

            female_coef_velocity.append(fcoef_velocity)
            female_male_diff.append(fmale_diff)
            female_added_velocity.append(fadded_velocity)
            female_velocity_check_incon.append(fvelocity_check_incon)
            female_cut_set.append(fcut_set)
            female_new_pos.append(fnew_pos)
            female_evaluations_new_pos.append(fevaluation)

        sorted_male_evaluations = sorted(male_evaluations_new_pos, key=lambda x: x[1])
        sorted_female_evaluations = sorted(female_evaluations_new_pos, key=lambda x: x[1])

        # Use list comprehensions for processing evaluations
        male_new_cur_sols = [sum(evaluation[2], []) for evaluation in sorted_male_evaluations]
        female_new_cur_sols = [sum(evaluation[2], []) for evaluation in sorted_female_evaluations]

        # Offspring generation and evaluation
        offspring_1 = []
        offspring_2 = []
        for sol in range(half_pop_size):
            offspring_pair = cxPartialyMatched(copy.deepcopy(male_new_cur_sols[sol]),copy.deepcopy(female_new_cur_sols[sol]))
            offspring_1.append(mutShuffleIndexes(offspring_pair[0],mutation_rate))
            offspring_2.append(mutShuffleIndexes(offspring_pair[1],mutation_rate))
        # Combining and evaluating offspring in one loop
        offspring_combine = offspring_1 + offspring_2
        offspring_evaluations = [evaluate_all_sols(offspring, df_item_pool,heavy_item_set,name_path_input) for offspring in offspring_combine]

        # Separate offspring to male and female randomly
        random.shuffle(offspring_evaluations)
        separate_offspring_male = offspring_evaluations[:half_pop_size]
        separate_offspring_female = offspring_evaluations[half_pop_size:]

        # sort offspring evaluations
        sorted_offs_male = sorted(separate_offspring_male, key=lambda x: x[1])
        sorted_offs_female = sorted(separate_offspring_female, key=lambda x: x[1])

        # Replace the worst solutions with the best new ones
        male_new_sols_replace = replace_with_better_offspring(sorted_male_evaluations, sorted_offs_male)
        female_new_sols_replace = replace_with_better_offspring(sorted_female_evaluations, sorted_offs_female)

        # Extracting new male and female solutions
        extrac_new_male_sols = [sum(sol[2], []) for sol in male_new_sols_replace]
        extrac_new_female_sols = [sum(sol[2], []) for sol in female_new_sols_replace]


        # Update current solutions with new ones if they are better
        for sol in range(half_pop_size):
            if female_new_sols_replace[sol][1] <= female_cur_sols_value[sol]:
                female_cur_sols_value[sol] = female_new_sols_replace[sol][1]
                female_cur_sols[sol] = extrac_new_female_sols[sol]  # ไม่จำเป็นต้องใช้ deepcopy ถ้าไม่มีการแก้ไขข้อมูลต่อ
            if male_new_sols_replace[sol][1] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = male_new_sols_replace[sol][1]
                male_cur_sols[sol] = extrac_new_male_sols[sol]  # ไม่จำเป็นต้องใช้ deepcopy ถ้าไม่มีการแก้ไขข้อมูลต่อ

            # ตรวจสอบว่าจำเป็นต้องสร้างสำเนาใหม่หรือไม่
            if male_velocity_dict[sol] != male_velocity_check_incon[sol]:
                male_velocity_dict[sol] = male_velocity_check_incon[sol]  # อาจหลีกเลี่ยง deepcopy หากไม่มีการแก้ไขข้อมูล
            if female_velocity_dict[sol] != female_velocity_check_incon[sol]:
                female_velocity_dict[sol] = female_velocity_check_incon[sol]  # อาจหลีกเลี่ยง deepcopy หากไม่มีการแก้ไขข้อมูล

        gbest_per_gen.append(gbest_value)
        progress_percent = (gen + 1) / num_gen * 100
        seed_all += 0
        print(f'Progress: {progress_percent:.2f}%')
    print(gbest_per_gen)
    print("----" * 50)
    print(f"Name of Input File: {name_path_input}")
    print(f"Number of Generations: {num_gen}")
    print(f"Population Size: {pop_size}")
    print(f"Final Best Solution (Tardiness): {gbest_value}")
    print("----" * 50)
    return gbest_per_gen

start_time = time.time()
num_gen = 100
pop_size = 50
a1 = 1
a2 = 2
a3 = 2
beta = 0.6
g = 0.5
alpha = 0.7
sub_size = 10
mutation_rate = 0.3
gbest_per_gen = []
male_value_history = []
female_value_history = []
name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
gbest_per_gen = mayfly(name_path_input, num_gen, pop_size, a1, a2,a3,g, beta,sub_size,alpha,mutation_rate )

# End the timer
end_time = time.time()
time_taken = end_time - start_time

# Convert time_taken to hours, minutes, and seconds
hours = int(time_taken // 3600)
minutes = int((time_taken % 3600) // 60)
seconds = time_taken % 60
gbest_value = min(gbest_per_gen)
gbest_per_gen = gbest_per_gen[:num_gen]

# Display final results
hours, remainder = divmod(time_taken, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
print(f"Time Taken (second) : {time_taken:.2f}")
print("----" * 50)


# Plotting the graph with proper limits
plt.figure(figsize=(10, 5))
plt.plot(gbest_per_gen, label='Best Global Solution (gbest)', marker='_')
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
    f"g= {g}\n"
    f"alpha = {alpha}\n"
    f"sub_size = {sub_size}\n"
    f"mutation_rate = {mutation_rate}"
)
# Position the text box in figure coords, and set the box style
text_box = plt.text(0.16, 0.04, param_descriptions, transform=plt.gcf().transFigure, fontsize=11,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

# Show the plot with the parameter descriptions
plt.show()

