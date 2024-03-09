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
# random.seed(1234)

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


def coef_times_position_with_gaussian(arc_first, arc_second,c_value,):
    import random

    # หาความแตกต่างของตำแหน่ง
    num_item = len(arc_first)
    total_diff = sum([len(set(arc_first[i]) - set(arc_second[i])) for i in range(num_item)])
    total_possible_diff = sum([len(set(arc_first[i])) for i in range(num_item)])

    # คำนวณค่าสัมประสิทธิ์ตามความต่างด้วยช่วงที่ละเอียดขึ้น
    coef_times_position_dict = []
    for item in range(num_item):
        item_dict = {}
        diff_item = len(set(arc_first[item]) - set(arc_second[item]))
        if diff_item > 0:
            difference_ratio = total_diff / total_possible_diff
            if difference_ratio <= 0.25:
                coef = random.uniform(0, 0.25)
            elif difference_ratio <= 0.5:
                coef = random.uniform(0.26, 0.5)
            elif difference_ratio <= 0.75:
                coef = random.uniform(0.51, 0.75)
            else:
                coef = 1
        else:
            coef = 1  # ถ้าไม่มีความต่าง, ให้ค่าสัมประสิทธิ์เป็น 1

        # จำกัดค่า coef ไม่ให้เกิน 1 หลังจากคูณด้วย c_value
        coef = min(c_value * coef, 1)

        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                item_dict[arc] = round(coef, 4)
        coef_times_position_dict.append(item_dict)

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

# Define the simplified nuptial dance function
def nuptial_dance(solution, attractor):
    i = random.randint(0, len(solution) - 1)
    if solution[i] in attractor:
        j = attractor.index(solution[i])
        if i != j and solution[j] != attractor[j]:
            solution[i], solution[j] = solution[j], solution[i]
    return solution

# Define the simplified random flight function
def random_flight(solution):
    i, j = random.sample(range(len(solution)), 2)
    solution[i], solution[j] = solution[j], solution[i]
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
    a1, a2,a3,g,alpha,seed = parameters
    random.seed(seed)
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
            male_current_value = male_cur_sols_value[sol]
            male_current_sol = male_cur_sols[sol]
            male_current_arc_sol = sol_from_list_to_arc(male_current_sol)
            male_current_arc_sol_cut = cut_arc_sol(male_current_arc_sol)

            # Update personal best if current solution is better or equal
            if male_current_value <= male_pbest_value[sol]:
                male_pbest_value[sol] = male_current_value
                male_pbest_sol[sol] = copy.deepcopy(male_current_sol)
                male_pbest_arc_sols[sol] = copy.deepcopy(male_current_arc_sol)
                male_pbest_arc_sols_cut[sol] = copy.deepcopy(male_current_arc_sol_cut)

                # Update global best if current personal best is better
                if male_pbest_value[sol] <= gbest_value:
                    gbest_value = male_pbest_value[sol]
                    gbest_sol = male_pbest_sol[sol]
                    gbest_arc_sol_cut = male_pbest_arc_sols_cut[sol]

            nuptial_dance(male_cur_sols[sol], gbest_sol)
            male_cur_arc_sols[sol] = sol_from_list_to_arc(copy.deepcopy(male_cur_sols[sol]))
            male_arc_sols_cut[sol] = cut_arc_sol(copy.deepcopy(male_cur_arc_sols[sol]))
            male_velocity_dict[sol] = init_velocity_sol(copy.deepcopy(male_arc_sols_cut[sol]))

            mcoef_velocity = coef_times_velocity(g,male_velocity_dict[sol])
            mpbest_diff = coef_times_position_with_gaussian(male_pbest_arc_sols_cut[sol],male_arc_sols_cut[sol],a1)
            mgbest_diff =coef_times_position_with_gaussian(gbest_arc_sol_cut, male_arc_sols_cut[sol],a2)
            madded_pbest_gbest = add_velocity(mgbest_diff, mpbest_diff)
            madded_velocity = add_velocity(mcoef_velocity, madded_pbest_gbest)
            mvelocity_check_incon = check_velocity_inconsistency(madded_velocity)
            mcut_set = creat_cut_set(mvelocity_check_incon, alpha )
            mnew_pos = sol_position_update(mcut_set, male_arc_sols_cut[sol], sub_E_list, male_cur_sols[sol][0],male_pbest_sol[sol][0], gbest_sol[0])[0]
            mevaluation = evaluate_all_sols(mnew_pos, df_item_pool,heavy_item_set ,name_path_input)


        # ---------------------- Female Mayfly Section ----------------------

            female_current_value = female_cur_sols_value[sol]
            female_current_sol = female_cur_sols[sol]
            female_current_arc_sol = sol_from_list_to_arc(female_current_sol)
            female_current_arc_sol_cut = cut_arc_sol(female_current_arc_sol)

            # Update personal best if current solution is better or equal
            if female_current_value <= female_pbest_value[sol]:
                female_pbest_value[sol] = female_current_value
                female_pbest_sol[sol] = copy.deepcopy(female_current_sol)
                female_pbest_arc_sols[sol] = copy.deepcopy(female_current_arc_sol)
                female_pbest_arc_sols_cut[sol] = copy.deepcopy(female_current_arc_sol_cut)

                # Update global best if current personal best is better
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    gbest_sol = female_pbest_sol[sol]
                    gbest_arc_sol_cut = female_pbest_arc_sols_cut[sol]

            if female_cur_sols_value[sol] == gbest_value:
                random_flight(copy.deepcopy(female_cur_sols[sol]))
                female_cur_arc_sols[sol] = sol_from_list_to_arc(copy.deepcopy(female_cur_sols[sol]))
                female_arc_sols_cut[sol] = cut_arc_sol(copy.deepcopy(female_cur_arc_sols[sol]))
                female_velocity_dict[sol] = init_velocity_sol(copy.deepcopy(female_arc_sols_cut[sol]))

            fcoef_velocity = coef_times_velocity(g, female_velocity_dict[sol])
            fmale_diff = coef_times_position_with_gaussian(male_arc_sols_cut[sol], female_arc_sols_cut[sol],a3)
            fadded_velocity = add_velocity(fcoef_velocity, fmale_diff)
            fvelocity_check_incon = check_velocity_inconsistency(fadded_velocity)
            fcut_set = creat_cut_set(fvelocity_check_incon, alpha )
            fnew_pos = sol_position_update(fcut_set, female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0]
            fevaluation = evaluate_all_sols(fnew_pos, df_item_pool,heavy_item_set,name_path_input)


            # Extracting new male and female solutions
            extrac_new_male_sols = sum(mevaluation[2],[])
            extrac_new_female_sols = sum(fevaluation[2],[])


        # Update current solutions with new ones if they are better
            if fevaluation[1] <= female_cur_sols_value[sol]:
                female_cur_sols_value[sol] = fevaluation[1]
                female_cur_sols[sol] = extrac_new_female_sols
                female_velocity_dict[sol] = fvelocity_check_incon
            if mevaluation[1] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = mevaluation[1]
                male_cur_sols[sol] = extrac_new_male_sols
                male_velocity_dict[sol] = mvelocity_check_incon
        gbest_per_gen.append(gbest_value)
        progress_percent = (gen + 1) / num_gen * 100

        print(f'Progress: {progress_percent:.2f}%')
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
a1 = 0.3
a2 = 0.5
a3 = 0.7
g = 0.9
alpha = 0.5
random_seed = 3124
# sub_size = 20
# mutation_rate = 0.3
gbest_per_gen = []
male_value_history = []
female_value_history = []
name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
gbest_per_gen = mayfly(name_path_input, num_gen, pop_size, a1, a2,a3,g,alpha,random_seed)

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
    f"g= {g}\n"
    f"alpha = {alpha}\n"
    # f"sub_size = {sub_size}\n"
    # f"mutation_rate = {mutation_rate}"
)
# Position the text box in figure coords, and set the box style
text_box = plt.text(0.16, 0.04, param_descriptions, transform=plt.gcf().transFigure, fontsize=11,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

# Show the plot with the parameter descriptions
plt.show()


