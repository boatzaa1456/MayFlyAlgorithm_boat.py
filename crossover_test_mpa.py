import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols_old import *
import itertools
import pandas as pd
random.seed(3124)
def read_input(name_path_input):
    df_item_sas_random = pd.read_csv('df_item_sas_random.csv')

    name_path_input = name_path_input

    df_duedate = pd.read_csv(name_path_input + '\\duedate_' + name_path_input + '.csv', header=None)

    df_item_oder = pd.read_csv(name_path_input + '\\input_location_item_' + name_path_input + '.csv', header=None)

    list_duedate = df_duedate[0].values
    list_duedate = df_duedate[0].tolist()

    num_order = df_item_oder.shape[1]

    list_order = []
    list_total_item = []

    df_item_pool = pd.DataFrame()

    for order in range(num_order):
        item = df_item_oder[order][df_item_oder[order] != 0]

        df_item_in_this_order = df_item_sas_random[df_item_sas_random['location'].isin(item)].copy()
        df_item_in_this_order['duedate'] = list_duedate[order]
        df_item_pool = pd.concat([df_item_pool, df_item_in_this_order])

        num_item_this_order = df_item_in_this_order.shape[0]
        for i in range(num_item_this_order):
            list_order.append(order)

        list_total_item.extend(item.tolist())

    df_item_pool['order'] = list_order

    df_item_pool.reset_index(drop=True, inplace=True)

    return df_item_pool, df_item_sas_random

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

def init_velocity_sol(arc_sol_cut) :
    import random
    num_item = len(arc_sol_cut)
    arc_sol_velocity_dict = [{} for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut[item]:
            arc_sol_velocity_dict[item][arc] = round(random.random(), 4)

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

import random

def cxPartialyMatchedLevyBrownianAdaptive(ind1, ind2, gbest):
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Mapping indexes to values for fast lookup
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    # Choose two random crossover points for PMX crossover and sort them
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

    # Apply PMX crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched values
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Update the position of the values in the mapping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    # Decide whether to explore or exploit based on similarity to gbest
    # Here we can use a more sophisticated measure of similarity if needed
    should_explore = (ind1 != gbest) and (ind2 != gbest)

    # Apply mutation if the individuals are similar to gbest
    if not should_explore:
        mutate(ind1, size)
        mutate(ind2, size)

    # Apply Levy flight or Brownian motion based on exploration need
    for ind in [ind1, ind2]:
        if should_explore:
            if random.random() < 0.25:  # Reduced probability for exploration
                i, j = sorted(random.sample(range(len(ind)), 2))
                ind[i:j] = reversed(ind[i:j])
        else:
            for _ in range(size // 20):  # Reduced swap times for exploitation
                i = random.randint(0, size - 2)
                ind[i], ind[i + 1] = ind[i + 1], ind[i]

    return ind1, ind2

def mutate(ind, size):
    """Applies a mutation to the individual by swapping two random elements."""
    i, j = random.sample(range(size), 2)
    ind[i], ind[j] = ind[j], ind[i]

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



def mayfly(name_path_input, num_gen, pop_size, *parameters):
    import copy
    # input data
    a1, a2, beta, gravity = parameters
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    half_pop_size = int(pop_size) // 2
    E_all = list(itertools.permutations(range(num_item), 2))
    sub_E_list = [[arc for arc in E_all if arc[0] == item or arc[1] == item] for item in range(num_item)]

    #-------------------------------------------------------------------------------------------------------------#
    # Initailze gbest
    gbest_value = [100000]
    gbest_sol = []
    gbest_arc_sol_cut = []
    gbest_per_gen = []

    # ---------------------- ส่วนของ Mayfly ตัวผู้ ---------------------- #
    # Initailze male pbest

    male_pbest_value = [100000 for _ in range(half_pop_size)]
    male_pbest_sol = [[] for sol in range(half_pop_size)]
    male_pbest_arc_sols = [[] for sol in range(half_pop_size)]
    male_pbest_arc_sols_cut = [ [] for sol in range(half_pop_size)]


    #  create initial male population
    male_mayfly_population = [[i for i in range(num_item)] for j in range(half_pop_size)]
    male_evaluations = []
    male_cur_sols = []
    male_cur_sols_value = []
    male_cur_arc_sols = []
    male_arc_sols_cut = []
    male_velocity_dict = []

    for male in male_mayfly_population:
        random.shuffle(male)
        male_evaluations.append(evaluate_all_sols(male, df_item_pool, name_path_input))
    for evaluation in male_evaluations:
        male_cur_sols.append(sum(evaluation[2], []))
        male_cur_sols_value.append(evaluation[1])
    for male in range(half_pop_size):
        male_cur_arc_sols.append(sol_from_list_to_arc(male_cur_sols[male]))
        male_arc_sols_cut.append(cut_arc_sol(male_cur_arc_sols[male]))
        male_velocity_dict.append(init_velocity_sol(male_arc_sols_cut[male]))

    # # ---------------------- ส่วนของ Mayfly ตัวเมีย ---------------------- #
    # Initailze male pbest
    female_pbest_value = [100000 for _ in range(half_pop_size)]
    female_pbest_sol = [[] for sol in range(half_pop_size)]
    female_pbest_arc_sols = [[] for sol in range(half_pop_size)]
    female_pbest_arc_sols_cut = [ [] for sol in range(half_pop_size)]

    #  create initial male population
    female_mayfly_population = [[i for i in range(num_item)] for j in range(half_pop_size)]
    female_evaluations = []
    female_cur_sols = []
    female_cur_sols_value = []
    female_cur_arc_sols = []
    female_arc_sols_cut = []
    female_velocity_dict = []

    for female in female_mayfly_population:
        random.shuffle(female)
        female_evaluations.append(evaluate_all_sols(female, df_item_pool, name_path_input))
    for evaluation in female_evaluations:
        female_cur_sols.append(sum(evaluation[2], []))
        female_cur_sols_value.append(evaluation[1])
    for female in range(half_pop_size):
        female_cur_arc_sols.append(sol_from_list_to_arc(female_cur_sols[female]))
        female_arc_sols_cut.append(cut_arc_sol(female_cur_arc_sols[female]))
        female_velocity_dict.append(init_velocity_sol(female_arc_sols_cut[female]))

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
                    gbest_per_gen.append(gbest_value)
                    gbest_sol = copy.deepcopy(current_sol)
                    gbest_arc_sol_cut = copy.deepcopy(current_arc_sol_cut)

        male_coef_velocity = []
        male_pbest_diff = []
        male_gbest_diff = []
        male_added_pbest_gbest = []
        male_added_velocity = []
        male_velocity_check_incon = []
        male_cut_set = []
        male_new_pos = []
        male_evaluations_new_pos = []
        male_new_cur_sols = []
        male_new_cur_sols_value = []
        test_position_minus_position = []
        for sol in range(half_pop_size):
            test_position_minus_position.append(position_minus_position(gbest_arc_sol_cut,male_arc_sols_cut[sol]))
        for sol in range(half_pop_size):
            male_coef_velocity.append(coef_times_velocity(gravity_calculation(0.9,0.4,gen,num_gen), male_velocity_dict[sol]))
            male_pbest_diff.append(coef_times_position(a1,position_minus_position(male_pbest_arc_sols_cut[sol], male_arc_sols_cut[sol])))
            male_gbest_diff.append(coef_times_position(a2, position_minus_position(gbest_arc_sol_cut, male_arc_sols_cut[sol])))
            male_added_pbest_gbest.append(add_velocity(male_gbest_diff[sol], male_pbest_diff[sol]))
            male_added_velocity.append(add_velocity(male_coef_velocity[sol], male_added_pbest_gbest[sol]))
            male_velocity_check_incon.append(check_velocity_inconsistency(male_added_velocity[sol]))
            male_cut_set.append(creat_cut_set(male_velocity_check_incon[sol], 0.05))
            male_new_pos.append(sol_position_update(male_cut_set[sol], male_arc_sols_cut[sol], sub_E_list,male_cur_sols[sol][0],male_pbest_sol[sol][0], gbest_sol[0] )[0])
            male_evaluations_new_pos.append(evaluate_all_sols(male_new_pos[sol], df_item_pool, name_path_input))
        for evaluation in male_evaluations_new_pos:
            male_new_cur_sols.append(sum(evaluation[2], []))
            male_new_cur_sols_value.append(evaluation[1])
        # Update male mayfly new solutions
        for sol in range(half_pop_size):
            if male_new_cur_sols_value[sol] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = male_new_cur_sols_value[sol]
                male_cur_sols[sol] = copy.deepcopy(male_new_cur_sols[sol])
        male_velocity_dict[sol] = copy.deepcopy(male_velocity_check_incon[sol])

    # ---------------------- ส่วนของ Mayfly ตัวเมีย ----------------------
        for sol in range(half_pop_size):
            if female_cur_sols_value[sol] <= female_pbest_value[sol]:
                female_pbest_value[sol] = female_cur_sols_value[sol]
                female_pbest_sol[sol] = copy.deepcopy(female_cur_sols[sol])
                female_pbest_arc_sols[sol] = copy.deepcopy(sol_from_list_to_arc(female_pbest_sol[sol]))
                female_pbest_arc_sols_cut[sol] = copy.deepcopy(cut_arc_sol(female_pbest_arc_sols[sol]))
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    gbest_per_gen.append(gbest_value)
                    gbest_sol = copy.deepcopy(female_pbest_sol[sol])
                    gbest_arc_sol_cut = copy.deepcopy(female_pbest_arc_sols_cut[sol])
                    if female_cur_sols_value[sol] <= male_cur_sols_value[sol]:
                        random.shuffle(female_cur_sols[sol])
                        female_cur_arc_sols[sol] = copy.deepcopy(sol_from_list_to_arc(female_cur_sols[sol]))
                        female_arc_sols_cut[sol] = copy.deepcopy(cut_arc_sol(female_cur_arc_sols[sol]))
                        female_velocity_dict[sol] = copy.deepcopy(init_velocity_sol(female_arc_sols_cut[sol]))


        female_coef_velocity = []
        female_male_diff = []
        female_added_velocity = []
        female_velocity_check_incon = []
        female_cut_set = []
        female_new_pos = []
        female_evaluations_new_pos = []
        female_new_cur_sols = []
        female_new_cur_sols_value = []

        for sol in range(half_pop_size):
            female_coef_velocity.append(coef_times_velocity(gravity_calculation(0.9,0.4,gen,num_gen), female_velocity_dict[sol]))
            female_male_diff.append(coef_times_position(a1, position_minus_position(male_arc_sols_cut[sol], female_arc_sols_cut[sol])))
            female_added_velocity.append(add_velocity(female_coef_velocity[sol], female_male_diff[sol]))
            female_velocity_check_incon.append(check_velocity_inconsistency(female_added_velocity[sol]))
            female_cut_set.append(creat_cut_set(female_velocity_check_incon[sol], 0.05))
            female_new_pos.append(sol_position_update(female_cut_set[sol], female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0])
            female_evaluations_new_pos.append(evaluate_all_sols(female_new_pos[sol], df_item_pool, name_path_input))
        for evaluation in female_evaluations_new_pos:
            female_new_cur_sols.append(sum(evaluation[2], []))
            female_new_cur_sols_value.append(evaluation[1])

        offspring_1 = []
        offspring_2 = []
        for sol in range(half_pop_size):
            offspring_1.append(cxPartialyMatchedLevyBrownianAdaptive(copy.deepcopy(male_new_cur_sols[sol]), copy.deepcopy(female_new_cur_sols[sol]),gbest_sol)[0])
            offspring_2.append(cxPartialyMatchedLevyBrownianAdaptive(copy.deepcopy(male_new_cur_sols[sol]), copy.deepcopy(female_new_cur_sols[sol]),gbest_sol)[1])
        offspring_combine = offspring_1 + offspring_2
        offspring_evaluations = []
        for offspring in offspring_combine:
            offspring_evaluations.append(evaluate_all_sols(offspring, df_item_pool, name_path_input))

        # evaluate the offspring
        offspring_evaluations = []
        offspring = offspring_1 + offspring_2
        for off in offspring:
            offspring_evaluations.append(evaluate_all_sols(off, df_item_pool, name_path_input))

        # Separate offspring to male and female randomly
        random.shuffle(offspring_evaluations)
        Separate_off_pop = len(offspring_evaluations) // 2
        separate_offspring_male = offspring_evaluations[:Separate_off_pop]
        separate_offspring_female = offspring_evaluations[Separate_off_pop:]

        extrac_new_male_sols = []
        extrac_new_female_sols = []
        male_new_sols_replace = copy.deepcopy(male_evaluations_new_pos)
        female_new_sols_replace = copy.deepcopy(female_evaluations_new_pos)
        # Replace the worst solutions with the best new ones
        male_new_sols_replace = compare_and_replace(male_new_sols_replace, separate_offspring_male)
        female_new_sols_replace = compare_and_replace(female_new_sols_replace, separate_offspring_female)
        for sol in male_new_sols_replace:
            extrac_new_male_sols.append(sum(sol[2],[]))
        for sol in female_new_sols_replace:
            extrac_new_female_sols.append(sum(sol[2],[]))


        for sol in range(half_pop_size):
            if female_new_sols_replace[sol][1] <= female_cur_sols_value[sol]:
                female_cur_sols_value[sol] = female_new_sols_replace[sol][1]
                female_cur_sols[sol] = copy.deepcopy(extrac_new_female_sols[sol])
            if male_new_sols_replace[sol][1] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = male_new_sols_replace[sol][1]
                male_cur_sols[sol] = copy.deepcopy(extrac_new_male_sols[sol])

        progress_percent = (gen + 1) / num_gen * 100
        print(f'Progress: {progress_percent:.2f}%')
        # print(f'male velocity {gen} : {male_velocity_dict[0]}')
        # print(f'female velocity {gen} : {female_velocity_dict[0]}')
        # print(f'male velocity check incon {gen} : {male_velocity_check_incon[0]}')
        # print(f'female velocity check incon {gen} : {female_velocity_check_incon[0]}')
    print(f'num_gen : {num_gen}')
    print(f'pop_size : {pop_size}')
    print(f'total tardiness : {gbest_value}')
    print(f'gbest solution : {gbest_sol}')
    print(f'male last generation : {male_pbest_value}')
    print(f'female last generation : {female_pbest_value} ')

    x_axis = list(range(len(gbest_per_gen)))
    plt.plot(x_axis, gbest_per_gen)
    plt.xlabel('solution in gbest')
    plt.ylabel('Fitness')
    plt.title(f'Num_gen = {num_gen}, pop_size = {pop_size}, total tardiness = {gbest_value}')
    plt.show()


start_time = time.time()

name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
mayfly(name_path_input, 100, 50, 0.5, 10, 0.2, 0.1)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")















