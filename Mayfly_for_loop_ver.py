import random
import time
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols import *
import itertools
import pandas as pd
from Mayfly_all_function import *
random.seed(3124)
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
        male_cur_sol = sum(evaluation[2], [])
        male_cur_sols.append(male_cur_sol)
        male_cur_sols_value.append(evaluation[1])
    for male in range(half_pop_size):
        male_cur_arc_sols.append(sol_from_list_to_arc(male_cur_sols[male]))
        male_arc_sols_cut.append(cut_arc_sol(male_cur_arc_sols[male]))
        male_velocity_dict.append(init_velocity_sol(male_arc_sols_cut[male]))

    # # ---------------------- ส่วนของ Mayfly ตัวเมีย ---------------------- #
    # Initailze male pbest
    female_pbest_value = [100000 for _ in range(half_pop_size)]
    female_pbest_sol = [[] for sol in range(half_pop_size)]
    female_pbest_arc_sols_cut = [[] for sol in range(half_pop_size)]

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
        female_cur_sol = sum(evaluation[2], [])
        female_cur_sols.append(female_cur_sol)
        female_cur_sols_value.append(evaluation[1])
    for female in range(half_pop_size):
        female_cur_arc_sols.append(sol_from_list_to_arc(female_cur_sols[female]))
        female_arc_sols_cut.append(cut_arc_sol(female_cur_arc_sols[female]))
        female_velocity_dict.append(init_velocity_sol(female_arc_sols_cut[female]))

    for gen in range(num_gen):
    #---------------------- ส่วนของ Mayfly ตัวผู้ ----------------------
        for sol in range(half_pop_size):
            if male_cur_sols_value[sol] <= male_pbest_value[sol]:
                male_pbest_value[sol] = male_cur_sols_value[sol]
                male_pbest_sol[sol] = male_cur_sols[sol][:]
                male_pbest_arc_sols_cut[sol] = copy.deepcopy(male_arc_sols_cut[sol])
                if male_pbest_value[sol] <= gbest_value:
                    gbest_value = male_pbest_value[sol]
                    gbest_per_gen.append(gbest_value)
                    gbest_sol = male_pbest_sol[sol]
                    gbest_arc_sol_cut = copy.deepcopy(male_pbest_arc_sols_cut[sol])

        male_cut_set = []
        for sol in range(half_pop_size):
            # ทำการคำนวณแต่ละขั้นตอนและรวมผลลัพธ์ในขั้นตอนเดียว
            male_coef_velocity = coef_times_velocity(gravity, male_velocity_dict[sol])
            male_pbest_diff = coef_times_position(a1,position_minus_position(male_pbest_arc_sols_cut[sol], male_arc_sols_cut[sol]))
            male_gbest_diff = coef_times_position(a2, position_minus_position(gbest_arc_sol_cut, male_arc_sols_cut[sol]))
            male_added_pbest_gbest = add_velocity(male_gbest_diff, male_pbest_diff)
            male_added_velocity = add_velocity(male_coef_velocity, male_added_pbest_gbest)
            male_velocity_check_incon = check_velocity_inconsistency(male_added_velocity)
            male_cut_set.append(creat_cut_set(male_velocity_check_incon, 0.05))

        male_new_pos = []
        male_evaluations_new_pos = []


        for sol in range(half_pop_size):
            male_new_pos.append(sol_position_update(male_cut_set[sol], male_arc_sols_cut[sol], sub_E_list,male_cur_sols[sol][0],male_pbest_sol[sol][0], gbest_sol[0] )[0])
            male_evaluations_new_pos.append(evaluate_all_sols(male_new_pos[sol], df_item_pool, name_path_input))

        # Rank the  male mayflies
        ranked_male_population = sorted([male_evaluations_new_pos[sol] for sol in range(half_pop_size)], key=lambda x: x[1])
        ranked_male_new_cur_sols = [sum(rank[2], []) for rank in ranked_male_population]

    # ---------------------- ส่วนของ Mayfly ตัวเมีย ---------------------- #
        for sol in range(half_pop_size):
            if female_cur_sols_value[sol] <= female_pbest_value[sol]:
                female_pbest_value[sol] = female_cur_sols_value[sol]
                female_pbest_sol[sol] = female_cur_sols[sol][:]
                female_pbest_arc_sols_cut[sol] = copy.deepcopy(female_arc_sols_cut[sol])
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    gbest_per_gen.append(gbest_value)
                    gbest_sol = female_pbest_sol[sol]
                    gbest_arc_sol_cut = copy.deepcopy(female_pbest_arc_sols_cut[sol])

        female_cut_set = []
        for sol in range(half_pop_size):
            # ทำการคำนวณแต่ละขั้นตอนและรวมผลลัพธ์ในขั้นตอนเดียว
            female_coef_velocity = coef_times_velocity(gravity, female_velocity_dict[sol])
            female_male_diff = coef_times_position(a2,position_minus_position(male_arc_sols_cut[sol], female_arc_sols_cut[sol]))
            female_added_velocity = add_velocity(female_coef_velocity, female_male_diff)
            female_velocity_check_incon = check_velocity_inconsistency(female_added_velocity)
            female_cut_set.append(creat_cut_set(female_velocity_check_incon, 0.05))

        female_new_pos = []
        female_evaluations_new_pos = []

        for sol in range(half_pop_size):
            female_new_pos.append(sol_position_update(female_cut_set[sol], female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0])
            female_evaluations_new_pos.append(evaluate_all_sols(female_new_pos[sol], df_item_pool, name_path_input))

        # Rank the  female mayflies
        ranked_female_population = sorted([female_evaluations_new_pos[sol] for sol in range(half_pop_size)], key=lambda x: x[1])
        ranked_female_new_cur_sols = [sum(rank[2], []) for rank in ranked_female_population]

        # Mate the mayflies
        offspring_1 = []
        offspring_2 = []
        for sol in range(half_pop_size):
            mate = cxPartialyMatched(ranked_male_new_cur_sols[sol],ranked_female_new_cur_sols[sol])
            offspring_1.append(mate[0])
            offspring_2.append(mate[1])

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

        # Replace the worst solutions with the best new ones
        male_replace_new_values = []
        male_replace_new_cur_sols = []
        female_replace_new_values = []
        female_replace_new_cur_sols = []
        male_new_sols_replace = compare_and_replace(ranked_male_population,separate_offspring_male)
        female_new_sols_replace = compare_and_replace(ranked_female_population,separate_offspring_female)
        for sol in range(half_pop_size):
            male_replace_new_values.append(male_new_sols_replace[sol][1])
            female_replace_new_values.append(female_new_sols_replace[sol][1])
        for evaluation in  male_new_sols_replace:
            male_extrac_new_cur_sols = sum(evaluation[2], [])
            male_replace_new_cur_sols.append(male_extrac_new_cur_sols)
        for evaluation in female_new_sols_replace:
            female_extrac_new_cur_sols = sum(evaluation[2], [])
            female_replace_new_cur_sols.append(female_extrac_new_cur_sols)

        # Update mayfly new solutions
        for sol in range(half_pop_size):
            if male_replace_new_values[sol] <= male_cur_sols_value[sol]:
                male_cur_sols_value[sol] = male_replace_new_values[sol]
                male_cur_sols[sol] = male_replace_new_cur_sols[sol]


        for sol in range(half_pop_size):
            if female_replace_new_values[sol] <= female_cur_sols_value[sol]:
                female_cur_sols_value[sol] = female_replace_new_values[sol]
                female_cur_sols[sol] = female_replace_new_cur_sols[sol]

        progress_percent = (gen + 1) / num_gen * 100
        print(f'Progress: {progress_percent:.2f}%')
    print(f'total tardiness : {gbest_value}')
    print(f'gbest solution : {gbest_sol}')
    print(f'male last generation : {male_pbest_value}')
    print(f'female last generation : {female_pbest_value} ')

    x_axis = list(range(len(gbest_per_gen)))
    plt.plot(x_axis, gbest_per_gen)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


start_time = time.time()

name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
mayfly(name_path_input, 1000, 10, 1, 1.5, 0.2, 0.1)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")















