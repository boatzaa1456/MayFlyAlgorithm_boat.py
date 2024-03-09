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
random.seed(3124)
np.random.seed(3124)


def mayfly(name_path_input, num_gen, pop_size, *parameters):
    a1, a2,a3, beta, gmax,gmin,vmax,vmin,delta,nuptial_dance,random_flight,alpha,mutation_rate = parameters
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
                    nuptial_dance_fraction = exponential_decay(nuptial_dance, gen,delta)
                    partial_shuffle(current_sol, nuptial_dance_fraction)
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
        male_new_value = []


        for sol in range(half_pop_size):
            mcoef_velocity = coef_times_velocity_mayfly(gmax,gmin,gen,num_gen,male_velocity_dict[sol])
            mpbest_diff = coef_times_position_with_gaussian(male_pbest_arc_sols_cut[sol],male_arc_sols_cut[sol],a1,beta)
            mgbest_diff =coef_times_position_with_gaussian(gbest_arc_sol_cut, male_arc_sols_cut[sol],a2, beta)
            madded_pbest_gbest = add_velocity(mgbest_diff, mpbest_diff)
            madded_velocity = add_velocity(mcoef_velocity, madded_pbest_gbest)
            mvelocity_check_incon = check_velocity_inconsistency(madded_velocity)
            mcut_set = creat_cut_set(mvelocity_check_incon, alpha )
            mnew_pos = sol_position_update(mcut_set, male_arc_sols_cut[sol], sub_E_list, male_cur_sols[sol][0], male_pbest_sol[sol][0],gbest_sol[0])[0]
            mevaluation = evaluate_all_sols(mnew_pos, df_item_pool, name_path_input)

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
                female_pbest_sol[sol] = copy.deepcopy(female_cur_sols[sol])
                female_pbest_arc_sols[sol] = copy.deepcopy(sol_from_list_to_arc(female_pbest_sol[sol]))
                female_pbest_arc_sols_cut[sol] = copy.deepcopy(cut_arc_sol(female_pbest_arc_sols[sol]))

                # Update global best if the current personal best is better
                if female_pbest_value[sol] <= gbest_value:
                    gbest_value = female_pbest_value[sol]
                    gbest_sol = copy.deepcopy(female_pbest_sol[sol])
                    gbest_arc_sol_cut = copy.deepcopy(female_pbest_arc_sols_cut[sol])
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
            fcoef_velocity = coef_times_velocity_mayfly(gmax, gmin, gen, num_gen, female_velocity_dict[sol])
            fmale_diff = coef_times_position_with_gaussian(male_arc_sols_cut[sol], female_arc_sols_cut[sol],a3,beta)
            fadded_velocity = add_velocity(fcoef_velocity, fmale_diff)
            fvelocity_check_incon = check_velocity_inconsistency(fadded_velocity)
            fcut_set = creat_cut_set(fvelocity_check_incon, alpha )
            fnew_pos = sol_position_update(fcut_set, female_arc_sols_cut[sol], sub_E_list, female_cur_sols[sol][0],female_pbest_sol[sol][0], gbest_sol[0])[0]
            fevaluation = evaluate_all_sols(fnew_pos, df_item_pool, name_path_input)

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
pop_size = 40
a1 = 1
a2 = 1.5
a3 = 1.5
beta = 0.008
gmax = 1
gmin = 0
vmax = 1
vmin = 0
alpha = 0.5
delta = 0.5
nuptial_dance = 0.5
random_flight = 0.5
mutation_rate = 0.2
gbest_per_gen = []
male_value_history = []
female_value_history = []
name_path_input = '1R-20I-150C-2P'
df_item_pool = read_input(name_path_input)
gbest_per_gen,male_value_history,female_value_history= mayfly(name_path_input, num_gen, pop_size, a1, a2,a3, beta, gmax,gmin,vmax,vmin,delta,nuptial_dance,random_flight,alpha,mutation_rate )

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
# # plt.plot(female_value_history, label='Female Solution Movement', linestyle='-.')
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
    # f"beta = {beta}\n"
    # f"gmax = {gmax}\n"
    # f"gmin = {gmin}\n"
    # f"vmax = {vmax}\n"
    # f"vmin = {vmin}\n"
    # f"delta = {delta}"
)
# Position the text box in figure coords, and set the box style
text_box = plt.text(0.16, 0.04, param_descriptions, transform=plt.gcf().transFigure, fontsize=11,
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

# Show the plot with the parameter descriptions
plt.show()

