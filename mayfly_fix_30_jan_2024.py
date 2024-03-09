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
    a1, a2,a3, beta, gmax,gmin,vmax,vmin,delta,nuptial_dance,random_flight,alpha,mutation_strength = parameters
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    half_pop_size = pop_size // 2
    E_all = list(itertools.permutations(range(num_item), 2))
    sub_E_list = [[arc for arc in E_all if arc[0] == item or arc[1] == item] for item in range(num_item)]
    gbest_per_gen = []
    male_value_history = []
    female_value_history = []
    # Initialize gbest and pbest values for males and females
    gbest_value, gbest_sol, gbest_arc_sol_cut = [100000], [], []
    male_pbest_value,female_pbest_value = [100000] * half_pop_size,[100000] * half_pop_size
    male_pbest_sol , female_pbest_sol = [[] for _ in range(half_pop_size)],[[] for _ in range(half_pop_size)]
    male_pbest_arc_sols , female_pbest_arc_sols = [[] for _ in range(half_pop_size)],[[] for _ in range(half_pop_size)]
    male_pbest_arc_sols_cut , female_pbest_arc_sols_cut = [[] for _ in range(half_pop_size)],[[] for _ in range(half_pop_size)]

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
        male_velocity_dict.append(init_velocity_sol(male_arc_sols_cut[male],vmax,vmin))

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
        female_velocity_dict.append(init_velocity_sol(female_arc_sols_cut[female],vmax, vmin))
    print("xxxx")









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
    # print(f"Best Male Solution: {min(male_value_history)}")
    # print(f"Best Female Solution: {min(female_value_history)}")
    print("----" * 50)
    return gbest_per_gen,female_value_history,male_value_history


start_time = time.time()
num_gen = 50
pop_size = 8
a1 = 0.3
a2 = 1
a3 = 1
beta = 0.02
gmax = 0.5
gmin = 0.1
vmax = 0.7
vmin = 0.3
alpha = 0.5
delta = 0.1
nuptial_dance = 0.3
random_flight = 0.1
mutation_strength = 0.5
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
plt.plot(male_value_history, label='Male Solution Movement', linestyle='--')
plt.plot(female_value_history, label='Female Solution Movement', linestyle='-.')
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

