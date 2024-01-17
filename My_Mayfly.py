import numpy as np
from evaluate_all_sols import *
from  Mayfly_all_function import *
np.random.seed(1234)

def mayfly(name_path_input, num_gen, pop_size, *parameters):
    # นำเข้าข้อมูล
    a1, a2, beta, gravity = parameters
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]

# ฟังก์ชันเป้าประสงค์ f(x) เพื่อลดความล่าช้ารวม
    # สร้างประชากร mayfly ตัวผู้และตัวเมีย
    half_pop_size = int(pop_size) // 2
    male_mayfly_population = np.tile(np.arange(num_item), (half_pop_size, 1))
    female_mayfly_population = np.tile(np.arange(num_item), (half_pop_size, 1))


    # ---------------------- ส่วนของ Mayfly ตัวผู้ ---------------------- #
    # ประเมินค่าเริ่มต้นสำหรับ mayfly ตัวผู้
    male_evaluations = []
    for male in male_mayfly_population:
        np.random.shuffle(male)  # สลับลำดับ
        male_evaluations.append(evaluate_all_sols(male, df_item_pool, name_path_input))


    # สร้าง pbest และความเร็วเริ่มต้นสำหรับ mayfly ตัวผู้
    male_mayfly_cur_sols = [sum(evaluation[2], []) for evaluation in male_evaluations]
    male_arc_sols = all_sol_from_list_to_arc(male_mayfly_cur_sols)
    male_arc_sols_cut = [cut_arc_sol(arc) for arc in male_arc_sols]
    male_velocity_dict = [init_velocity_sol(arc_sol_cut) for arc_sol_cut in male_arc_sols_cut]
    male_pbest_arc_sol_cut = male_arc_sols_cut.copy()


    # ---------------------- ส่วนของ Mayfly ตัวเมีย ---------------------- #
    # ประเมินค่าเริ่มต้นสำหรับ mayfly ตัวเมีย
    female_evaluations = []
    for female in female_mayfly_population:
        np.random.shuffle(female)  # Shuffle solution
        female_evaluations.append(evaluate_all_sols(female, df_item_pool, name_path_input))

    # สร้างตำแหน่งและความเร็วเริ่มต้นสำหรับ mayfly ตัวเมีย
    female_mayfly_cur_sols = [sum(evaluation[2], []) for evaluation in female_evaluations]
    female_arc_sols = all_sol_from_list_to_arc(female_mayfly_cur_sols)
    female_arc_sols_cut = [cut_arc_sol(arc) for arc in female_arc_sols]
    female_velocity_dict = [init_velocity_sol(arc_sol_cut) for arc_sol_cut in female_arc_sols_cut]


    # ---------------------- ส่วนของ Global Best (gbest) ---------------------- #
    # หาค่า global best เริ่มต้น
    combined_evaluations = male_evaluations + female_evaluations
    combined_tardiness = [eval_[1] for eval_ in combined_evaluations]
    gbest_total_tardiness = min(combined_tardiness)
    gbest_indices = [indices for indices, tardiness in enumerate(combined_tardiness) if tardiness == gbest_total_tardiness]
    gbest_solutions = [combined_evaluations[indices][2] for indices in gbest_indices]
    gbest_solutions = [sum(solution_pair, []) for solution_pair in gbest_solutions]
    gbest_init_arc_sols = all_sol_from_list_to_arc(gbest_solutions)
    gbest_init_arc_sols_replicated = [gbest_init_arc_sols[0] for _ in range(half_pop_size)]


    # ทำซ้ำ gbest arc solutions เพื่อเปรียบเทียบใน mayfly ตัวผู้
    gbest_init_arc_sols_replicated_cut = [cut_arc_sol(arc) for arc in gbest_init_arc_sols_replicated]

    # เข้าสู่ลูปหลักของอัลกอริทึม
    for _ in range(num_gen):
        # ---------------------- ส่วนการอัปเดตความเร็วและตำแหน่งของ Mayfly ตัวผู้ ---------------------- #
        male_coef_times_velocity = [coef_times_velocity(gravity, male_velocity_dict[sol]) for sol in range(half_pop_size)]
        male_coef_times_pbest_diff = [coef_times_position(a1, position_minus_position(male_arc_sols_cut[sol], male_arc_sols_cut[sol])) for sol in range(half_pop_size)]
        male_coef_times_gbest_diff = [coef_times_position(a2, position_minus_position(gbest_init_arc_sols_replicated_cut[sol], male_arc_sols_cut[sol]))for sol in range(half_pop_size)]
        male_new_velocity = [add_velocity(male_coef_times_velocity[sol],add_velocity(male_coef_times_pbest_diff[sol], male_coef_times_gbest_diff[sol]))for sol in range(half_pop_size)]
        male_velocity_check_incon = [check_velocity_inconsistency(male_new_velocity[sol] ) for sol in range(half_pop_size)]
        male_cut_set = [creat_cut_set(male_velocity_check_incon[sol],0.5) for sol in range(half_pop_size)]

        # ---------------------- ส่วนการอัปเดตความเร็วและตำแหน่งของ Mayfly ตัวเมีย ---------------------- #
        female_coef_times_velocity = [coef_times_velocity(gravity, female_velocity_dict[sol]) for sol in range(half_pop_size)]
        female_coef_times_male_diff = [coef_times_position(a2, position_minus_position(male_arc_sols_cut[sol], female_arc_sols_cut[sol])) for sol in range(half_pop_size)]
        female_new_velocity = [add_velocity(female_coef_times_velocity[sol], female_coef_times_male_diff[sol]) for sol in range(half_pop_size)]
        female_velocity_check_incon = [check_velocity_inconsistency(female_new_velocity[sol]) for sol in range(half_pop_size)]
        female_cut_set = [creat_cut_set(female_velocity_check_incon[sol],0.5) for sol in range(half_pop_size)]

        print(xxx)

        # Evaluate solutions

        # Rank the mayflies

        # Mate the mayflies

        # Evaluate offspring

        # Separate offspring to male and female randomly

        # Replace the worst solutions with the best new ones

        # Update pbest and gbest

    # Return the final best solution(s)

name_path_input = '1I-10-100-2'
df_item_pool = read_input(name_path_input)
mayfly(name_path_input, 1, 100, 1, 1, 0.5, 0.7)


















# def mayfly(name_path_input, num_gen, pop_size, *parameters):
#     import random
# # Objective function f(x) หาค่า tardiness ที่น้อยที่สุด : minimize the total tardiness
#     # input
#     a1, a2, beta, gravity = parameters
#     df_item_pool, df_item_sas_random = read_input(name_path_input)
#     num_item = df_item_pool.shape[0]
#
# # สร้างคำตอบเริ่มต้น โดยยังไม่แบ่งว่าเป็น ตัวผู้หรือตัวเมีย : initialize the mayfly poppulation without separate male and female mayfly
#     mayfly_cur_pos = [random.sample(range(num_item), num_item) for _ in range(pop_size)]
#     pbest_sols = [[] for _ in range(pop_size)]
#     pbest_tardiness = [float('inf')] * pop_size
#     g_best_tardiness = float('inf')
#     g_best_items_in_batch = []
#
# # ประเมินคำตอบเริ่มต้นและหา gbest : evaluate solutions
#     item_in_batch_all_sols = []
#     for i, sol in enumerate(mayfly_cur_pos):
#         _, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(sol, df_item_pool, name_path_input)
#
#         extended_item_in_batch_sol = []
#         for batch in item_in_batch_sol:
#             extended_item_in_batch_sol.extend(batch)
#         item_in_batch_all_sols.append(extended_item_in_batch_sol)
#
#         # อัปเดต pbest สำหรับ mayfly แต่ละตัว
#         pbest_sols[i] = extended_item_in_batch_sol
#         pbest_tardiness[i] = total_tardiness_sol
#
# # อัปเดต gbest : find global best solution (gbest)
#         if total_tardiness_sol < g_best_tardiness:
#             g_best_tardiness = total_tardiness_sol
#             g_best_items_in_batch = [extended_item_in_batch_sol]  # เริ่มต้นรายการใหม่ด้วยคำตอบที่ดีที่สุด
#         elif total_tardiness_sol == g_best_tardiness:
#             g_best_items_in_batch.append(extended_item_in_batch_sol)  # เพิ่มคำตอบปัจจุบันไปยังรายการของคำตอบที่ดีที่สุด
#
#
#     extended_item_in_batch_all_sols = item_in_batch_all_sols.copy()
#
# # แยก mayfly ตัวผู้และตัวเมียและความเร็วเริ่มต้น จากคำตอบที่ซ่อมแล้ว  : Separate the male and female mayfly population xi and yi with velocities vmi(male) and vfi(female) after evaluate
#     num_males = num_females = pop_size // 2
#     male_mayfly = []
#     female_mayfly = []
#
#     for _ in range(num_males):
#         male_mayfly.append(random.choice(extended_item_in_batch_all_sols))
#
#     for _ in range(num_females):
#         female_mayfly.append(random.choice(extended_item_in_batch_all_sols))
#
#     male_mayfly_arc = all_sol_from_list_to_arc(male_mayfly)
#     female_mayfly_arc = all_sol_from_list_to_arc(female_mayfly)
#
#     male_mayfly_arc_sol_cut = [cut_arc_sol(arc) for arc in male_mayfly_arc]
#     female_mayfly_arc_sol_cut = [cut_arc_sol(arc) for arc in female_mayfly_arc]
#
#     male_mayfly_velocity_dict = [init_velocity_sol(arc_sol_cut) for arc_sol_cut in male_mayfly_arc_sol_cut]
#     female_mayfly_velocity_dict = [init_velocity_sol(arc_sol_cut) for arc_sol_cut in female_mayfly_arc_sol_cut]
#
#         # Main optimization loop
#         # for _ in range(num_gen):
#             # pbest ของตัวผู้และตัวเมีย
#
#     male_mayfly_cur_pbest = []
#     male_mayfly_cur_gbest = []
#
#
#             # Update velocities for male and female mayflies
#             # coef*vtij
#     male_mayfly_gravity_times_velocity = [coef_times_velocity(gravity, sol) for sol in male_mayfly_velocity_dict]
#     female_mayfly_gravity_times_velocity = [coef_times_velocity(gravity, sol) for sol in female_mayfly_velocity_dict]
#
#             # update_velocities(male_mayflies, female_mayflies)
#             # update_positions(male_mayflies, female_mayflies)
#
#             # ประเมินคำตอบใหม่
#             # evaluate_solutions(male_mayflies, female_mayflies)
#
#             # จัดอันดับ mayflies
#             # rank_mayflies(male_mayflies, female_mayflies)
#
#             # ผสมพันธุ์ mayflies
#             # offspring = mate_mayflies(male_mayflies, female_mayflies)
#
#             # ประเมินลูกหลาน
#             # evaluate_offspring(offspring)
#
#             # แยกลูกหลานไปยังตัวผู้และตัวเมีย
#             # separate_offspring(offspring)
#
#             # แทนที่คำตอบที่แย่ที่สุดด้วยคำตอบใหม่ที่ดีที่สุด
#             # replace_worst_solutions(male_mayflies, female_mayflies, offspring)
#
#             # อัปเดต pbest และ gbest
#             # update_pbest_gbest(male_mayflies, female_mayflies)
#
#
#             # ประมวลผลหลังการทำงานและการแสดงผล
#         # if stopping_criteria_met():
#         #     break
