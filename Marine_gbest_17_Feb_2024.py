import random
import pandas as pd
from evaluate_all_sols import *
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
import timeit
import scipy
import csv
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import sys
from numpy.random import Generator, PCG64

# seed_num = 3124
# random.seed(seed_num)
# numpy_randomGen = Generator(PCG64(seed_num))
# truncnorm.random_state = numpy_randomGen

#ใช้ gbest สร้าง prey ใหม่
value_heavy = 40

#ฟังก์ชันอ่านไฟล์
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

#new* แปลงจาก list ให้กลายเป็น arc ต้องรู้ว่ามีทั้งหมดกี่ตัว
def sol_from_list_to_arc(sol):   #เอาคำตอบเข้าไปแค่ 1 ตัว
    num_item = len(sol)
    arc_sol = []
    for i in range(num_item-1):
        arc_sol.append((sol[i],sol[i+1]))
    return arc_sol

def all_sols_from_list_to_arc(all_sols):
    num_sol = len(all_sols)  #เก็บจำนวนคำตอบของ all_sol
    num_item = len(all_sols[0])
    all_arc_sols = [[(all_sols[i][j],all_sols[i][j+1]) for j in range(num_item-1)] for i in range(num_sol)] #เรียงลำดับในการวน loop j วนก่อน i
    return all_arc_sols

def cut_arc_sol(arc_sol): #argument เป็น arc_sol ของคำตอบเดียว
    num_item = len(arc_sol)+1
    arc_sol_cut = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol:
            if item == arc[0] or item == arc[1]:
                arc_sol_cut[item].append(arc)
    return arc_sol_cut

#สร้างความเร็วเริ่มต้น
def init_velocity_sol(arc_sol_cut):
    # import random
    num_item = len(arc_sol_cut) #นับจำนวน item จากจำนวนสมาชิกของ list arc_sol_cut
    arc_sol_velocity_dict = [{} for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut[item]:
            arc_sol_velocity_dict[item][arc] = round(random.random(),4)
    return arc_sol_velocity_dict

#สร้างความเร็ว Vt+1 = w*Vt + c1r1(Pbest-Xt) + c2r2(Gbest-Xt)
# Xt + 1 = Xt + Vt + 1  Solution update --> cut_set, Xt, sub_E_list
#นำสปส.ไปคูณหน้าค่าคงที่ของความเร็ว ออกมาเป็นความเร็ว (w*Vt)
#เติมความเร็ว
def coef_times_velocity(coef_velocity, arc_sol_velocity_dict):
    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)] #สร้างlistของdict [{}, {}, {},...]
    for item in range(num_item):
        for arc in arc_sol_velocity_dict[item].keys():
            coef = coef_velocity * random.random()   # p = 0.5
            if coef*arc_sol_velocity_dict[item][arc] > 1:
                coef_times_velocity_dict[item][arc] = 1
            else:
                coef_times_velocity_dict[item][arc] = round(coef*arc_sol_velocity_dict[item][arc],4)
    return coef_times_velocity_dict

#P*CF
def calculate_p_cf(iter, max_iter,p):
    cf = (1 - iter/max_iter) ** (2 * iter/max_iter )
    p_cf = p * cf
    return p_cf

def coef_times_velocity_cpf(coef_velocity, arc_sol_velocity_dict):
    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)] #สร้างlistของdict [{}, {}, {},...]
    for item in range(num_item):
        for arc in arc_sol_velocity_dict[item].keys():
            coef = coef_velocity
            if coef*arc_sol_velocity_dict[item][arc] > 1:
                coef_times_velocity_dict[item][arc] = 1
            else:
                coef_times_velocity_dict[item][arc] = round(coef*arc_sol_velocity_dict[item][arc],4)
    return coef_times_velocity_dict

#เอาตำแหน่ง-ตำแหน่ง
def position_minus_position(arc_first, arc_second):
    num_item = len(arc_first)
    pos_minus_pos = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                pos_minus_pos[item].append(arc)
    return pos_minus_pos

#เอาค่าคงที่คูณกับผลต่างของตำแหน่งที่ได้มา
def coef_times_position(c_value,arc_diff):
    # import random
    num_item = len(arc_diff)
    coef_times_position_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_diff[item]:
            coef = c_value*random.random()
            if coef > 1:
                coef = 1
            coef_times_position_dict[item][arc] = round(coef,4)
    return coef_times_position_dict

#สร้างการแจกแจงแบบ Brownian
def coef_times_position_marine_r_brownian(arc_diff):
    # import random
    # from numpy.random import Generator, PCG64
    # numpy_randomGen = Generator(PCG64(my_seed))
    # truncnorm.random_state = numpy_randomGen

    num_item = len(arc_diff)
    coef_times_position_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_diff[item]:
            coef = truncnorm.rvs(-2, 2, 0.5, 0.25)
            if coef > 1:
                coef = 1
            coef_times_position_dict[item][arc] = round(coef,4)
    return coef_times_position_dict

#สร้างการแจกแจงแบบ Levy
def levy_sample_1(alpha=0.5, beta=-1, size=1):
    # import numpy as np
    # np.random.seed(seed_num)
    cauchy_samples = np.random.standard_cauchy(size)
    levy_sample = alpha / (1 + beta * cauchy_samples ** 2)
    return abs(levy_sample[0])

def coef_times_position_marine_r_levy(arc_diff,bound=5): #ค่า bound  เป็นค่าตัวเลขสูงสุดที่จะปัดความน่าจะเป็นให้เป็น 1 ถ้าเกินจากค่านี้ ให้สุ่ม arc มาจาก Sub_E_list
    # import random
    num_item = len(arc_diff)
    coef_times_position_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_diff[item]:
            coef = levy_sample_1()
            if coef > 1 and coef <= bound:
                coef = 1
            coef_times_position_dict[item][arc] = round(coef,4)
    return coef_times_position_dict

#เอาความเร็วมาบวกกัน
def add_velocity(velocity_first, velocity_second):
    num_item = len(velocity_first)
    added_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in velocity_first[item].keys():
            added_velocity_dict[item][arc] = velocity_first[item][arc]
        for arc in velocity_second[item].keys():
            if arc in added_velocity_dict[item].keys():
                if velocity_second[item][arc] > added_velocity_dict[item][arc]:
                    added_velocity_dict[item][arc] = velocity_second[item][arc]
            else:
                added_velocity_dict[item][arc] = velocity_second[item][arc]
    return added_velocity_dict

#เช็ค arc ที่มีค่ามากมาใช้ คัดแต่ของดี
def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    #ใช้วิธีนี้แทน
    new_added_velocity_dict = [{arc:prob for arc,prob in added_velocity_dict[item].items()} for item in range(num_item)]
    for item in range(num_item): #วน item
        for arc_first in added_velocity_dict[item].keys():  #arc ตัวที่อยู่ใน added_velocity_dict[item].key() มีอะไรบ้าง
            if arc_first in added_velocity_dict[arc_first[0]].keys():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[0]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[0]][arc_first]
            if arc_first in added_velocity_dict[arc_first[1]].keys():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[1]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[1]][arc_first]
    return new_added_velocity_dict

#cut เอา arc ที่มีค่าความน่าจะเป็นมากกว่า 0.6
def create_cut_set(added_velocity_dict, alpha):
    num_item = len(added_velocity_dict)
    cut_set = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc] >= alpha:
                cut_set[item].append(arc)
    return cut_set

#cut set _r_levy
def create_cut_set_r_levy(added_velocity_dict, alpha):
    num_item = len(added_velocity_dict)
    cut_set = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc] >= alpha and added_velocity_dict[item][arc] <= 1:
                cut_set[item].append(arc)
            elif added_velocity_dict[item][arc] > 1:
                random_dest = random.choice([i for i in range(num_item) if i != item])
                cut_set[item].append((item, random_dest))
    return cut_set

#*sets = สามารถใส่ได้หลาย argument
#item ต้นทาง = source,  item ถัดไป = dest
#ถ้าใน picked_list มี item ที่ถูกเก็บเข้าไปแล้ว รอบต่อไปหากพบซ้ำอีกจะไม่เก็บ ถือว่าเก็บไปแล้ว
def select_dest_from_source(source, picked_list, *sets):
    #ฟังก์ชันนี้ทำหน้าที่ในการเลือก item ที่เราจะเดินเก็บถัดไป (dest) จากตำแหน่งปัจจุบันที่เราอยู่ (source) โดยเราต้องการได้ผลลัพธ์เป็น arc ของ (source,dest) และ picked_list เป็น list ที่เก็บ item ที่เราเดินเก็บไปแล้ว
    # import random
    for set in sets:
        new_set = []
        if len(set[source]) > 0:
            for arc in set[source]:
                if arc[1] not in picked_list and arc[0] == source:
                    new_set.append(arc)
        if len(new_set) > 0:
            dest = random.choice(new_set)[1]
            break
    arc_source_dest = (source,dest)
    return dest, arc_source_dest

#update solution เทียบคำตอบเก่าและใหม่ คัดเอาแต่ของดี
#เช็คตน.แรกของ start_previous_x (item เริ่มต้น) , เช็คตน.แรกของ start_gbest
def sol_position_update(cut_set, previous_x, sub_E_list, start_previous_x, start_gbest):
    # import random
    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []
    source = random.choice([start_previous_x, start_gbest, random.choice(range(num_item))])
    picked_list.append(source)

    for item_counter in range(num_item-1):
        dest, arc_source_dest = select_dest_from_source(source, picked_list, cut_set, previous_x, sub_E_list)
        source = dest
        picked_list.append(dest)
        picked_list_arc.append(arc_source_dest)
    return picked_list, picked_list_arc

#FADs
def fad_marine_single(prey_pos, fads_rate, fads):
    # สร้างค่าสุ่ม r เพื่อตัดสินใจว่าจะทำการ mutation หรือไม่
    r = random.random()
    if r < fads_rate:
        # สร้างค่าสุ่มอีกครั้งเพื่อตัดสินใจว่าจะทำการ shuffle ทั้งหมดหรือบางส่วน
        r2 = random.random()
        if r2 < fads:
            # Exploration: shuffle ทั้งหมด
            random.shuffle(prey_pos)
        else:
            # Exploitation: shuffle บางส่วน, เลือก 2 ตำแหน่งแบบสุ่มและสลับค่า
            idx1, idx2 = random.sample(range(len(prey_pos)), 2)
            prey_pos[idx1], prey_pos[idx2] = prey_pos[idx2], prey_pos[idx1]
    # ไม่มีการ mutation ถ้า r >= mut_rate
    return prey_pos

# ในฟังก์ชัน measure_initial_iterations, ปรับเปลี่ยนการเรียกใช้ mayfly:
def measure_initial_iterations(iterations_to_measure):
    times = []
    for _ in range(iterations_to_measure):
        start_time = time.time()
        marine_predator(name_path_input, ng, ps, mu, sigma, p[0], alpha_cut_set[0], alpha_levy[0], bound[0])
        end_time = time.time()
        times.append(end_time - start_time)
    return times

def calculate_average_time(times):
    return sum(times) / len(times)


def print_estimated_completion_time(average_time, total_iterations):
    total_estimated_time = average_time * total_iterations
    finish_time = datetime.now() + timedelta(seconds=total_estimated_time)
    print(f"Based on initial measurements, the estimated completion time is: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if total_estimated_time < 3600:
        print(f"Estimated to take around {total_estimated_time / 60:.2f} minutes.")
    elif total_estimated_time < 86400:
        print(f"Estimated to take around {total_estimated_time / 3600:.2f} hours.")
    else:
        print(f"Estimated to take around {total_estimated_time / 86400:.2f} days.")

#-------------------------สร้าง marine--------------------------*
#อ่านไฟล์
name_path_input = '1R-100I-250C-2P'
df_item_pool, df_item_sas_random = read_input(name_path_input)

running_time_list = []
start_time = timeit.default_timer()

def marine_predator(name_path_input, num_gen, pop_size,*parameters):
    from itertools import permutations
    mu, sigma, p, alpha_cut_set, alpha_levy, bound = parameters
    # mu, sigma, p, alpha_cut_set, alpha_levy, bound, fads_rate, fads = parameters
    df_item_pool, df_item_sas_random = read_input(name_path_input)

    #สร้างคำตอบเริ่มต้น สร้างตำแหน่งของ prey เริ่มต้น
    num_item = df_item_pool.shape[0]
    E_all = list(permutations(range(num_item),2))
    sub_E_list = [[arc for arc in E_all if arc[0] == item or arc[1] == item] for item in range(num_item)]
    # สร้าง set ของ item ที่ถือว่าเป็น item หนัก
    num_item = len(df_item_pool)
    heavy_item_set = set(df_item_pool[df_item_pool['weight'] >= value_heavy].index)
    prey_cur_pos = []

    # สร้างคำตอบของ pray เริ่มต้น
    for sol in range(pop_size):
        temp_pos = list(range(num_item))       #เรียงเลขตั้งแต่ 0 ถึง 9
        random.shuffle(temp_pos)               #สลับที่กัน
        prey_cur_pos.append(temp_pos)          #เอาเข้าไปเก็บใน prey_cur_pos
    picker_assignment_all_sols = []
    total_tardiness_all_sols = []
    item_in_batch_all_sols = []
    #หาค่าคำตอบของ prey แต่ละตัว ใน population
    for sol in range(pop_size):
        picker_assignment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(prey_cur_pos[sol],df_item_pool,heavy_item_set, name_path_input)
        picker_assignment_all_sols.append(picker_assignment_sol)
        total_tardiness_all_sols.append(total_tardiness_sol)
        item_in_batch_all_sols.append(item_in_batch_sol)

    #สร้าง list ของคำตอบใหม่ หลังจากซ่อมคำตอบแล้ว
    prey_cur_pos = [[] for sol in range(pop_size)]
    for sol in range(pop_size):
        for batch in item_in_batch_all_sols[sol]:
            prey_cur_pos[sol].extend(batch)
    g_best_index = total_tardiness_all_sols.index(min(total_tardiness_all_sols))
    g_best_sol = prey_cur_pos[g_best_index]
    g_best_tardiness = min(total_tardiness_all_sols)

    #สร้าง arc
    prey_arc_cur_pos = all_sols_from_list_to_arc(prey_cur_pos)
    #cut arc ที่เกี่ยวข้องกับเลข 0,1,2,3,4,5,6,7,8,9
    prey_cut_arc_cur_pos = [cut_arc_sol(prey_arc_cur_pos[sol]) for sol in range(pop_size)]
    #สร้าง tardiness อยู่กับ arc พร้อมบอกลำดับ index
    prey_tardiness_arc_cur_pos_index = [[total_tardiness_all_sols[sol], prey_arc_cur_pos[sol],sol] for sol in range(pop_size)]
    #เรียงลำดับค่า tardiness น้อยไปมาก
    ranked_prey_tardiness_arc_cur_pos_index = sorted(prey_tardiness_arc_cur_pos_index)
    #หาค่า g_best หาตัวที่มีค่าต่ำสุด
    g_best_arc_pos = ranked_prey_tardiness_arc_cur_pos_index[0][1][:]
    #สร้าง elite หน้าตาเหมือนกันจาก g_best
    elite_arc_cur_pos = [ranked_prey_tardiness_arc_cur_pos_index[0][1][:] for sol in range(pop_size)]
    elite_cut_arc_cur_pos = [cut_arc_sol(elite_arc_cur_pos[sol]) for sol in range(pop_size)]
    #สร้างความเร็วเริ่มต้น
    prey_cur_velo = [init_velocity_sol(prey_cut_arc_cur_pos[sol]) for sol in range(pop_size)]

#ใช้ g_best สร้าง prey
    # สร้าง elite หน้าตาเหมือนกันจาก g_best
    prey_gbest_arc_cur_pos = [g_best_arc_pos[:] for sol in range(pop_size)]
    prey_gbest_cut_arc_cur_pos = [cut_arc_sol(prey_gbest_arc_cur_pos[sol]) for sol in range(pop_size)]
    prey_gbest_tardiness_arc_cur_pos_index = [[total_tardiness_all_sols[sol], prey_gbest_arc_cur_pos[sol], sol] for sol in range(pop_size)]
    ranked_prey_gbest_tardiness_arc_cur_pos_index = sorted(prey_gbest_tardiness_arc_cur_pos_index)
    gbest_arc_pos_for_gen = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:]
    elite_gbest_arc_cur_pos = [ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:] for sol in range(pop_size)]
    elite_gbest_cut_arc_cur_pos = [cut_arc_sol(elite_gbest_arc_cur_pos[sol]) for sol in range(pop_size)]

    gbest_arc_pos_value = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1]
    gbest_arc_pos_each_gen = [gbest_arc_pos_value]
    tardiness_gbest_value = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][0]
    gbest_each_gen = [tardiness_gbest_value]
    gbest_arc_pos = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1]
    for iter in range(num_gen):
        #Phase 1 : While Iter < 1/3 Max_Iter  (Brownian)
        #กำหนดวนรอบ iter , num_gen = max_iter
        # iter = 4
        if iter < int(1 / 3 * num_gen):
        # while iter < int(1 / 3 * num_gen):
            elite_cut_arc_new = []
            for sol in range(pop_size):
                #สร้างความเร็วเริ่มต้น step_size_cur_prey
                step_size_cur_prey = init_velocity_sol(prey_gbest_cut_arc_cur_pos[sol])
                #สร้างความเร็วเริ่มต้น step_size_r_brownian (r_brownian*(elite-prey))
                step_size_r_brownian = coef_times_position_marine_r_brownian(position_minus_position(elite_gbest_cut_arc_cur_pos[sol],prey_gbest_cut_arc_cur_pos[sol]))
                #เอา step_size_cur_prey + step_size_r_brownian
                new_step_size = add_velocity(step_size_cur_prey, step_size_r_brownian)
                #ไม่คูณ P และ R randomm
                #check_velocity_inconsistency
                check_velocity_step_size = check_velocity_inconsistency(new_step_size)
                #cut_set
                step_size_cut_set = create_cut_set(check_velocity_step_size, alpha_cut_set)
                #update_prey ใหม่ (prey + step_size_cut_set)
                new_prey_cur_pos_phase_1, new_prey_arc_cur_pos_phase_1 = sol_position_update(step_size_cut_set, prey_gbest_cut_arc_cur_pos[sol], sub_E_list, prey_cur_pos[sol][0],elite_arc_cur_pos[sol][0][0])
                # FADs
                # new_prey_cur_pos_phase_1 = fad_marine_single(new_prey_cur_pos_phase_1, fads_rate, fads)
                #preyค่าใหม่ แทน preyค่าเก่า
                prey_cur_pos[sol] = new_prey_cur_pos_phase_1
                prey_arc_cur_pos[sol] = new_prey_arc_cur_pos_phase_1
                #หาค่าคำตอบของ prey แต่ละตัว ใน population หา picker, tardiness, batch
                picker_assignment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(prey_cur_pos[sol],df_item_pool,heavy_item_set,name_path_input)
                picker_assignment_all_sols[sol] = picker_assignment_sol[:]
                total_tardiness_all_sols[sol] = total_tardiness_sol
                item_in_batch_all_sols[sol] = item_in_batch_sol[:]
                # สร้าง tardiness อยู่กับ arc พร้อมบอกลำดับ index
                prey_tardiness_arc_cur_pos_index[sol] = [total_tardiness_sol, new_prey_arc_cur_pos_phase_1, sol]
            #เรียงลำดับค่า tardiness น้อยไปมาก
            ranked_prey_gbest_tardiness_arc_cur_pos_index = sorted(prey_tardiness_arc_cur_pos_index)
            #หาค่า g_best หาตัวที่มีค่าต่ำสุด
            gbest_arc_pos_for_gen = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:]
            #สร้าง elite หน้าตาเหมือนกันจาก g_best และ cut set
            elite_gbest_arc_cur_pos = [ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:] for sol in range(pop_size)]
            elite_gbest_cut_arc_cur_pos = [cut_arc_sol(elite_gbest_arc_cur_pos[sol]) for sol in range(pop_size)]
            elite_gbest_tardiness_arc_cur_pos_index = ranked_prey_gbest_tardiness_arc_cur_pos_index[0]

        # Phase 2: 1/3 Max_Iter <= Iter < 2/3 Max_Iter  (Brownian,Levy)
        elif int(1 / 3 * num_gen) <= iter < int(2 / 3 * num_gen):
        # while int(1 / 3 * num_gen) <= iter < int(2 / 3 * num_gen):
            half_pop_size = int(pop_size/2)
            # ครึ่ง r_brownian
            for sol in range(half_pop_size):
                #สร้างความเร็วเริ่มต้น step_size_cur_prey
                step_size_cur_prey = init_velocity_sol(prey_gbest_cut_arc_cur_pos[sol])
                #สร้างความเร็ว step_size_r_brownian (r_brownian*(elite-prey))
                step_size_r_brownian = coef_times_position_marine_r_brownian(position_minus_position(elite_gbest_cut_arc_cur_pos[sol],prey_gbest_cut_arc_cur_pos[sol]))
                #เอาความเร็วเริ่มต้นบวกความเร็ว Brownian  (step_size_cur_prey + step_size_r_brownian)
                new_step_size_r_brownian = add_velocity(step_size_cur_prey, step_size_r_brownian)
                #new_step_size * P.CF
                new_step_size_velocity_brownian = coef_times_velocity_cpf(calculate_p_cf(iter,num_gen,p),new_step_size_r_brownian)
                # check_velocity_inconsistency
                check_velocity_step_size_brownian = check_velocity_inconsistency(new_step_size_velocity_brownian)
                # cut_set
                step_size_cut_set_brownian = create_cut_set(check_velocity_step_size_brownian, alpha_cut_set)
                # update_prey ใหม่ (elite + step_size_cut_set)
                new_prey_cur_pos_phase_2_brownian, new_prey_arc_cur_pos_phase_2_brownian = sol_position_update(step_size_cut_set_brownian, elite_gbest_cut_arc_cur_pos[sol], sub_E_list, prey_cur_pos[sol][0], elite_arc_cur_pos[sol][0][0])
                # FADs
                # new_prey_cur_pos_phase_2_brownian = fad_marine_single(new_prey_cur_pos_phase_2_brownian, fads_rate, fads)
                # preyค่าใหม่ แทน preyค่าเก่า
                prey_cur_pos[sol] = new_prey_cur_pos_phase_2_brownian
                prey_arc_cur_pos[sol] = new_prey_arc_cur_pos_phase_2_brownian
                #หาค่าคำตอบของ prey แต่ละตัว ใน population หา picker, tardiness, batch
                picker_assignment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(prey_cur_pos[sol],df_item_pool,heavy_item_set,name_path_input)
                picker_assignment_all_sols[sol] = picker_assignment_sol[:]
                total_tardiness_all_sols[sol] = total_tardiness_sol
                item_in_batch_all_sols[sol] = item_in_batch_sol[:]
                # สร้าง tardiness อยู่กับ arc พร้อมบอกลำดับ index
                prey_tardiness_arc_cur_pos_index[sol] = [total_tardiness_sol, new_prey_arc_cur_pos_phase_2_brownian, sol]
                # เรียงลำดับค่า tardiness น้อยไปมาก
            ranked_prey_gbest_tardiness_arc_cur_pos_index = sorted(prey_tardiness_arc_cur_pos_index)
            # หาค่า g_best หาตัวที่มีค่าต่ำสุด
            gbest_arc_pos_for_gen = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:]
            # สร้าง elite หน้าตาเหมือนกันจาก g_best และ cut set
            elite_gbest_arc_cur_pos = [ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:] for sol in range(pop_size)]
            elite_gbest_cut_arc_cur_pos = [cut_arc_sol(elite_gbest_arc_cur_pos[sol]) for sol in range(pop_size)]
            elite_gbest_tardiness_arc_cur_pos_index = ranked_prey_gbest_tardiness_arc_cur_pos_index[0]

            # ครึ่ง r_levy
            for sol in range(half_pop_size,pop_size):
                # สร้างความเร็วเริ่มต้น step_size_cur_prey
                step_size_cur_prey = init_velocity_sol(prey_gbest_cut_arc_cur_pos[sol])
                # ตำแหน่ง elite-prey
                r_levy_cut_arc_cur_pos = position_minus_position(elite_gbest_cut_arc_cur_pos[sol],prey_gbest_cut_arc_cur_pos[sol])
                # สร้างความเร็ว step_size_r_levy (r_levy*(elite-prey))
                step_size_r_levy = coef_times_position_marine_r_levy(r_levy_cut_arc_cur_pos, bound)
                # เอาความเร็วเริ่มต้นบวกความเร็ว Levy (step_size_cur_prey + step_size_r_levy)
                new_step_size_velocity_levy = add_velocity(step_size_cur_prey, step_size_r_levy)
                # ไม่คูณ P และ R randomm
                # check_velocity_inconsistency
                check_velocity_step_size_levy = check_velocity_inconsistency(new_step_size_velocity_levy)
                # cut_set
                step_size_cut_set_levy = create_cut_set_r_levy(check_velocity_step_size_levy,alpha_levy)
                # update_prey ใหม่ (prey + step_size_cut_set)
                new_prey_cur_pos_phase_2_levy, new_prey_arc_cur_pos_phase_2_levy = sol_position_update(step_size_cut_set_levy,prey_gbest_cut_arc_cur_pos[sol], sub_E_list, prey_cur_pos[sol][0], elite_arc_cur_pos[sol][0][0])
                # FADs
                # new_prey_cur_pos_phase_2_levy = fad_marine_single(new_prey_cur_pos_phase_2_levy, fads_rate, fads)
                # preyค่าใหม่ แทน preyค่าเก่า
                prey_cur_pos[sol] = new_prey_cur_pos_phase_2_levy
                prey_arc_cur_pos[sol] = new_prey_arc_cur_pos_phase_2_levy
                #หาค่าคำตอบของ prey แต่ละตัว ใน population หา picker, tardiness, batch
                picker_assignment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(prey_cur_pos[sol],df_item_pool,heavy_item_set,name_path_input)
                picker_assignment_all_sols[sol] = picker_assignment_sol[:]
                total_tardiness_all_sols[sol] = total_tardiness_sol
                item_in_batch_all_sols[sol] = item_in_batch_sol[:]
                # สร้าง tardiness อยู่กับ arc พร้อมบอกลำดับ index
                prey_tardiness_arc_cur_pos_index[sol] = [total_tardiness_sol, new_prey_arc_cur_pos_phase_2_levy,sol]
            # เรียงลำดับค่า tardiness น้อยไปมาก
            ranked_prey_gbest_tardiness_arc_cur_pos_index = sorted(prey_tardiness_arc_cur_pos_index)
            # หาค่า g_best หาตัวที่มีค่าต่ำสุด
            gbest_arc_pos_for_gen = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:]
            # สร้าง elite หน้าตาเหมือนกันจาก g_best และ cut set
            elite_gbest_arc_cur_pos = [ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:] for sol in range(pop_size)]
            elite_gbest_cut_arc_cur_pos = [cut_arc_sol(elite_gbest_arc_cur_pos[sol]) for sol in range(pop_size)]
            elite_gbest_tardiness_arc_cur_pos_index = ranked_prey_gbest_tardiness_arc_cur_pos_index[0]

        #Phase 3 : While Iter > 2/3 Max_Iter  (levy)
        elif iter >= int(2 / 3 * num_gen) and iter < num_gen:
        # while iter >= int(2 / 3 * num_gen) and iter < num_gen:
            for sol in range(pop_size):
                # สร้างความเร็วเริ่มต้น step_size_cur_prey
                step_size_cur_prey = init_velocity_sol(prey_gbest_cut_arc_cur_pos[sol])
                # (elite-prey)
                r_levy_cut_arc_cur_pos = position_minus_position(elite_gbest_cut_arc_cur_pos[sol], prey_gbest_cut_arc_cur_pos[sol])
                # สร้างความเร็วเริ่มต้น step_size_r_levy (r_levy*(elite-prey))
                step_size_r_levy = coef_times_position_marine_r_levy(r_levy_cut_arc_cur_pos, bound)
                # เอา step_size_cur_prey + step_size_r_levy
                new_step_size_r_levy = add_velocity(step_size_cur_prey, step_size_r_levy)
                # new_step_size_r_levy * P.CF
                new_step_size_velocity_levy = coef_times_velocity_cpf(calculate_p_cf(iter, num_gen,p),new_step_size_r_levy)
                # check_velocity_inconsistency
                check_velocity_step_size = check_velocity_inconsistency(new_step_size_velocity_levy)
                # cut_set
                step_size_cut_set = create_cut_set_r_levy(check_velocity_step_size, alpha_levy)
                # update_prey ใหม่ (prey + step_size_cut_set)
                new_prey_cur_pos_phase_3, new_prey_arc_cur_pos_phase_3 = sol_position_update(step_size_cut_set,elite_gbest_cut_arc_cur_pos[sol], sub_E_list,prey_cur_pos[sol][0],elite_arc_cur_pos[sol][0][0])
                # FADs
                # new_prey_cur_pos_phase_3 = fad_marine_single(new_prey_cur_pos_phase_3, fads_rate, fads)
                # preyค่าใหม่ แทน preyค่าเก่า
                prey_cur_pos[sol] = new_prey_cur_pos_phase_3
                prey_arc_cur_pos[sol] = new_prey_arc_cur_pos_phase_3
                #หาค่าคำตอบของ prey แต่ละตัว ใน population หา picker, tardiness, batch
                picker_assignment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(prey_cur_pos[sol],df_item_pool,heavy_item_set,name_path_input)
                picker_assignment_all_sols[sol] = picker_assignment_sol[:]
                total_tardiness_all_sols[sol] = total_tardiness_sol
                item_in_batch_all_sols[sol] = item_in_batch_sol[:]
                # สร้าง tardiness อยู่กับ arc พร้อมบอกลำดับ index
                prey_tardiness_arc_cur_pos_index[sol] = [total_tardiness_sol, new_prey_arc_cur_pos_phase_3, sol]
                # เรียงลำดับค่า tardiness น้อยไปมาก
            ranked_prey_gbest_tardiness_arc_cur_pos_index = sorted(prey_tardiness_arc_cur_pos_index)
            # หาค่า g_best หาตัวที่มีค่าต่ำสุด
            gbest_arc_pos_for_gen = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:]
            # สร้าง elite หน้าตาเหมือนกันจาก g_best และ cut set
            elite_gbest_arc_cur_pos = [ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1][:] for sol in range(pop_size)]
            elite_gbest_cut_arc_cur_pos = [cut_arc_sol(elite_gbest_arc_cur_pos[sol]) for sol in range(pop_size)]
            elite_gbest_tardiness_arc_cur_pos_index = ranked_prey_gbest_tardiness_arc_cur_pos_index[0]

        tardiness_elite_value = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][0]  #เก็บ tardiness_gbest_value
        elite_arc_pos = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][1]
        # เช็ค tardiness ที่ดีที่สุด
        if tardiness_elite_value <= tardiness_gbest_value:
            tardiness_gbest_value = tardiness_elite_value
            gbest_arc_pos = elite_arc_pos

            elite_index = ranked_prey_gbest_tardiness_arc_cur_pos_index[0][2]
            elite_item_in_batch = item_in_batch_all_sols[elite_index]
            picker_assignment = picker_assignment_all_sols[elite_index]
            item = []
            item_by_batch = elite_item_in_batch   #เก็บ item_in_batch_all_sols
            picker_by_batch = picker_assignment   #เก็บ picker_assignment_all_sols
            for i in elite_item_in_batch:
                item.extend(i)
                gbest_item_in_batch = item
            # print(f'Item is {item}')
            # print(f'Item by batch is {item_by_batch}')
            # print(f'Picker by batch is {picker_by_batch}')
        gbest_each_gen.append(tardiness_gbest_value)   #เก็บ gbest_each_gen
    #     print(f'รอบที่ : {iter}')
    # print(f'Tardiness gbest is {tardiness_gbest_value}')
    # print(f'Tardiness gbest for each gen is {gbest_each_gen}')
    return tardiness_gbest_value, gbest_each_gen


mu = 0
sigma = 1
p = [1.5,2]
alpha_cut_set = [0.6,0.8]
alpha_levy = [0.7,1]
bound = [3,5,7]
num_pop = [(1,1),(1,1)]

ng, ps = num_pop[0]

headers = ['mu', 'sigma', 'p', 'alpha_cut_set', 'alpha_levy', 'bound', 'num_pop', 'tardiness_gbest_value', 'gbest_each_gen', 'replication','seeds']
results = []

# Perform initial time measurements
num_rep = 5
measured_times = measure_initial_iterations(num_rep)
average_time = calculate_average_time(measured_times)

# Calculate total number of iterations
total_iterations = len(p) * len(alpha_cut_set) * len(alpha_levy) * len(bound) * len(num_pop)*num_rep*num_rep

# Print estimated completion time
print_estimated_completion_time(average_time, total_iterations)
seed_list = [1132,1456,1975,2492,2820]
# Processing loop with progress bar
with tqdm(total=total_iterations, desc="Processing", file=sys.stdout) as pbar:
    for r in range(num_rep):
        for ng, ps in num_pop:
            for a in p:
                for b in alpha_cut_set:
                    for c in alpha_levy:
                        for d in bound:
                            for rep in range(num_rep):
                                seed_num = seed_list[rep]
                                random.seed(seed_num)
                                numpy_randomGen = Generator(PCG64(seed_num))
                                truncnorm.random_state = numpy_randomGen
                                start_time = time.time()
                                tardiness_gbest_value, gbest_each_gen = marine_predator(name_path_input, ng, ps, mu, sigma, a, b, c, d)
                                num_pop_str = (ng, ps)
                                results.append([mu, sigma, a, b, c, d, num_pop_str, tardiness_gbest_value, gbest_each_gen, r, seed_num])
                                pbar.update(1)

with open('Testwrite_Marine_1R-20I-150C-2P_ver6.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for x in results:
        writer.writerow(x)

print("Process completed.")




# running_time = timeit.default_timer() - start_time
# running_time_list.append(running_time)
# print(f'running time is {running_time}')


# # parameters
# num_gen = 10
# pop_size = 5
# fads_rate = 0.5
# alpha_cut_set = 0.6
# mu = 0
# sigma = 1
# p = 2
# alpha_levy = 1
# bound = 5
# fads = 0.2
# tardiness_gbest_value, gbest_each_gen = marine_predator(name_path_input, num_gen, pop_size, mu, sigma, p, alpha_cut_set, alpha_levy, bound, fads_rate, fads)

# # Plotting graph
# plt.figure(figsize=(10, 5))
# plt.plot(gbest_each_gen, label='Best Global Solution (gbest)')
# plt.xlabel(f'Time taken')
# plt.ylabel('Tardiness')
# plt.title(f'Marine Algorithm - {name_path_input} - {pop_size} Population size - {num_gen} Generations - Total tardiness: {tardiness_gbest_value}')
# # Set the x-axis limit from 0 to num_gen
# plt.xlim(0, num_gen)
# plt.legend()
# plt.show()
# plt.close()






