import pandas as pd
import time
import random
from evaluate_all_sols import*
import numpy as np
import itertools
import matplotlib.pyplot as plt
value_heavy = 40
random.seed(3124)

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


def sol_from_list_to_arc(sol):
    num_item = len(sol)
    arc_sol = [(sol[i],sol[i+1]) for i in range(num_item-1)]
    return arc_sol
def all_sols_from_list_to_arc(all_sols):
    num_sol = len(all_sols) #เก็บจำนวนคำตอบของ all_sols
    num_item = len(all_sols[0])
    all_arc_sols = [[] for _ in range(num_sol)]
    #สร้าง list ซ้อน list โดยใช้ list
    all_arc_sols = [[]for _ in range(num_sol)]

    return all_arc_sols



def all_sols_from_list_to_arc(all_sols):
    num_sol = len(all_sols)#เก็บจำนวนคำตอบของ all_sols
    num_item = len(all_sols[0])

    #สร้างlistซ้อน list โดยใช้ list comprehension
    all_arc_sols = [[] for _ in range(num_sol)]
    #อีกทางเลือกหนึ่ง ถ้าไม่อยากใช้ list comprehension คือ วน for loop
    for i in range(num_sol):
        for j in range(num_item-1):
            all_arc_sols[i].append((all_sols[i][j],all_sols[i][j+1]))
    return all_arc_sols

def cut_arc_sol(arc_sol): #argumennt เป็น arc_sol ของคำตอบเดียว
    num_item = len(arc_sol)+1
    arc_sol_cut = [[]for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol:
            if item == arc[0] or item == arc[1]:
                arc_sol_cut[item].append(arc)
    return arc_sol_cut

def init_velocity_sol(arc_sol_cut):
    import random
    num_item=len(arc_sol_cut) #นับจำนวน item จากจำนวนสมาชิกของ list arc_sol_cut
    arc_sol_velocity_dict=[{}for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut[item]:
            arc_sol_velocity_dict[item][arc]=round(random.random(),4)
    return arc_sol_velocity_dict

def coef_times_velocityold(coef,arc_sol_velocity_dict):
    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_velocity_dict[item].keys():
            if coef*arc_sol_velocity_dict[item][arc]>1:
                coef_times_velocity_dict[item][arc]=1
            else:
                coef_times_velocity_dict[item][arc]=round(coef * arc_sol_velocity_dict[item][arc],4)
    return coef_times_velocity_dict

def coef_times_velocity(coef,arc_sol_velocity_dict):
    num_item = len(arc_sol_velocity_dict)
    coef_times_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_velocity_dict[item].keys():
            if coef*arc_sol_velocity_dict[item][arc]>1:
                coef_times_velocity_dict[item][arc]=1
            else:
                coef_times_velocity_dict[item][arc]=round(coef * arc_sol_velocity_dict[item][arc],4)
    return coef_times_velocity_dict

def position_minus_position(arc_first,arc_second):
    num_item = len(arc_first)
    pos_minus_pos =[[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                pos_minus_pos[item].append(arc)
    return pos_minus_pos

def coef_times_position(c_value, arc_diff):
    import  random
    num_item = len(arc_diff)
    coef_times_position_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_diff[item]:
            coef = c_value*random.random()
            if coef >1:
                coef = 1
            coef_times_position_dict[item][arc] = round(coef,4)
    return  coef_times_position_dict

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
    return  added_velocity_dict


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

def create_cut_set(added_velocity_dict, alpha):
    num_item=len(added_velocity_dict)
    cut_set = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc]>= alpha:
                cut_set[item].append(arc)
    return cut_set

def select_dest_from_source(source,picked_list,*sets):
    #ฟังก์ชันนี้ทำหน้าที่ในการเลือก item ที่เราจะเดินเก็บถัดไป(dest)จากตำแหน่งปัจจุบันที่เราอยู่ (source) โดยเราต้องได้ผลลัพธ์เป็น arc ของ (source,dest)
    # และ picked_list เป็น list ที่เก็บ item ที่เราเดินเก็บไปแล้ว
    import  random
    for set in sets:
        new_set = []
        if len(set[source])>0:
            for arc in set[source]:
                if arc[1]not in picked_list and arc[0] == source:
                    new_set.append(arc)
        if len(new_set)>0:
            dest = random.choice(new_set)[1]
            break
    arc_source_dest = (source,dest)
    return dest,arc_source_dest

def sol_position_update(cut_set, previous_x , sub_E_list, start_previous_x, start_gbest):
    import random
    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []
    source = random.choice([start_previous_x, start_gbest,random.choice(range(num_item))])
    picked_list.append(source)
    for item_counter in range(num_item-1):
        dest, arc_source_dest = select_dest_from_source(source, picked_list, cut_set, previous_x, sub_E_list)
        source = dest
        picked_list.append(dest)
        picked_list_arc.append(arc_source_dest)
    return picked_list,picked_list_arc


def squirrel(name_path_input,num_gen,pop_size,*parameters):
    from itertools import permutations
    # input data
    dg,Gc,Pdp = parameters
    df_item_pool,df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]
    list_item = list(range(num_item))
    E_all = list(permutations(list_item,2))
    sub_E_list = [[arc for arc in E_all if arc[0]==item or arc[1]==item] for item in range(num_item)]
    num_item = len(df_item_pool)
    heavy_item_set = set(df_item_pool[df_item_pool['weight'] >= value_heavy].index)

    # -----------------------------สร้างคำตอบเริ่มต้น-----------------------------
    squirrel_cur_pos = []
    for sol in range(pop_size):
        temp_pos = list(range(num_item))
        random.shuffle(temp_pos)
        squirrel_cur_pos.append(temp_pos)
    picker_assigment_all_sols = []
    total_tardiness_all_sols = []
    item_in_batch_all_sols = []

    for sol in range(pop_size):
        picker_assigment_sol,total_tardiness_sol,item_in_batch_sol = evaluate_all_sols(squirrel_cur_pos[sol],df_item_pool,heavy_item_set,name_path_input)
        picker_assigment_all_sols.append(picker_assigment_sol)
        total_tardiness_all_sols.append(total_tardiness_sol)
        item_in_batch_all_sols.append(item_in_batch_sol)
    # สร้าง list ของคำตอบใหม่ หลังจากซ่อมคำตอบแล้ว
    squirrel_cur_pos = [[] for sol in range(pop_size)]
    for sol in range(pop_size):
        for batch in item_in_batch_all_sols[sol]:
            squirrel_cur_pos[sol].extend(batch)
    squirrel_arc_cur_pos = all_sols_from_list_to_arc(squirrel_cur_pos)
    squirrel_cut_arc_cur_pos = [cut_arc_sol(squirrel_arc_cur_pos[sol]) for sol in range(pop_size)]

    squirrel_tardiness_arc_cur_pos_index = [[total_tardiness_all_sols[sol],squirrel_cut_arc_cur_pos[sol],
    sol] for sol in range(pop_size)]
    ranked_squirrel_tardiness_cur_pos_index = sorted(squirrel_cur_pos)
    ranked_squirrel_tardiness_arc_cur_pos_index =sorted(squirrel_tardiness_arc_cur_pos_index) #จะมีการเปลี่ยนแปลงคำตอบกัน
    ranked_squirrel_velo = [init_velocity_sol(ranked_squirrel_tardiness_arc_cur_pos_index[sol][1]) for sol in range(pop_size)]
    squirrel_velo = [ranked_squirrel_velo[i] for i in range(len(ranked_squirrel_velo))]


    num_oak_tree = 3
    num_normal_tree = pop_size - num_oak_tree -1

    #สร้าง list สำหรับเก็บค่า tardiness ของแต่ละต้นไม้ และคำตอบของแต่ละต้นไม้ ในแต่ละรุ่น (gen)
    hickory_tardiness_each_gen = []
    hickory_arc_sol_each_gen = []
    oak_to_hickory_tardiness_each_gen = [[] for _ in range(num_gen)]
    oak_random_tardiness_each_gen = [[] for _ in range(num_gen)]
    normal_to_hickory_tardiness_each_gen = [[] for _ in range(num_gen)]
    normal_to_oak_tardiness_each_gen = [[] for _ in range(num_gen)]

    for gen in range(num_gen):
        hickory_tree_tardiness_arc_cur_pos_index = ranked_squirrel_tardiness_arc_cur_pos_index[0]
        # hickory_tree_cut_arc_cur_pos = ranked_squirrel_tardiness_arc_cur_pos_index[0][1]
        hickory_tree_cut_arc_cur_pos = hickory_tree_tardiness_arc_cur_pos_index[1]
        hickory_tardiness_each_gen.append(hickory_tree_tardiness_arc_cur_pos_index[0])
        hickory_arc_sol_each_gen.append(hickory_tree_tardiness_arc_cur_pos_index[1])

        oak_to_hickory_tardiness_each_gen.append([])
        oak_tree_cur_sol = []
        oak_tree_arc_sol = []
        #ปรับปรุงคำตอบ oak tree โดยเคลื่อนที่เข้าหา hickory tree
        for t in range(1,(num_oak_tree+1)):
            oak_to_hickory_tardiness_each_gen[gen].append((ranked_squirrel_tardiness_arc_cur_pos_index[t][0]))
            if random.random() >Pdp:
                oak_tree_cut_arc_cur_pos = ranked_squirrel_tardiness_arc_cur_pos_index[t][1]
                hickory_minus_oak = position_minus_position(hickory_tree_cut_arc_cur_pos, oak_tree_cut_arc_cur_pos)
                oak_tree_velo_gbest = coef_times_position(1, hickory_minus_oak)
                oak_tree_prev_velo = coef_times_velocity(0.7, ranked_squirrel_velo[t])
                oak_tree_added_velo = add_velocity(oak_tree_prev_velo, oak_tree_velo_gbest)
                oak_tree_added_velo = check_velocity_inconsistency(oak_tree_added_velo)
                oak_tree_cut_set = create_cut_set(oak_tree_added_velo, 0.6)
                start_x = squirrel_cur_pos[ranked_squirrel_tardiness_arc_cur_pos_index[t][2]][0]
                start_gbest = squirrel_cur_pos[ranked_squirrel_tardiness_arc_cur_pos_index[0][2]][0]
                new_sol_oak_tree, new_sol_oak_tree_arc = sol_position_update(oak_tree_cut_set, oak_tree_cut_arc_cur_pos,sub_E_list, start_x, start_gbest)
                squirrel_velo[t] = oak_tree_added_velo
            else:
                new_sol_oak_tree = list(range(num_item))
                random.shuffle(new_sol_oak_tree)
                new_sol_oak_tree_arc = sol_from_list_to_arc(new_sol_oak_tree)
                new_sol_oak_tree_cut_arc = cut_arc_sol(new_sol_oak_tree_arc)
                new_velo = init_velocity_sol(new_sol_oak_tree_cut_arc)
                squirrel_velo[t] = new_velo
            picker_assigment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(new_sol_oak_tree,df_item_pool,heavy_item_set,name_path_input)
            squirrel_cur_pos[t] = []
            for batch in item_in_batch_sol:
                squirrel_cur_pos[t].extend(batch)
            squirrel_arc_cur_pos[t] = sol_from_list_to_arc(squirrel_cur_pos[t])
            squirrel_cut_arc_cur_pos[t] = cut_arc_sol(squirrel_arc_cur_pos[t])
            squirrel_tardiness_arc_cur_pos_index[t] = [total_tardiness_sol, squirrel_cut_arc_cur_pos[t], t]

    #----------------------------------------------------------------------------------------------------------------
        normal_tree_cur_sol = []
        normal_tree_arc_sol = []
    # ปรับปรุงคำตอบ normal tree โดยเคลื่อนที่เข้าหา hickory tree
        for t in range(num_oak_tree + 1, (num_oak_tree + 1 + int(num_normal_tree / 2))):
            normal_to_hickory_tardiness_each_gen[gen].append((ranked_squirrel_tardiness_arc_cur_pos_index[t][0]))
            if random.random() > Pdp:
                normal_tree_cut_arc_cur_pos = ranked_squirrel_tardiness_arc_cur_pos_index[t][1]
                hickory_minus_normal = position_minus_position(hickory_tree_cut_arc_cur_pos, normal_tree_cut_arc_cur_pos)
                normal_to_hickory_tree_velo_gbest = coef_times_position(1, hickory_minus_normal)
                normal_to_hickory_tree_prev_velo = coef_times_velocity(0.7, ranked_squirrel_velo[t])
                normal_tree_added_velo = add_velocity(normal_to_hickory_tree_prev_velo, normal_to_hickory_tree_velo_gbest)
                normal_tree_added_velo = check_velocity_inconsistency(normal_tree_added_velo)
                normal_tree_cut_set = create_cut_set(normal_tree_added_velo, 0.6)

                start_x = squirrel_cur_pos[ranked_squirrel_tardiness_arc_cur_pos_index[t][2]][0]
                start_gbest = squirrel_cur_pos[ranked_squirrel_tardiness_arc_cur_pos_index[0][2]][0]
                new_sol_normal_tree, new_sol_normal_tree_arc = sol_position_update(normal_tree_cut_set, normal_tree_cut_arc_cur_pos,sub_E_list, start_x, start_gbest)
            else:
                new_sol_normal_tree = list(range(num_item))
                random.shuffle(new_sol_normal_tree)
                new_sol_normal_tree_arc = sol_from_list_to_arc(new_sol_normal_tree)

            picker_assigment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(new_sol_oak_tree,df_item_pool,heavy_item_set,name_path_input)
            squirrel_cur_pos[t] = []
            for batch in item_in_batch_sol:
                squirrel_cur_pos[t].extend(batch)
            squirrel_arc_cur_pos[t] = sol_from_list_to_arc(squirrel_cur_pos[t])
            squirrel_cut_arc_cur_pos[t] = cut_arc_sol(squirrel_arc_cur_pos[t])
            squirrel_tardiness_arc_cur_pos_index[t] = [total_tardiness_sol, squirrel_cut_arc_cur_pos[t], t]
    #------------------------------------------------------------------------------------------------------------------------------
        oak_tree_list = list(range(1,num_oak_tree+1))
        # ลองทำงานกับ for loop สำหรับ ต้น normal
        for t in range(num_oak_tree+1+int(num_normal_tree/2), pop_size):
            oak_tree_random = random.choice(oak_tree_list)
            oak_random_tardiness_arc_cur_pos_index = [ranked_squirrel_tardiness_arc_cur_pos_index[oak_tree_random]]
            oak_random_cut_arc_cur_pos = oak_random_tardiness_arc_cur_pos_index[0][1]
            oak_random_tardiness_each_gen.append(oak_random_tardiness_arc_cur_pos_index[0][0])
            normal_to_oak_tardiness_each_gen[gen].append((ranked_squirrel_tardiness_arc_cur_pos_index[t][0]))
            if random.random() > Pdp:
                normal_tree_cut_arc_cur_pos = ranked_squirrel_tardiness_arc_cur_pos_index[t][1]
                oak_minus_normal = position_minus_position(oak_tree_cut_arc_cur_pos, normal_tree_cut_arc_cur_pos)
                normal_to_oak_tree_velo_gbest = coef_times_position(1, oak_minus_normal)
                normal_to_oak_tree_prev_velo = coef_times_velocity(0.7, ranked_squirrel_velo[t])
                normal_tree_added_velo = add_velocity(normal_to_oak_tree_prev_velo, normal_to_oak_tree_velo_gbest)
                normal_tree_added_velo = check_velocity_inconsistency(normal_tree_added_velo)
                normal_tree_cut_set = create_cut_set(normal_tree_added_velo, 0.6)
                start_x = squirrel_cur_pos[ranked_squirrel_tardiness_arc_cur_pos_index[t][2]][0]
                start_gbest = squirrel_cur_pos[ranked_squirrel_tardiness_arc_cur_pos_index[0][2]][0]
                new_sol_normal_tree, new_sol_normal_tree_arc = sol_position_update(normal_tree_cut_set,normal_tree_cut_arc_cur_pos,sub_E_list, start_x, start_gbest)
            else:
                new_sol_normal_tree = list(range(num_item))
                random.shuffle(new_sol_normal_tree)
                new_sol_normal_tree_arc = sol_from_list_to_arc(new_sol_normal_tree)
            picker_assigment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(new_sol_oak_tree,df_item_pool,heavy_item_set,name_path_input)
            squirrel_cur_pos[t] = []
            for batch in item_in_batch_sol:
                squirrel_cur_pos[t].extend(batch)
            squirrel_arc_cur_pos[t] = sol_from_list_to_arc(squirrel_cur_pos[t])
            squirrel_cut_arc_cur_pos[t] = cut_arc_sol(squirrel_arc_cur_pos[t])
            squirrel_tardiness_arc_cur_pos_index[t] = [total_tardiness_sol, squirrel_cut_arc_cur_pos[t], t]
        # เปรียบเทียบคำตอบใหม่กับคำตอบเดิม และเลือกคำตอบที่ดีที่สุด และ ranking คำตอบ
        ranked_squirrel_tardiness_arc_cur_pos_index = sorted(squirrel_tardiness_arc_cur_pos_index)
        ranked_squirrel_velo = [squirrel_velo[i] for i in range(len(squirrel_velo))]
        print("xxx")
    return hickory_tardiness_each_gen


num_gen = 2
pop_size = 10


start_time = time.time()
name_path_input = '1R-20I-150C-2P'
df_item_pool =read_input(name_path_input)
hickory_tardiness_each_gen = squirrel(name_path_input,num_gen,pop_size,1,2,3)


# End the timer
end_time = time.time()
time_taken = end_time - start_time

# Convert time_taken to hours, minutes, and seconds
hours = int(time_taken // 3600)
minutes = int((time_taken % 3600) // 60)
seconds = time_taken % 60

# Display final results
hours, remainder = divmod(time_taken, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
print(f"Time Taken (second) : {time_taken:.2f}")
print("----" * 50)


# Plotting the graph with proper limits
plt.figure(figsize=(10, 5))
plt.plot(hickory_tardiness_each_gen, label='Best Global Solution (gbest)', marker='_')
plt.xlabel(f"Time Taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
plt.ylabel('Tardiness')
plt.title(f'(Mayfly Algorithm - {name_path_input}) - {pop_size} Population Size - {num_gen} Generations - Total tardiness: {min(hickory_tardiness_each_gen)}')

# Set the x-axis limit from 0 to num_gen
plt.xlim(0, num_gen)

# Add the legend
plt.legend()
plt.show()








