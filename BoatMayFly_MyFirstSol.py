import pandas as pd
import random
from evaluate_all_sols  import *
random.seed(1234)

df_item_sas_random = pd.read_csv('df_item_sas_random.csv')
name_path_input = '1I-10-100-2'
df_duedate = pd.read_csv(name_path_input +'\\duedate_'+name_path_input +'.csv', header=None)
df_item_order = pd.read_csv(name_path_input + '\\input_location_item_'+ name_path_input +'.csv', header=None)

list_duedate = df_duedate[0].tolist()

num_order = df_item_order.shape[1]
list_order = []
list_total_item = []
df_item_pool = pd.DataFrame()

for order in range(num_order):
    item = df_item_order[order][df_item_order[order] != 0]
    df_item_in_this_order = df_item_sas_random[df_item_sas_random['location'].isin(item)].copy()
    df_item_in_this_order['duedate'] = list_duedate[order]
    df_item_pool = pd.concat([df_item_pool, df_item_in_this_order])
    num_item_this_order = df_item_in_this_order.shape[0]
    for i in range(num_item_this_order):
        list_order.append(order)
    list_total_item.extend(item.tolist())

df_item_pool['order'] = list_order
df_item_pool.reset_index(drop=True, inplace=True)

num_item = df_item_pool.shape[0]
num_sol = 5
cur_sol = []

#ก่อนปรับปรุงเป็น list comprehension
for sol in range(num_sol):
    now_sol = list(range(num_item))
    random.shuffle(now_sol)
    cur_sol.append(now_sol)

# a คือจำนวนพนักงานที่เดินเก็บ (picker)
# b คือ ค่าความล่าช้ารวม(Total_Tardiness)
# c คือ ชุดคำสั่งซื้อ(Batch)


# a,b,c = evaluate_all_sols(cur_sol[4],df_item_pool,name_path_input)
# print(f'b = {b}')

# ก่อนปรับปรุงเป็น list comprehension
# def sol_from_list_to_arc(sol):
#     num_item = len(sol)
#     arc_sol = []
#     for i in range(num_item - 1):
#         arc_sol.append((sol[i], sol[i + 1]))
#     return arc_sol
def sol_from_list_to_arc(sol):
    return [(sol[i], sol[i + 1]) for i in range(len(sol) - 1)] # หลังจากปรับปรุงเป็น list comprehension


arc_sol = sol_from_list_to_arc(cur_sol[0])

def all_sols_from_list_to_arc(all_sols):
    num_sol = len(all_sols)
    num_item = len(all_sols[0])
    all_arc_sols = [[(all_sols[i][j], all_sols[i][j + 1]) for j in range(num_item - 1)] for i in range(num_sol)]
    return all_arc_sols
all_arc_sols = all_sols_from_list_to_arc(cur_sol)

#ก่อนปรับปรุงเป็น list comprehension
def cut_arc_sol(arc_sol):
    num_item = len(arc_sol) + 1
    arc_sol_dict = {item: [] for item in range(num_item)}
    for arc in arc_sol:
        arc_sol_dict[arc[0]].append(arc)
        arc_sol_dict[arc[1]].append(arc)

    arc_sol_cut = [arc_sol_dict[item] for item in range(num_item)]

    return arc_sol_cut
def cut_arc_sol(arc_sol): # หลังจากปรับปรุงเป็น list comprehension
    return [{arc for arc in arc_sol if arc[0] == item or arc[1] == item} for item in range(len(arc_sol) + 1)]
arc_sol_cut = cut_arc_sol(all_arc_sols[0])

# ก่อนปรับปรุงเป็น list comprehension
# def init_velocity_sol(arc_sol_cut):
#     num_item = len(arc_sol_cut)
#     arc_sol_velocity_dict = [{} for _ in range(num_item)]
#     for item in range(num_item):
#         for arc in arc_sol_cut[item]:
#             arc_sol_velocity_dict[item][arc] = round(random.random(), 4)
#     return arc_sol_velocity_dict
def init_velocity_sol(arc_sol_cut):# หลังจากปรับปรุงเป็น list comprehension
    return [{arc: round(random.random(), 4) for arc in arc_list} for arc_list in arc_sol_cut]

arc_sol_velocity_dict = init_velocity_sol(arc_sol_cut)
# print(f'arc_sol_velocity_cut = {arc_sol_velocity_dict}')

# ปรับปรุงความเร็ว vit+1 = wvti + c1r1(Pbest - xi) + c2r2(Gbest - xi)

# ก่อนปรับปรุงเป็น list comprehension
# def coef_time_volocity(coef,arc_sol_velocity_dict):
#     num_item = len(arc_sol_velocity_dict)
#     coef_time_volocity_dict = [{} for item in range(num_item)]
#     #[{} {} {} ...]
#     for item in range(num_item):
#         for arc in arc_sol_velocity_dict[item].keys():
#             if coef*arc_sol_velocity_dict[item][arc] > 1:
#                 coef_time_volocity_dict[item][arc] = 1
#             else:
#                 coef_time_volocity_dict[item][arc] = round(coef*arc_sol_velocity_dict[item][arc],4)
#     return  coef_time_volocity_dict

# หลังจากปรับปรุงเป็น list comprehension
def coef_time_volocity(coef, arc_sol_velocity_dict):
    return [
        {arc: min(1, round(coef * speed, 4)) for arc, speed in velocity_dict.items()}
        for velocity_dict in arc_sol_velocity_dict
    ]
coef_time_volocity_dict = coef_time_volocity(0.7,arc_sol_velocity_dict)
# print(f'coef_time_volocity_dict = {coef_time_volocity_dict}')

# #[[(0,1),(2,0)],[(0,1),(1,22)] .... ]
arc_first = [[(0,2)],[(2,1)],[(0,2),(2,1)]]
arc_second = [[(0,1)],[(0,1),(1,2)],[(1,2)]]

# ก่อนปรับปรุงเป็น list comprehension
# def position_minus_position(arc_first,arc_second):
#     num_item = len(arc_first)
#     pos_minus_pos = [[] for item in range(num_item)]
#     for item in range(num_item):
#         for arc in arc_first[item]:
#             if arc not in arc_second[item]:
#                 pos_minus_pos[item].append(arc)
#     return pos_minus_pos

# หลังจากปรับปรุงเป็น list comprehension
def position_minus_position(arc_first, arc_second):
    return [[arc for arc in first_set if arc not in second_set] for first_set, second_set in zip(arc_first, arc_second)]
pos_minus_pos = position_minus_position(arc_first,arc_second)
# print(f'pos_minus_pos = {pos_minus_pos}')

#arc_diff = [[(0, 2)], [(2, 1)], [(0, 2), (2, 1)]]

# ก่อนปรับปรุงเป็น list comprehension
# def coef_time_position_MrGumNhud(c_value, arc_diff):
#     import random
#     num_item = len(arc_diff)
#     coef_time_position_dict = [{} for item in range(num_item)]
#     for item in range (num_item):
#         for arc in arc_diff[item]:
#             coef = c_value*random.random()
#             if coef > 1 :
#                 coef = 1
#             coef_time_position_dict[item][arc] = round(coef,4)
#     return coef_time_position_dict

# หลังจากปรับปรุงเป็น list comprehension
def coef_time_position_MrGumNhud(c_value, arc_diff):
    return [
        {arc: round(min(c_value * random.random(), 1), 4) for arc in diff_list}
        for diff_list in arc_diff
    ]

coef_time_position = coef_time_position_MrGumNhud(2,pos_minus_pos)
# print(f'coef_time_position = {coef_time_position}')

v_first = [{(0,1):0.5, (2,0):0.3}, {(0,1):0.6}, {(2,0):0.4}] #2 0 1
v_second = [{(2,0):0.2}, {(1,2):0.9}, {(2,0):0.5, (1,2):0.8}] #1 2 0
#added_v = [{(0,1):0.5, (2,0):0.3}, {(0,1):0.6, (1,2):0.9}, {(2,0):0.5, (1,2):0.8}]
def add_velocity(velocity_fiest, velocity_second):
    num_item = len(velocity_fiest)
    added_velocity_dict = [{} for item in range(num_item)]
    for item in range(num_item):
        for arc in velocity_fiest[item]:
            added_velocity_dict[item][arc] = velocity_fiest[item][arc]
        for arc in velocity_second[item]:
            if arc in added_velocity_dict[item].keys():
                if velocity_second[item][arc] > added_velocity_dict[item][arc]:
                    added_velocity_dict[item][arc] = velocity_second[item][arc]
            else:
                added_velocity_dict[item][arc] = velocity_second[item][arc]
    return added_velocity_dict

added_v = add_velocity(v_first, v_second)

def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    import copy
    # new_added_velocity_dict = copy.deepcopy(added_velocity_dict)
    new_added_velocity_dict = [{arc: prob for arc, prob in added_velocity_dict[item].items()} for item in
                               range(num_item)]
    print(new_added_velocity_dict)
    for item in range(num_item):
        for arc_first in added_velocity_dict[item].keys():
            if arc_first in added_velocity_dict[arc_first[0]].keys():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[0]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[0]][arc_first]
                # else:
                #     new_added_velocity_dict[arc_first[0]][arc_first] = added_velocity_dict[item][arc_first]
            if arc_first in added_velocity_dict[arc_first[1]].keys():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[1]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[1]][arc_first]
                # else:
                #     new_added_velocity_dict[arc_first[1]][arc_first] = added_velocity_dict[item][arc_first]
    return new_added_velocity_dict

def create_cut_set(added_velocity_dict,alpha):
    num_item = len(added_velocity_dict)
    cut_set = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc] >= alpha:
                cut_set[item].append(arc)
    return cut_set

def select_desination_form_source(source,picked_list, *sets):
    # ฟังก์ชั่นนี้ทำหน้าที่ในการเลือก item ที่เราจะเดินเก็บถัดไป(desination) จากปัจจุบันที่เราอยู่(source) โดยเราต้องการได้ผลลัพท์เป็น arc ของ (source,desination) และ picked_list คือ list ที่เก็บ item ที่เราเดินเก็บไปแล้ว
    import random
    for set in sets:
        new_set = []
        if len(set[source])>0:
            for arc in set[source]:
                if arc[1] not in picked_list and arc[0] == source:
                    new_set.append(arc)
        if len(new_set)>0:
            destination = random.choice(new_set)
            break
    arc_source_destination = (source,destination)
    return destination, arc_source_destination

def sol_position_update(cut_set,previos_x,sub_E_list,alpha,start_previous_x,start_pbest,strat_gbest):
    import random
    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []

    source = random.choice(start_previous_x, start_pbest, strat_gbest,random.choice(range(num_item)))
    picked_list.append(source)
    for item_counter in range(num_item-1):
        destination, arc_source_destination = select_desination_form_source(source,picked_list,cut_set,previos_x,sub_E_list)
        source = destination
        picked_list.append(destination)
        picked_list_arc.append(arc_source_destination)
    return picked_list, picked_list_arc

print('---------'*30)

# สร้างคำตอบเริ่มต้นสำหรับ Mayfly
My_Mayfly_arc = all_sols_from_list_to_arc(cur_sol)

# แบ่งตำแหน่งของ Mayfly ออกเป็น Xi, Pbest และ Gbest
My_Mayfly_Xi = cut_arc_sol(My_Mayfly_arc[0])
My_Mayfly_Pbest = cut_arc_sol(My_Mayfly_arc[1])
My_Mayfly_Gbest = cut_arc_sol(My_Mayfly_arc[2])

# สร้างความเร็วเริ่มต้นสำหรับ Mayfly
My_Mayfly_Xi_velocity = init_velocity_sol(My_Mayfly_Xi)
My_Mayfly_Pbest_velocity = init_velocity_sol(My_Mayfly_Pbest)
My_Mayfly_Gbest_velocity = init_velocity_sol(My_Mayfly_Gbest)

# ใช้ค่าสัมประสิทธิ์กับความเร็ว Xi  vit+1 = wvti
My_Mayfly_Xi_coef_time_velocity = coef_time_volocity(0.7, My_Mayfly_Xi_velocity)

# คำนวณความต่างของตำแหน่ง (Pbest - Xi) และ (Gbest - Xi)
My_Mayfly_Pbest_Minus_Xi = position_minus_position(My_Mayfly_Pbest_velocity, My_Mayfly_Xi_velocity)
My_Mayfly_Gbest_Minus_Xi = position_minus_position(My_Mayfly_Gbest_velocity, My_Mayfly_Xi_velocity)

# ใช้ค่าสัมประสิทธิ์กับความต่างของตำแหน่ง c1r1(Pbest - Xi) และ c2r2(Gbest - Xi)
My_Mayfly_Pbest_Minus_Xi_With_c1r1 = coef_time_position_MrGumNhud(2, My_Mayfly_Pbest_Minus_Xi)
My_Mayfly_Gbest_Minus_Xi_With_c1r1 = coef_time_position_MrGumNhud(2, My_Mayfly_Gbest_Minus_Xi)

# คำนวณความเร็วใหม่สำหรับ Mayfly  vit+1 = wvti + c1r1(Pbest - xi) + c2r2(Gbest - xi)
My_Mayfly_New_Velocity = add_velocity(My_Mayfly_Pbest_Minus_Xi_With_c1r1, My_Mayfly_Gbest_Minus_Xi_With_c1r1)
New_velocity_for_Vit = add_velocity(My_Mayfly_Xi_coef_time_velocity, My_Mayfly_New_Velocity)

# ตรวจสอบความเร็วที่ไม่สอดคล้องกับเงื่อนไข
added_consistency_velocity = check_velocity_inconsistency(New_velocity_for_Vit)

# สร้าง cut set
cut_set = create_cut_set(added_consistency_velocity, 0.6)

#แสดงผล
print(f'My_Mayfly_Xi = {My_Mayfly_Xi}')
print(f'My_Mayfly_Xi_velocity = {My_Mayfly_Xi_velocity}')
print(f'My_Mayfly_Xi_coef_time_velocity ={My_Mayfly_Xi_coef_time_velocity}')

# print(f'My_Mayfly_New_Velocity = {My_Mayfly_New_Velocity}')
# print(f'New_velocity_for_Vit = {New_velocity_for_Vit}')
# print(f'added_consistency_velocity = {added_consistency_velocity}')
# print(f'cut_set = {cut_set}')


#ตัวแปรเก็บค่า b ที่ได้จากการประเมินค่าความล่าช้ารวม(Total_Tardiness)
#ตัวแปรเก็บค่า c ที่ได้จากการประเมินชุดคำสั่งซื้อ(Batch)
# My_First_Tardiness_After_3_Month = []
# My_First_Batch_After_3_Month = []
# # My_First_Picker_After_3_Month = []
# for sol in range(num_sol):
#     a,b,c = evaluate_all_sols(cur_sol[sol],df_item_pool,name_path_input)
#     # My_First_Picker_After_3_Month.append(a)
#     My_First_Tardiness_After_3_Month.append(b)
#     My_First_Batch_After_3_Month.append(c)
# # print(f'My_First_Picker_After_3_Month = {My_First_Picker_After_3_Month}')
# print(f'My_First_Tardiness_After_3_Month  = {My_First_Tardiness_After_3_Month }')
# print(f'My_First_Batch_After_3_Month = {My_First_Batch_After_3_Month}')
#
