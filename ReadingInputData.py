import pandas as pd
import random
random.seed(1234)
from evaluate_all_sols import *

def read_input(name_path_input):
    df_item_sas_random = pd.read_csv('df_item_sas_random.csv')
    # print(df_item_sas_random)

    name_path_input = '1I-10-100-2'

    df_duedate = pd.read_csv(name_path_input + '\\duedate_' + name_path_input + '.csv', header=None)
    # print(df_duedate)

    df_item_oder = pd.read_csv(name_path_input + '\\input_location_item_' + name_path_input + '.csv', header=None)
    # print(df_item_oder)

    list_duedate = df_duedate[0].values
    list_duedate = df_duedate[0].tolist()
    # print(list_duedate)
    # print(type(list_duedate))

    num_order = df_item_oder.shape[1]
    # print(num_order)

    list_order = []  # ทำหน้าที่ในการเก็บเลข order ของแต่ละ Item
    list_total_item = []  # ทำหน้าที่ในการเก็บเลข item ตามลำดับ order

    df_item_pool = pd.DataFrame()

    for order in range(num_order):
        item = df_item_oder[order][df_item_oder[order] != 0]
        # print(item)
        # print(df_item_sas_random['location'].isin(item))
        df_item_in_this_order = df_item_sas_random[df_item_sas_random['location'].isin(item)]
        df_item_in_this_order['duedate'] = list_duedate[order]
        # print(df_item_in_this_order)
        df_item_pool = pd.concat([df_item_pool, df_item_in_this_order])
        # print(df_item_pool)

        num_item_this_order = df_item_in_this_order.shape[0]
        for i in range(num_item_this_order):
            list_order.append(order)
        # print(list_order)
        list_total_item.extend(item.tolist())
        # print(list_total_item)
    # print(df_item_pool)
    df_item_pool['order'] = list_order
    # print(df_item_pool)
    df_item_pool.reset_index(drop=True, inplace=True)
    # print(df_item_pool)
    return df_item_pool, df_item_sas_random
# num_item = df_item_pool.shape[0]
num_sol = 5 #จำนวนคำตอบในแต่ละรอบ
cur_sol = []


#แปลงจาก list ให้กลายเป็น arc ต้องรู้ว่ามีทั้งหมดกี่ตัว
def sol_from_list_to_arc(sol):
    num_item = len(sol)
    arc_sol = []
    for i in range(num_item-1):
        arc_sol.append((sol[i],sol[i+1]))
    return arc_sol

# arc_sol = sol_from_list_to_arc(cur_sol[0])
# print(arc_sol)

def all_sol_from_list_to_arc(all_sols):
    num_sol = len(all_sols) #เก็บจำนวนคำตอบของ all_sols
    num_item = len(all_sols[0])

    #สร้าง list ซ้อน list โดยใช้ list comprehension
    #all_arc_sols = [[] for _ in range(num_sol)]

    #อีกทางเลือกหนึ่ง ถ้าไม่อยากใช้ list comprehension คือ วน for loop
    #all_arc_sols = []
    #for _ in range(num_sol):
    #    all_arc_sols.append()
    all_arc_sols = [[(all_sols[i][j], all_sols[i][j+1]) for j in range(num_item-1)] for i in range(num_sol)]
    #for i in range(num_sol):
       #for j in range(num_item-1):
          #all_arc_sols[i].append((all_sols[i][j],all_sols[i][j+1]))

    return all_arc_sols

# all_arc_sols = all_sol_from_list_to_arc(cur_sol)
# print(all_arc_sols)

def cut_arc_sol(arc_sol): #argument เป็น arc_sol ของคำตอบเดียว
    num_item = len(arc_sol)+1
    arc_sol_cut = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol:
            if item == arc[0] or item == arc[1]:
                arc_sol_cut[item].append(arc)
    return arc_sol_cut

# arc_sol_cut = cut_arc_sol(all_arc_sols[0])
# print(arc_sol_cut)

def init_velocity_sol(arc_sol_cut) :
    import random
    num_item = len(arc_sol_cut) #นับจำนวน item จากจำนวนสมาชิกของ list arc_sol_cut
    arc_sol_velocity_dict = [{} for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut[item]:
            arc_sol_velocity_dict[item][arc] = round(random.random(), 4)

    return arc_sol_velocity_dict

# arc_sol_velocity_dict = init_velocity_sol(arc_sol_cut)
# print(arc_sol_velocity_dict)

# Vt+1 = w*Vt + c1r1(pbest-xi) + c2r2(gbest-xi)
# xt+1 = xt+vt+1 Solution Update --> cut_set
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
# coef_times_velocity_dict = coef_times_velocity(0.7,arc_sol_velocity_dict)
# print(coef_times_velocity_dict)

# # arc_first เช่น [[(0,1),(2,0)], [(0,1),(1,22)], ...]
# arc_first = [[(0,2)], [(2,1)], [(0,2),(2,1)]] #0 2 1
# arc_second = [[(0,1)], [(0,1),(1,2)], [(1,2)]] #0 1 2
# #arc_first - arc_second = [[(0,2)], [(2,1)], [(0,2),(2,1)]]
def position_minus_position(arc_first, arc_second):
    num_item = len(arc_first)
    pos_minus_pos = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                pos_minus_pos[item].append(arc)
    return pos_minus_pos
# pos_minus_pos = position_minus_position(arc_first, arc_second)
# print(pos_minus_pos)

#arc_first - arc_second = arc_diff = [[(0,2)], [(2,1)], [(0,2),(2,1)]]
#coef_times_position_dict = [{(0,2):1}, {(2,1):0.4}, {(0,2):0.5, (2,1):0.7}]
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
# coef_times_position_dict = coef_times_position(2,pos_minus_pos)
# print(coef_times_position_dict)

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

# added_v = add_velocity(v_first, v_second)
# print(f'added_velocity is {added_v}')

def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    import copy
    #new_added_velocity_dict = copy.deepcopy(added_velocity_dict)
    new_added_velocity_dict = [{arc:prob for arc,prob in added_velocity_dict[item].keys()} for item in range(num_item)]
    for item in range(num_item):
        for arc_first in added_velocity_dict[item].keys():
            if arc_first in added_velocity_dict[arc_first[0]].items():
               if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[0]][arc_first]:
                   new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[0]][arc_first]
               #else:
               #    new_added_velocity_dict[arc_first[0]][arc_first] = added_velocity_dict[item][arc_first]
            if arc_first in added_velocity_dict[arc_first[1]].items():
                if added_velocity_dict[item][arc_first] < added_velocity_dict[arc_first[1]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[1]][arc_first]
                #else:
                #    new_added_velocity_dict[arc_first[1]][arc_first] = added_velocity_dict[item][arc_first]
    return new_added_velocity_dict

# new_added_v = check_velocity_inconsistency(added_v)
# print(f'new_added_v is {new_added_v}')

def creat_cut_set(added_velocity_dict,alpha):
    num_item = len(added_velocity_dict)
    cut_set = [[] for _ in range(num_item)]
    for item in range(num_item):
        for arc in added_velocity_dict[item].keys():
            if added_velocity_dict[item][arc] >= alpha:
                cut_set[item].append(arc)
    return cut_set
# cut_set = creat_cut_set(new_added_v,0.6)
# print(cut_set)

def select_dest_from_source(source, picked_list, *sets):
    # function ทำหน้าที่ในการเลือก item ที่เราจะเดินเก็บถัดไป (dest) จากตำแหน่งปัจจุบันที่เราอยู่ (source) โดยเราต้องการได้ผลลัพธ์เป็น
    # arc ของ (source,dest) และ picked_list เป็น list ที่เก็บ item ที่เราเดินเก็บไปแล้ว
    import random
    for set in sets:
        new_set = []
        if len(set[source])>0:
            for arc in set[source]:
                if arc[1] not in picked_list and arc[0] == source:
                    new_set.append(arc)
        if len(new_set)>0:
            dest = random.choice(new_set)
            break
    arc_source_dest = (source,dest)
    return dest, arc_source_dest

def sol_position_update(cut_set, previous_x, sub_E_list, alpha, start_previous_x, start_pbest, start_gbest):
    import random

    num_item = len(cut_set)
    picked_list = []
    picked_list_arc = []

    source = random.choice(start_previous_x, start_pbest, start_gbest, random.choice(range(num_item)))
    picked_list.append(source)

    for item_counter in range(num_item-1):
        dest, arc_source_dest = select_dest_from_source(source, picked_list, cut_set, previous_x, sub_E_list)
        source = dest
        picked_list.append(dest)
        picked_list_arc.append(arc_source_dest)

    return picked_list, picked_list_arc


name_path_input = '1I-10-100-2'
df_item_pool = read_input(name_path_input)

#input หลังจาก evaluate_all_sols แล้ว
#input item_in_batch_all_sols =  [[[5, 4, 2, 9, 3, 8], [0, 7, 6, 1]], [[5, 4, 7, 6, 1, 9, 3, 8], [0, 2]], [[5, 4, 7, 6, 1, 9, 3, 8], [0, 2]], [[0, 2, 9, 3, 8], [5, 4, 7, 6, 1]], [[0, 5, 4, 6, 1, 9, 3, 8], [2], [7]]]
#input total_tardiness_all_sols = [0.25, 13.617, 13.617, 0, 11.817]
#input picker_assigment_all_sols = [[[1], [2]], [[1], [2]], [[1], [2]], [[2], [1]], [[1, 2], [3]]]
def mayfly(name_path_input, num_gen, pop_size, *parameters):
    import random
    # แยกตัวแปร parameters เป็นตัวแปรย่อยๆ
    a1, a2, beta, gravity = parameters

    # อ่านข้อมูลจากไฟล์ที่กำหนดและนับจำนวนสิ่งของ (items)
    df_item_pool, df_item_sas_random = read_input(name_path_input)
    num_item = df_item_pool.shape[0]

    # สร้างประชากรแมลงผาย (mayfly) และตั้งค่าเริ่มต้น
    # mayfly_cur_pos: ตำแหน่งปัจจุบันของแมลงผายในประชากร
    # pbest_sols, pbest_tardiness: บันทึกคำตอบและค่าความล่าช้าที่ดีที่สุดในประชากร
    # g_best_tardiness, g_best_items_in_batch: บันทึกค่าความล่าช้าและสิ่งของที่ดีที่สุดในประชากรทั้งหมด
    mayfly_cur_pos = [random.sample(range(num_item), num_item) for _ in range(pop_size)]
    pbest_sols = [[] for _ in range(pop_size)]
    pbest_tardiness = [float('inf')] * pop_size
    g_best_tardiness = float('inf')
    g_best_items_in_batch = []

    # ประเมินคำตอบเริ่มต้นและหา gbest (คำตอบที่ดีที่สุดในประชากร)
    # ทำการวนลูปเพื่อประเมินแต่ละคำตอบของประชากร
    picker_assigment_all_sols = []
    total_tardiness_all_sols = []
    item_in_batch_all_sols = []
    for i, sol in enumerate(mayfly_cur_pos):
        picker_assigment_sol, total_tardiness_sol, item_in_batch_sol = evaluate_all_sols(sol, df_item_pool,
                                                                                         name_path_input)
        picker_assigment_all_sols.append(picker_assigment_sol)
        total_tardiness_all_sols.append(total_tardiness_sol)
        item_in_batch_all_sols.append(item_in_batch_sol)

        if total_tardiness_sol < pbest_tardiness[i]:
            pbest_sols[i] = item_in_batch_sol
            pbest_tardiness[i] = total_tardiness_sol

    min_tardiness = min(total_tardiness_all_sols)
    if min_tardiness <= g_best_tardiness:
        if min_tardiness < g_best_tardiness:
            g_best_items_in_batch.clear()
            g_best_tardiness = min_tardiness
        for i, tardiness in enumerate(total_tardiness_all_sols):
            if tardiness == g_best_tardiness:
                extended_batch = []
                for batch in item_in_batch_all_sols[i]:
                    extended_batch.extend(batch)
                g_best_items_in_batch.append(extended_batch)

    # สร้าง extended_item_in_batch_all_sols
    # สร้างรายการใหม่โดยขยายข้อมูลจากรายการ item_in_batch_all_sols
    extended_item_in_batch_all_sols = []
    for sol in item_in_batch_all_sols:
        extended_sol = []
        for batch in sol:
            extended_sol.extend(batch)
        extended_item_in_batch_all_sols.append(extended_sol)

    # สร้างประชากรแมลงผายตัวผู้และตัวเมีย
    # กำหนดจำนวนแมลงผายตัวผู้และตัวเมียเท่าๆ กันจากประชากร
    num_males = num_females = pop_size // 2
    male_mayfly = []
    female_mayfly = []

    # สุ่มเลือกตัวผู้และตัวเมียจากรายการที่ขยายแล้ว
    # โดยสุ่มเลือกตัวอย่างจากรายการ extended_item_in_batch_all_sols
    for _ in range(num_males):
        male_mayfly.append(random.choice(extended_item_in_batch_all_sols))

    for _ in range(num_females):
        female_mayfly.append(random.choice(extended_item_in_batch_all_sols))



        # วนซ้ำจนกว่าเงื่อนไขหยุดจะถูกพบ
    # while not stopping_criteria_met():
        # อัปเดตความเร็วและคำตอบ
        #xit+1 = xit + vit+1
    new_mayfly_cur_sol = []
     # แปลง solution ของตัวผู้และตัวเมียให้เป็น arc
    male_mayfly_arc = all_sol_from_list_to_arc(male_mayfly)
    female_mayfly_arc = all_sol_from_list_to_arc(female_mayfly)

    # แบ่ง arc ของตัวผู้และตัวเมียให้เป็น cut_set โดยให้แต่ละ arc ผ่านฟังก์ชัน cut_arc_sol
    male_mayfly_arc_sol_cut = [cut_arc_sol(arc) for arc in male_mayfly_arc]
    female_mayfly_arc_sol_cut = [cut_arc_sol(arc) for arc in female_mayfly_arc]

    # สร้างความเร็วของตัวผู้และตัวเมีย
    male_mayfly_velocity_dict = [init_velocity_sol(arc_sol_cut) for arc_sol_cut in male_mayfly_arc_sol_cut]
    female_mayfly_velocity_dict = [init_velocity_sol(arc_sol_cut) for arc_sol_cut in female_mayfly_arc_sol_cut]

    # คูณความเร็วของตัวผู้และตัวเมียด้วยค่าคงที่ gravity * vtij
    male_mayfly_gravity_times_velocity = [coef_times_velocity(gravity, sol) for sol in male_mayfly_velocity_dict]
    female_mayfly_gravity_times_velocity = [coef_times_velocity(gravity, sol) for sol in female_mayfly_velocity_dict]

    # คูณความเร็วของตัวผู้และตัวเมียด้วยค่าคงที่ a1 * (pbest - xi)
        # update_velocities(male_mayflies, female_mayflies)
        # update_positions(male_mayflies, female_mayflies)

        # ประเมินคำตอบใหม่
        # evaluate_solutions(male_mayflies, female_mayflies)

        # จัดอันดับ mayflies
        # rank_mayflies(male_mayflies, female_mayflies)

        # ผสมพันธุ์ mayflies
        # offspring = mate_mayflies(male_mayflies, female_mayflies)

        # ประเมินลูกหลาน
        # evaluate_offspring(offspring)

        # แยกลูกหลานไปยังตัวผู้และตัวเมีย
        # separate_offspring(offspring)

        # แทนที่คำตอบที่แย่ที่สุดด้วยคำตอบใหม่ที่ดีที่สุด
        # replace_worst_solutions(male_mayflies, female_mayflies, offspring)

        # อัปเดต pbest และ gbest
        # update_pbest_gbest(male_mayflies, female_mayflies)

        # ประมวลผลหลังการทำงานและการแสดงผล
        # postprocess_and_visualize(male_mayflies, female_mayflies)
        # pass
    print(xxx)















    # # สร้างคำตอบเริ่มต้นของ Mayfly
    # for sol in range(pop_size):
    #     temp_pos =list(range(num_item))
    #     random.shuffle(temp_pos)
    #     male_mayfly_cur_pos.append(temp_pos)
    # picker_assigment_all_sols = []
    # total_tardiness_all_sols = []
    # item_in_batch_all_sols = []
    #     # หาค่าคำตอบของ male mayfly แต่ละตัวใน population
    # for sol in range(pop_size):
    #     picker_assigment_sol,total_tardiness_sol,item_in_batch_sol = evaluate_all_sols(male_mayfly_cur_pos[sol], df_item_pool, name_path_input)
    #     picker_assigment_all_sols.append(picker_assigment_sol)
    #     total_tardiness_all_sols.append(total_tardiness_sol)
    #     item_in_batch_all_sols.append(item_in_batch_sol)
    # #สร้าง list ของคำตอบใหม่หลังจากซ่อมคำตอบใน evaluate_all_sols ไปแล้ว
    # male_mayfly_cur_pos = [[] for sol in range(pop_size)]
    # for sol in range(pop_size):
    #     for batch in item_in_batch_all_sols[sol]:
    #         male_mayfly_cur_pos[sol].extend(batch)
    # g_best_index = total_tardiness_all_sols.index(min(total_tardiness_all_sols))
    # g_best_index = male_mayfly_cur_pos[g_best_index]
    # g_best_tardiness = min(total_tardiness_all_sols)


mayfly(name_path_input,2,10,2,2,0.5,0.7)

