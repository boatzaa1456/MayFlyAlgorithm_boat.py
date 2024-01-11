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

# ------------------------------------------------------------------------------------------------------------------------
#Mayfly Algorithm
# Function Initialize the male mayfly population xi
def initialize_mayfly_population(num_sol, num_item):
    import random
    mayfly_population = []
    for sol in range(num_sol):
        mayfly = list(range(num_item))
        random.shuffle(mayfly)
        mayfly_population.append(mayfly)
    return mayfly_population

def evaluate_mayfly_population(num_sol, mayfly_population, df_item_pool, name_path_input):
    picker_list = []
    tardiness = []
    batch_list = []

    for sol in range(num_sol):
        a, b, c = evaluate_all_sols(mayfly_population[sol], df_item_pool, name_path_input)
        picker_list.append(a)
        tardiness.append(b)
        batch_list.append(c)

    return picker_list, tardiness, batch_list


# ก่อนปรับปรุงเป็น list comprehension
# def sol_from_list_to_arc(sol):
#     num_item = len(sol)
#     arc_sol = []
#     for i in range(num_item - 1):
#         arc_sol.append((sol[i], sol[i + 1]))
#     return arc_sol

def sol_to_arc_for_gbest(sol):
    return [(sol[i], sol[i + 1]) for i in range(len(sol) - 1)]

def all_sols_from_list_to_arc(sol_male, sol_female, pbest_male, pbest_female):
    def sol_to_arc(sol):
        return [(sol[i], sol[i + 1]) for i in range(len(sol) - 1)]

    arc_sol_male = [sol_to_arc(sol) for sol in sol_male]
    arc_sol_female = [sol_to_arc(sol) for sol in sol_female]
    arc_pbest_male = [sol_to_arc(pbest_male[i]) for i in range(len(pbest_male))]
    arc_pbest_female = [sol_to_arc(pbest_female[i]) for i in range(len(pbest_female))]

    return arc_sol_male, arc_sol_female, arc_pbest_male, arc_pbest_female

def cut_all_arc_sols(arc_sols_male, arc_sols_female):
    def cut_arc_sol(arc_sol):
        arc_sol_dict = {}
        for arc in arc_sol:
            if arc[0] not in arc_sol_dict:
                arc_sol_dict[arc[0]] = set()
            if arc[1] not in arc_sol_dict:
                arc_sol_dict[arc[1]] = set()
            arc_sol_dict[arc[0]].add(arc)
            arc_sol_dict[arc[1]].add(arc)

        return [arc_sol_dict.get(item, set()) for item in range(max(arc_sol_dict.keys()) + 1)]

    cut_arc_sol_male = [cut_arc_sol(arc_sol) for arc_sol in arc_sols_male]
    cut_arc_sol_female = [cut_arc_sol(arc_sol) for arc_sol in arc_sols_female]

    return cut_arc_sol_male, cut_arc_sol_female

def init_all_velocity_sols(cut_arc_sols_male, cut_arc_sols_female):
    def init_velocity_sol(arc_sol_cut):
        num_item = len(arc_sol_cut)
        arc_sol_velocity_dict = [{} for _ in range(num_item)]
        for item in range(num_item):
            for arc in arc_sol_cut[item]:
                arc_sol_velocity_dict[item][arc] = round(random.random(), 4)
        return arc_sol_velocity_dict

    arc_sol_male_velocity_dict = [init_velocity_sol(arc_sol) for arc_sol in cut_arc_sols_male]
    arc_sol_female_velocity_dict = [init_velocity_sol(arc_sol) for arc_sol in cut_arc_sols_female]

    return arc_sol_male_velocity_dict, arc_sol_female_velocity_dict

def coef_time_volocity(coef, arc_sol_velocity_lists):
    coef_time_volocity_lists = []
    for velocity_list in arc_sol_velocity_lists:
        coef_time_list = []
        for velocity_dict in velocity_list:
            coef_time_dict = {}
            if isinstance(velocity_dict, dict):
                for arc, velocity in velocity_dict.items():
                    adjusted_velocity = round(coef * velocity, 4)
                    coef_time_dict[arc] = min(adjusted_velocity, 1)
            coef_time_list.append(coef_time_dict)
        coef_time_volocity_lists.append(coef_time_list)
    return coef_time_volocity_lists

def position_minus_position(arc_first,arc_second):
    num_item = len(arc_first)
    pos_minus_pos = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_first[item]:
            if arc not in arc_second[item]:
                pos_minus_pos[item].append(arc)
    return pos_minus_pos


def coef_time_position_MrGumNhud(c_value, arc_diff):
    import random
    num_item = len(arc_diff)
    coef_time_position_dict = [{} for item in range(num_item)]
    for item in range (num_item):
        for arc in arc_diff[item]:
            coef = c_value*random.random()
            if coef > 1 :
                coef = 1
            coef_time_position_dict[item][arc] = round(coef,4)
    return coef_time_position_dict


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


def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    import copy
    # new_added_velocity_dict = copy.deepcopy(added_velocity_dict)
    new_added_velocity_dict = [{arc: prob for arc, prob in added_velocity_dict[item].items()} for item in range(num_item)]
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


#find global best solution
def process_mayfly_data(tardiness_male, tardiness_female, batch_list_male, batch_list_female, pbest_male=None, pbest_female=None):
    def flatten(nested_list):
        return [item for sublist in nested_list for item in sublist]

    # ตรวจสอบและกำหนดค่าเริ่มต้นสำหรับ pbest_male และ pbest_female
    if pbest_male is None:
        pbest_male = list(range(len(batch_list_male)))
    if pbest_female is None:
        pbest_female = list(range(len(batch_list_female)))

    # ค้นหา Gbest
    combined_tardiness = [(value, 'male', i) for i, value in enumerate(tardiness_male)] + [(value, 'female', i) for i, value in enumerate(tardiness_female)]
    min_tardiness, origin, index = min(combined_tardiness, key=lambda x: x[0])
    Gbest_batch = batch_list_male[index] if origin == 'male' else batch_list_female[index]
    cur_sol_mayfly_gbest = flatten(Gbest_batch)

    # อัปเดต pbest
    for i, tardiness in enumerate(tardiness_male):
        if tardiness < tardiness_male[pbest_male[i]]:
            pbest_male[i] = i
    for i, tardiness in enumerate(tardiness_female):
        if tardiness < tardiness_female[pbest_female[i]]:
            pbest_female[i] = i

    # แปลง batch_list_male และ batch_list_female
    cur_sol_male_mayfly = [flatten(batch) for batch in batch_list_male]
    cur_sol_female_mayfly = [flatten(batch) for batch in batch_list_female]

    # แปลง pbest_male และ pbest_female เป็น batch ที่แท้จริง
    pbest_male_batches = [flatten(batch_list_male[i]) for i in pbest_male]
    pbest_female_batches = [flatten(batch_list_female[i]) for i in pbest_female]

    return cur_sol_mayfly_gbest, cur_sol_male_mayfly, cur_sol_female_mayfly, pbest_male_batches, pbest_female_batches

#ฟังก์ชั่น (gbest - xi)
def gbest_minus_position(arc_gbest,arc_sol):
    num_item = len(arc_sol)
    gbest_minus_xi = [[] for item in range(num_item)]
    for item in range(num_item):
        for arc in arc_gbest:
            if arc not in arc_sol[item]:
                gbest_minus_xi[item].append(arc)
    return gbest_minus_xi


print('---------'*30)

num_item = df_item_pool.shape[0]
num_sol = 5

#initialize the mayfly population
male_mayfly_population = initialize_mayfly_population(num_sol,num_item)
female_mayfly_population = initialize_mayfly_population(num_sol,num_item)

#evaluate the male and female mayfly population
picker_list_from_male_mayfly, tardiness_from_male_mayfly, batch_list_from_male_mayfly = evaluate_mayfly_population(num_sol, male_mayfly_population, df_item_pool, name_path_input)
picker_list_from_female_mayfly, tardiness_from_female_mayfly, batch_list_from_female_mayfly = evaluate_mayfly_population(num_sol, female_mayfly_population, df_item_pool, name_path_input)

#เตรียมข้อมูลสำหรับการปรับปรุง
cur_sol_mayfly_gbest, cur_sol_male_mayfly, cur_sol_female_mayfly, pbest_male, pbest_female = process_mayfly_data(tardiness_from_male_mayfly, tardiness_from_female_mayfly, batch_list_from_male_mayfly, batch_list_from_female_mayfly)
arc_gbest_mayfly = sol_to_arc_for_gbest(cur_sol_mayfly_gbest)
arc_sol_male_mayfly, arc_sol_female_mayfly, arc_pbest_male_mayfly, arc_pbest_female_mayfly = all_sols_from_list_to_arc(cur_sol_male_mayfly, cur_sol_female_mayfly, pbest_male, pbest_female)
cut_arc_sol_male_mayfly, cut_arc_sol_female_mayfly = cut_all_arc_sols(arc_sol_male_mayfly, arc_sol_female_mayfly)
arc_sol_male_mayfly_velocity_dict, arc_sol_female_mayfly_velocity_dict = init_all_velocity_sols(cut_arc_sol_male_mayfly, cut_arc_sol_female_mayfly)

#male mayfly
coef_time_male_mayfly_volocity = coef_time_volocity(0.5,arc_sol_male_mayfly_velocity_dict)
male_mayfly_pbest_minus_xi = position_minus_position(arc_pbest_male_mayfly,arc_sol_male_mayfly)
male_mayfly_gbest_minus_xi = gbest_minus_position(arc_gbest_mayfly,arc_sol_male_mayfly)
added_coef_to_male_mayfly_form_pbest_diff = coef_time_position_MrGumNhud(0.7,male_mayfly_pbest_minus_xi)
added_coef_to_male_mayfly_form_gbest_diff = coef_time_position_MrGumNhud(0.7,male_mayfly_pbest_minus_xi)

print(f'added_coef_to_male_mayfly_form_pbest_diff = {added_coef_to_male_mayfly_form_pbest_diff}')
print(f'added_coef_to_male_mayfly_form_gbest_diff = {added_coef_to_male_mayfly_form_gbest_diff}')



#female mayfly
coef_time_female_mayfly_volocity = coef_time_volocity(0.5,arc_sol_female_mayfly_velocity_dict)








