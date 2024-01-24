import random
import numpy as np
import matplotlib.pyplot as plt
from evaluate_all_sols import *
import itertools
import pandas as pd
import random
random.seed(3124)
def read_input(name_path_input):
    df_item_sas_random = pd.read_csv('df_item_sas_random.csv')
    # print(df_item_sas_random)

    name_path_input = '1R-20I-150C-2P'

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
        df_item_in_this_order = df_item_sas_random[df_item_sas_random['location'].isin(item)].copy()
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

# added_v = add_velocity(v_first, v_second)
# print(f'added_velocity is {added_v}')

def check_velocity_inconsistency(added_velocity_dict):
    num_item = len(added_velocity_dict)
    import copy
    #new_added_velocity_dict = copy.deepcopy(added_velocity_dict)
    new_added_velocity_dict = [{arc:prob for arc,prob in added_velocity_dict[item].items()}for item in range(num_item)]
    for item in range(num_item):
        for arc_first in added_velocity_dict[item].keys():
            if arc_first in added_velocity_dict[arc_first[0]].keys():
                if added_velocity_dict[item][arc_first]< added_velocity_dict[arc_first[0]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[0]][arc_first]
                # else:
                #     new_added_velocity_dict[arc_first[0]][arc_first] = added_velocity_dict[item][arc_first]
            if arc_first in added_velocity_dict[arc_first[1]].keys():
                if added_velocity_dict[item][arc_first]< added_velocity_dict[arc_first[1]][arc_first]:
                    new_added_velocity_dict[item][arc_first] = added_velocity_dict[arc_first[1]][arc_first]
    #             else:
    #                 new_added_velocity_dict[arc_first[1]][arc_first] = added_velocity_dict[item][arc_first]
    return new_added_velocity_dict

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

def cxPartialyMatched(ind1, ind2):
    """Executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place. This crossover expects
    :term:`sequence` individuals of indices, the result for any other type of
    individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values in a certain range of the two parents and swapping the values
    of those indexes. For more details see [Goldberg1985]_.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2
def compare_and_replace(parents, offspring):
    new_solutions = []
    for parent, child in zip(parents, offspring):
        # เปรียบเทียบค่า tardiness
        if child[1] <= parent[1]:
            new_solutions.append(child)
        else:
            new_solutions.append(parent)
    return new_solutions

