import pandas as pd
import random

df_item_sas_random = pd.read_csv('df_item_sas_random.csv')
df_duedate = pd.read_csv('duedate_2l-20-45-0.csv',header=None)
df_item_order = pd.read_csv('input_location_item_2l-20-45-0.csv',header=None)

list_duedate = df_duedate[0].tolist()

num_order = df_item_order.shape[1]
list_order = []  #ทำหน้าที่การเก็บเลข order ของแต่ละ item
list_total_item = []  #ทำหน้าที่ในการเก็บเลข item ตามลำดับ order

df_item_pool = pd.DataFrame()

for order in range(num_order):
    item = df_item_order[order][df_item_order[order]!=0]
    df_item_in_this_order = df_item_sas_random[df_item_sas_random['location'].isin(item)]
    df_item_in_this_order['duedate'] = list_duedate[order]
    df_item_pool = pd.concat([df_item_pool,df_item_in_this_order])
    num_item_this_order = df_item_in_this_order.shape[0]
    for i in range(num_item_this_order):
        list_order.append(order)
    list_total_item.extend(item.tolist())
df_item_pool['order'] = list_order  #สร้างคอลัมน์ใหม่ขึ้นมาชื่อว่า order

#reset_index ใหม่
df_item_pool.reset_index(drop=True,inplace=True)  #inplace=False หมายถึงไม่แทนที่ #inplace=True หมายถึงแทนที่

num_item = df_item_pool.shape[0]
num_sol = 20 #จำนวนคำตอบในแต่ละรอบ
cur_sol = []
#วิธีพี่app
for sol in range(num_sol):
    now_sol = list(range(num_item))
    random.shuffle(now_sol)
    cur_sol.append(now_sol)

#new* แปลงจาก list ให้กลายเป็น arc ต้องรู้ว่ามีทั้งหมดกี่ตัว
def sol_from_list_to_arc(sol):
    num_item = len(sol)
    arc_sol = []
    for i in range(num_item-1):
        arc_sol.append((sol[i],sol[i+1]))
    return

arc_sol = sol_from_list_to_arc(cur_sol[0])

def all_sols_from_list_to_arc(all_sols):
    num_sol = len(all_sols)  #เก็บจำนวนคำตอบของ all_sol
    num_item = len(all_sols[0])

    all_arc_sols = [[(all_sols[i][j],all_sols[i][j+1]) for j in range(num_item-1)] for i in range(num_sol)]

    return all_arc_sols
all_arc_sols = all_sols_from_list_to_arc(cur_sol)


def cut_arc_sol(arc_sol):
    num_item = len(arc_sol) + 1
    arc_sol_dict = [[] for _ in range(num_item)]

    for arc in arc_sol:
        arc_sol_dict[arc[0]].append(arc)
        arc_sol_dict[arc[1]].append(arc)

    arc_sol_cut = [arc_sol_dict[item] for item in range(num_item)]

    return arc_sol_cut

arc_sol_cut = cut_arc_sol(all_arc_sols[0])
print(f'arc_sol_cut = {arc_sol_cut}')

def init_velocity_sol(arc_sol_cut):
    num_item = len(arc_sol_cut) #นับจำนวน item จากจำนวนสมาชิกของ list arc_sol_cut
    arc_sol_velocity_dict = [{} for _ in range(num_item)]
    for item in range(num_item):
        for arc in arc_sol_cut [item]:
            arc_sol_velocity_dict[item][arc] = round(random.random(),4)
    return arc_sol_velocity_dict

arc_sol_velocity_dict = init_velocity_sol(arc_sol_cut)
print(f'arc_sol_velocity_cut = {arc_sol_velocity_dict}')


