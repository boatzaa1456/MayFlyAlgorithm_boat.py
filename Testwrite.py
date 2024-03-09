import pandas as pd
import csv
import random
import no_cross as mayfly
import sys

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

# Assuming read_input is already defined as shown previously

name_path_input = '1R-20I-150C-2P'
df_item_pool, df_item_sas_random = read_input(name_path_input)

# Parameter lists
a1 = [0.6]
a2 = [0.3, 0.5, 0.7, 0.9]
a3 = [0.6]
g = [0.6]
alpha = [0.3, 0.5, 0.7, 0.9]
num_gen = [100]
pop_size = [50]
seeds = [1111,2222,3333,4444,5555]

headers = ['a1', 'a2', 'a3', 'g', 'alpha', 'num_gen', 'pop_size', 'seed', 'gbest_per_gen']
results = []

for ng in num_gen:
    for ps in pop_size:
        for x in a1:
            for y in a2:
                for z in a3:
                    for i in g:
                        for a in alpha:
                            for seed in seeds:
                                # Set the random seed for reproducibility
                                random.seed(seed)
                                # Correctly call the function within the no_cross module
                                gbest_per_gen = mayfly.mayfly(name_path_input, ng, ps, x, y, z, i, a, seed)
                                # Append the result along with the parameters and the seed to the results list
                                results.append([x, y, z, i, a, ng, ps, seed, gbest_per_gen])


# Write the results to a CSV file
with open('parameter_test_results_with_replication.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for result in results:
        writer.writerow(result)
