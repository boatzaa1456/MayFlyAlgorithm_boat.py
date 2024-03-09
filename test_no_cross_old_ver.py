import pandas as pd
import csv
import random
import Mayfly_NoCrossOver_22_feb_2024 as mayfly
from tqdm import tqdm
import time
from datetime import datetime, timedelta
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


# ในฟังก์ชัน measure_initial_iterations, ปรับเปลี่ยนการเรียกใช้ mayfly:
def measure_initial_iterations(iterations_to_measure):
    times = []
    for _ in range(iterations_to_measure):
        start_time = time.time()
        random.seed(seeds[0])
        # อัปเดตการเรียกใช้ mayfly ให้ตรงกับการเปลี่ยนแปลง
        mayfly.mayfly(name_path_input, ng, ps, a1[0], a2[0], a3[0], gmax[0], gmin[0], alpha[0],seeds[0])
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


name_path_input = '1R-20I-150C-2P'
df_item_pool, df_item_sas_random = read_input(name_path_input)

# Parameter lists
a1 = [0.5,1,1.5,2]
a2 = [0.5,1,1.5,2]
a3 = [0.5,1,1.5,2]
gmax = [0.7,0.9]
gmin = [0.3,0.5]
alpha = [0.5,0.7]
seeds = [3124]
num_pop = [(100, 50)]
rep = [1]


ng, ps = num_pop[0]
# อัปเดตหัวข้อสำหรับ CSV
headers = ['a1', 'a2', 'a3', 'gmax', 'gmin', 'alpha', 'num_pop', 'seed', 'gbest_per_gen', 'replication']
results = []

# Perform initial time measurements
iterations_to_measure = 5
measured_times = measure_initial_iterations(iterations_to_measure)
average_time = calculate_average_time(measured_times)

# Calculate total number of iterations
total_iterations = len(a1) * len(a2) * len(a3) * len(gmax) * len(gmin) * len(alpha) * len(num_pop) * len(seeds) * len(rep)

# Print estimated completion time
print_estimated_completion_time(average_time, total_iterations)

# Processing loop with progress bar
with tqdm(total=total_iterations, desc="Processing", file=sys.stdout) as pbar:
    for n_p in num_pop:
        ng, ps = n_p
        for r in rep:
            for seed in seeds:
                random.seed(seed)
                for x in a1:
                    for y in a2:
                        for z in a3:
                            for i in gmax:
                                for m in gmin:
                                    for a in alpha:
                                        start_time = time.time()
                                        gbest_each_gen = mayfly.mayfly(name_path_input, ng, ps, x, y, z, i, m, a, seed)
                                        num_pop_str = (ng, ps)
                                        results.append([x, y, z, i, m, a, num_pop_str, seed, gbest_each_gen, r])
                                        pbar.update(1)

# Writing results to a CSV file
with open('test_new_srtucture_update_male_first_15_feb_2024.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for result in results:
        writer.writerow(result)

print("Process completed.")