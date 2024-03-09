from multiprocessing import Pool
import pandas as pd
import csv
import random
import Mayfly_NoCrossOver_22_feb_2024 as mayfly_module
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import sys

# อ่านไฟล์ CSV เพียงครั้งเดียว
def read_input(name_path_input):
    df_item_sas_random = pd.read_csv('df_item_sas_random.csv')
    duedate_path = f'{name_path_input}\\duedate_{name_path_input}.csv'
    input_location_path = f'{name_path_input}\\input_location_item_{name_path_input}.csv'
    df_duedate = pd.read_csv(duedate_path, header=None)
    df_item_oder = pd.read_csv(input_location_path, header=None)
    list_duedate = df_duedate[0].tolist()
    num_order = df_item_oder.shape[1]
    order_items = [df_item_oder[order][df_item_oder[order] != 0] for order in range(num_order)]
    df_item_pools = [
        df_item_sas_random[df_item_sas_random['location'].isin(order_item)].assign(duedate=list_duedate[order], order=order) for order, order_item in
        enumerate(order_items)]
    df_item_pool = pd.concat(df_item_pools, ignore_index=True)
    list_order = [order for order in range(num_order) for _ in range(len(order_items[order]))]
    list_total_item = [item for order_item in order_items for item in order_item.tolist()]
    return df_item_pool, df_item_sas_random

# multiprocessing
def mayfly_wrapper(name_path_input, ng, ps, a1, a2, a3, gmax, gmin, alpha, seed):
    # ฟังก์ชันนี้ควรรับ 10 อาร์กิวเมนต์แยกกัน, ไม่ใช่ tuple
    random.seed(seed)
    return mayfly_module.mayfly(name_path_input, ng, ps, a1, a2, a3, gmax, gmin, alpha, seed)


# ปรับเปลี่ยนฟังก์ชัน measure_initial_iterations ให้ใช้ multiprocessing
def measure_initial_iterations_parallel(args_list):
    with Pool() as pool:
        # Change to pool.starmap to correctly unpack the tuple of arguments
        results = pool.starmap(mayfly_wrapper, args_list)
    # Assuming mayfly_wrapper returns the end time directly
    return results


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

def main():
    name_path_input = '1R-20I-150C-2P'
    df_item_pool, df_item_sas_random = read_input(name_path_input)

    # Parameter lists
    a1 = [0.5, 1, 1.5, 2]
    a2 = [0.5, 1, 1.5, 2]
    a3 = [0.5, 1, 1.5, 2]
    gmax = [0.7, 0.9]
    gmin = [0.3, 0.5]
    alpha = [0.5, 0.7]
    seeds = [3124]
    num_pop = [(100, 50)]
    rep = [1]

    ng, ps = num_pop[0]
    headers = ['a1', 'a2', 'a3', 'gmax', 'gmin', 'alpha', 'num_pop', 'seed', 'gbest_per_gen', 'replication']
    results = []

    # Update the args_list to be a list of tuples for starmap
    args_list = [
        (name_path_input, ng, ps, a1_val, a2_val, a3_val, gmax_val, gmin_val, alpha_val, seed)
        for a1_val in a1
        for a2_val in a2
        for a3_val in a3
        for gmax_val in gmax
        for gmin_val in gmin
        for alpha_val in alpha
        for seed in seeds
    ]

    # Use Pool for parallel processing
    # ใน main function
    with Pool(processes=6) as pool:
        results = pool.starmap(mayfly_wrapper, args_list)

    # Perform initial time measurements with multiprocessing
    measured_times = measure_initial_iterations_parallel(args_list[:1])  # ตัวอย่าง: วัดเวลาสำหรับ 5 iterations แรก
    average_time = calculate_average_time(measured_times)

    # Calculate total number of iterations
    total_iterations = len(args_list)

    # Print estimated completion time
    print_estimated_completion_time(average_time, total_iterations)

    # Processing loop with progress bar
    # Processing loop with progress bar
    with tqdm(total=total_iterations, desc="Processing", file=sys.stdout) as pbar:
        for args in args_list:
            start_time = time.time()
            gbest_each_gen = mayfly_wrapper(*args)
            num_pop_str = f"{args[1]}, {args[2]}"  # ng, ps
            result_row = [args[3], args[4], args[5], args[6], args[7], args[8], num_pop_str, args[9], gbest_each_gen, 1]
            results.append(result_row)
            pbar.update(1)

    # Writing results to a CSV file
    # Writing results to a CSV file
    with open('test_new_srtucture_update_male_first_15_feb_2024.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for result in results:
            # Ensure that result is a list or tuple
            writer.writerow(result)

    print("Process completed.")

# ทำให้โค้ดรันเมื่อสคริปต์นี้เป็นโปรแกรมหลัก
if __name__ == "__main__":
    main()
