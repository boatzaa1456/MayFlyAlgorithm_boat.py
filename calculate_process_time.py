import pandas as pd
import numpy as np
import math
#
def calculate_completion_time(list_distance_batch, number_item_each_batch, *info):

    ''' คำนวณเวลาเสร็จสิ้น 
    setup_time = 3                      # min per batch
    picking_and_searching_time = 20     # sec per item
    speed_picker = 20                   # LU per min 
    ตัวอย่างเช่น  
        batch 1 มี 11 item, ใช้ระยะทางเดิน 616 LU 
        = 3 [min] + {20 [sec/item] x 11 [item]}  + {(1/20) [min/LU] x 616 [LU]} 
        = 3 [min] +  3.67 [(20/60)*11 min] + 20.8 [min]
        = 37.47 min'''
    
    setup_time, picking_and_searching_time, speed_picker, sorting_time = info
    processing_time = setup_time + ((picking_and_searching_time/60)*number_item_each_batch)+(list_distance_batch/speed_picker) + (3*number_item_each_batch)

    return processing_time

# def calculate_due_date(df_item_poor,num_order):
#
#     tardiness_each_order = []
#     for i in range(0, num_order):
#         df_order = df_item_poor[df_item_poor['order'] == i]
#         max_completion_time = df_order['completion_time'].max()
#         max_duedate = df_order['duedate'].max()
#         max_completion_time = float(max_completion_time)
#         max_duedate = float(max_duedate)
#         time_different = max_completion_time - max_duedate
#         time = round(time_different, 3)
#         if time_different < 0:
#             time = 0
#         else:
#             time = time
#         tardiness_each_order.append(time)
#         for j in df_order:
#             df_item_poor.loc[df_item_poor['order'] == i, 'tardiness'] = time
#
#     return tardiness_each_order

# def calculate_due_date_sequencing_batch(df_item_poor, num_order, num_picker, completion_time_batch, list_sequencing_batch):

    # calculate completion time of each batching :
    # Function [calculate_due_date_sequencing_batch] from file calulate_due_date.py
    #     Example :     Number of Picker = 2, list_picker = [[7, 1, 5, 6], [8, 2, 0, 3, 4]]
    #                    ['29.850', '29.933', '23.667', '13.967', '18.100', '16.433', '84.400', '14.883', '5.933']
    #
    #                   Picker No. 1 = b7=(14.883), b1=(29.933), b5=(16.433), b6=(84.400)
    #                   Picker No. 2 = b8=(5.933), b2=(23.667), b0=(29.850), b3=(13.967), b4=(18.100)
    #
    #                   Picker No. 1 = [14.883, 44.816, 61.249, 145.649]
    #                   Picker No. 2 = [5.933, 29.6, 59.45, 73.417, 91.517]

    list_process_time_picker = []
    for i in range(0, num_picker):
        k = 0
        time_picker = 0
        process_time_picker = []
        for j in list_sequencing_batch[i]:
            completion_time = float(completion_time_batch[j])
            completion_time = round(completion_time, 3)
            time_picker += completion_time
            process_time_picker.append(round(time_picker, 3))
            k += 1
        list_process_time_picker.append(process_time_picker)

    return list_process_time_picker




#if __name__ == '__main__':
#     ''' Setting parameter '''
#     setup_time = 3                      # min per batch
#     picking_and_searching_time = 20     # sec per item
#     speed_picker = 20                   # LU per min
#
#
#     list_distance_batch = [616.0, 382.0, 493.0, 548.0, 250.0, 222.0, 154.0, 118.0, 125.0, 106.0]
#     number_item_each_batch = [11, 6, 9, 9, 4, 3, 2, 2, 2, 1]
#     # list_batching_item = [[174, 744, 341, 265, 827, 600, 868, 800, 646, 153, 447],
#     #                       [57, 846, 428, 899, 64, 156],
#     #                       [369, 238, 212, 548, 667, 364, 214, 616, 285],
#     #                       [610, 496, 133, 487, 635, 141, 230, 322, 384],
#     #                       [770, 148, 41, 170], [776, 638, 380], [586, 853], [165, 278], [550, 606], [280]]
#     list_batching_item = [174, 744, 341, 265, 827, 600, 868, 800, 646, 153, 447]
#     ''' คำนวณเวลาเสร็จสิ้น '''
#     df_item_poor = []
#     path_1 = 'D:\\AB_2\\2l-20-45-0\\item_poor.csv'
#     df_item_poor = pd.read_csv(path_1)
#     #caculate_completion_time(list_distance_batch, number_item_each_batch)
#     #df_item_sas_random[df_item_sas_random['location'].isin(item)]
#     batch = df_item_poor[df_item_poor['location'].isin(list_batching_item)]
#     for i in range(0, len(list_distance_batch)):
#         completion_time = calculate_completion_time(list_distance_batch[i], number_item_each_batch[i], setup_time, picking_and_searching_time, speed_picker)
#         print(f'completion time each batch [{i}] = {completion_time}')




