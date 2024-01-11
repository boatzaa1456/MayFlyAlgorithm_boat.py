import pandas as pd
import time

value_item_heavy = 100


def batching_A(df_item_poor, list_index_item, *info):

    # จับเวลาการทำงานของกระบวนการ
    start_time = time.time()

    capacity_picker, value_threshold, name_path_input = info

    list_batching_item = []
    list_total_weight_batch = [] # น้ำหนักทั้งหมด พิจารณาสินค้าปัจุบัน
    num_item = len(df_item_poor) #นับจำนวนสินค้าทั้งหมด

    count_batch = 1 # กำหนดหมายเลข Batch
    df_item_record = pd.DataFrame() #สร้าง Data Frame ไว้บันทึกข้อมูลทั้งหมด
    list_item_in_batch = []
    dataframe_new = []
    while len(list_index_item) != 0:
        total_weight_batch = 0
        list_item_new = []

        df_item = pd.DataFrame()
#
        for i in range(0, num_item):
            if i >= len(list_index_item):
                break
            item_current = list_index_item[i]
            data_item_current = df_item_poor.iloc[[list_index_item[i]]]
            weight_item_current = data_item_current['weight'].values
            list_item_new.append(item_current)

            ''' ทำ จนกว่า น้ำหนักรวมสินค้าใน batch >= capacity_picker '''
            total_weight_batch += weight_item_current

            ''' check : capacity condition !!! '''
            if total_weight_batch > capacity_picker:
                list_item_new.remove(item_current)
                break

            ''' Check category condition ของสินค้า 
                ตัวอย่าง :      food = 1
                              nonfood = 0'''
            if weight_item_current >= value_item_heavy:
                ''' *** check : threshold condition !!! 
                        คำถาม : ต้องบวกน้ำหนักตตัวเองไปคิดเลยไหม หรือไม่ ???? '''
                if total_weight_batch > value_threshold:
                    list_item_new.remove(item_current)
                    df_item_poor_drop = df_item[df_item['location'] == item_current].index.values
                    df_item = df_item.drop(df_item_poor_drop)
                    break

            '''สร้าง datafram each batch '''
            df_item = pd.concat([df_item, data_item_current])
            dataframe_new = df_item.sort_values(by=['category', 'self_capacity'], ascending=[True, False])

            list_item_new = dataframe_new.index.values.tolist()
            list_weight_of_self_capacity_current = []
            list_self_capacity = []

            ''' *** check : self capacity condition of each item !!! '''
            for m in range(0, len(dataframe_new)):

                if m == 0:

                    list_self_capacity.append(capacity_picker)
                    weight = dataframe_new['weight'].values[m]
                    total_weight_dataframe_new = weight
                    list_weight_of_self_capacity_current.append(total_weight_dataframe_new)

                else:

                    list_self_capacity.append(dataframe_new['self_capacity'].values[m - 1].sum())
                    total_weight_dataframe_new = dataframe_new['weight'].values[m:len(dataframe_new)].sum()
                    list_weight_of_self_capacity_current.append(total_weight_dataframe_new)

            ''' *** check   : self capacity condition of each item !!! 
                    และ ลบ item ที่ถูกใช้แล้วออกจาก solution space'''
            for m in range(0, len(dataframe_new)):

                if list_weight_of_self_capacity_current[m] > list_self_capacity[m]:
                    list_item_new.remove(item_current)
                    df_item_poor_drop = df_item.index[-1]
                    df_item = df_item.drop(df_item_poor_drop)
                    break

        df_item = df_item.sort_values(by=['category', 'self_capacity'], ascending=[True, False])
        df_item.loc[:, 'batch'] = count_batch
        list_item = []

        for i in range(0, len(df_item)):
            location = df_item['location'].values[i]
            list_item.append(location)

        list_batching_item.append(list_item)
        df_item_record = pd.concat([df_item_record, df_item], ignore_index = True)
        num_columns = len(df_item_record.columns)
        list_zero = [0 for x in range(0, num_columns)]
        df_item_record.loc[len(df_item_record.index)] = list_zero
        total_weight_save = dataframe_new['weight'].values[0:len(list_item_new) + 1].sum()
        list_total_weight_batch.append(total_weight_save)

        list_item_in_batch.append(list_item_new)
        list_index_item = [ele for ele in list_index_item if ele not in list_item_new]

        ''' list item poor = [ ]'''
        if not list_index_item:
            break

        count_batch += 1

    ''' บันทึกข้อมูล Batch '''
    # path_1 = 'output' + '\\Output_' + str(name_path_input) + '_Dataframe_solution.csv'
    # df_item_record.to_csv(path_1, index=False)

    # ''' Check number of item in lists'''
    # count = 0
    # for element in list_batching_item:
    #     count += len(element)
    #
    # count_1 = 0
    # for element in list_item_in_batch:
    #     count_1 += len(element)

    end_time = time.time()
    total_time_run = end_time - start_time
    total_time_run = format(total_time_run, '.3f')
    #print('total time run :', total_time_run, 'seconds')

    return list_batching_item, df_item_record, list_item_in_batch

def batching_2(df_item_poor, list_index_item, *info):

    '''Input : list_index_item = [3, 5, 7, 1, 9, 8, 6, 2, 0, 4]'''

    ''' จับเวลาการทำงานของกระบวนการ '''
    start_time = time.time()


    capacity_picker, value_threshold, name_path_input = info
    ''' ค่ากำหนดสินค้าว่าป็นสินค้าหนักหรือสินค้าเบา (Item ที่มีน้ำหนักมากกว่า 40 เป็นสินค้าหนัก)  '''

    df_item_record = pd.DataFrame()
    list_batching_item, list_item_in_batch = [], []
    num_item = len(df_item_poor)  # นับจำนวนสินค้าทั้งหมด

    for i in range(num_item):
        '''Index สินค้าที่ถูกพิจารณาปัจจุบัน '''
        item_current = list_index_item[i]
        '''ข้อมูลสินค้าปัจจุบัน
        EX :    index    location  item     category  weight    self_capacity     duedate     order
                52       731        314         1       5             19          10.5417     19'''
        data_item_current = df_item_poor.iloc[[list_index_item[i]]]
        ''' น้ำหนักสินค้าปัจจุบัน '''
        weight_item_current = data_item_current['weight'].values
        data_item_current = data_item_current.copy()
        data_item_current.loc[item_current, 'batch'] = 1
        df_item_record = pd.concat([df_item_record, data_item_current]).astype(int)

        while True:

            ''' ตรวจสอบ 'จำนวน Batch' ที่มีอยู่ปัจจุบันทั้งหมด มีกี่ Batch
                        ถ้า สินค้า (i) มากกว่า 1 ไปเช็ค df_item_record แถวสุดท้าย '''
            count_batch = df_item_record['batch'].values[-1]
            list_weight_each_batch, list_batch_remain = [], []

            for j in range(1, int(count_batch) + 1):
                conditions = 0
                '''ตรวจสอบ น้ำหนัก Item รวมของแต่ละ Batch '''
                weight_each_batch = df_item_record.loc[df_item_record['batch'] == j, 'weight'].sum()
                list_weight_each_batch.append(weight_each_batch)

                ''' ใส่ Item ลงไปใน  Batch น้ำหนัก มากเกิน ความสามารถในการหยิบของพนักงานหรือไหม'''
                if weight_each_batch > capacity_picker:
                    df_item_record.loc[item_current, 'batch'] = j+1
                    ''' ละเมิดเงื่อนไข '''
                    conditions += 1
                    continue

                ''' เงื่อนไขที่ 1  ด้านน้ำหนักสินค้าถ้าสินค้าปัจจุบันเป็นสินค้าหนัก(Heavy item) ในปัจจุบัน
                    และน้ำหนักทั้งหมดใน Batch >  ค่าความสามารถในหารรับสินค้าของอุปกรณ์ (Threshold) ให้สินค้าปัจจุบันไปอยู่ Batch ใหม่'''
                if weight_item_current >= value_item_heavy:
                    ''' รวมน้ำหนักของสินค้าหนักใน batch '''
                    df_item_batch_current = df_item_record.loc[df_item_record['batch'] == j]
                    weight_item_heavy = df_item_batch_current.loc[df_item_batch_current['category'] == 0, 'weight'].sum()
                    if weight_item_heavy > value_threshold:
                        df_item_record.loc[item_current, 'batch'] = j + 1
                        ''' ละเมิดเงื่อนไข '''
                        conditions += 1
                        continue

                '''เงื่อนไขที่ 2 ด้านหมวดหมู่สินค้า --> จัดเรียงสินค้าตามหมวดหมู่สินค้า'''
                '''สร้าง df_batch เพื่อเช็คเงื่อนไข batch โดยการคัดลอกข้อมูล item เฉพาะ Batch ที่ j '''
                df_batch = df_item_record[df_item_record['batch'] == j].copy()
                '''จัดเรียงสินค้าใน data frame โดยแบ่งกลุ่มหวดหมวด (category) 0 = Nonfood, 1 = food
                    และแต่ละกลุ่ม เรียงตามความสามารถในการรับน้ำหนักมากไปหาน้ำหนักน้อย'''
                df_batch = df_batch.sort_values(by=['category', 'self_capacity'], ascending=[True, False])
                ''' เงื่อนไขที่ 3 ด้านความเปราะ -->
                    Ex. list self capacity ของแต่ละ Item เช่น '''
                list_self_capacity, list_total_weight_item_before = [], []
                num_i = len(df_batch)
                for item in range(num_i):
                    if item == 0:
                        self_capacity_item_before = capacity_picker
                        list_self_capacity.append(self_capacity_item_before)
                        '''น้ำหนักรวมที่ item[0] จนถึง item[item-num_i] (น้ำหนักรวมจนถึง item ก่อนหน้าที่พิจารณา) '''
                        total_weight_item = df_batch['weight'].values[:num_i - item].sum()
                        list_total_weight_item_before.append(total_weight_item)
                    else:
                        self_capacity_item_before = df_batch['self_capacity'].values[item - 1]
                        list_self_capacity.append(self_capacity_item_before)
                        '''น้ำหนักรวมที่ item[0] จนถึง item[item-num_i] (น้ำหนักรวมจนถึง item ก่อนหน้าที่พิจารณา) '''
                        total_weight_item = df_batch['weight'].values[item:num_i].sum()
                        list_total_weight_item_before.append(total_weight_item)
                    ''' นำ list ที่ถูกบันทึกไว้ มาเปรียบเทียบ
                            Ex. list self capacity ของแต่ละ Item เช่น
                            list_self_capacity = [100, 39, 21, 18, 15, 46, 29]
                            list_total_weight_item_before = [28, 27, 20, 10, 8, 6, 3] '''
                    if list_total_weight_item_before[item] > list_self_capacity[item]:
                        '''ถ้าละเมิดเงื่อนไขทุก Batch และถ้า batch นั้นเป็น batch สุดท้าย  ให้ Item นั้น อยู่ใน batch เปิดใหม่'''
                        df_item_record.loc[item_current, 'batch'] = j + 1
                        conditions += 1
                        continue

            ''' ถ้าไม่มีละเมิดเงื่อนไข ทำการพิจารณา item ถัดไป '''
            if conditions == 0:
                break

    df_item_record = df_item_record.sort_values(by=['batch', 'category', 'self_capacity'], ascending=[True, False, True])

    # name_file = 'Dataframe_solution_' + str(num_population)
    # path_1 = name_path_input + '\\' + name_file + '.csv'
    # df_item_record.to_csv(path_1, index=False)
    # pd.set_option('display.max_row', None)

    end_time = time.time()
    total_time_run = end_time - start_time
    total_time_run = format(total_time_run, '.3f')
    #print(f'total_time_run = {total_time_run}')

    '''แปลงข้อมูลใน Data frame ลง list'''
    num_batch = df_item_record['batch'].iloc[-1]
    for run in range(1, int(num_batch)+1):
        marks_list = df_item_record[df_item_record['batch'] == run]
        list_batching_item.append(marks_list['location'].values.tolist())
        list_item_in_batch.append(marks_list.index.tolist())

    ''' Output Function
    list_batching_item = [[242, 776, 870, 534, 526, 109, 806, 895], [871, 122]]
    list_item_in_batch = [[8, 3, 9, 1, 6, 7, 4, 5], [2, 0]]'''
    return list_batching_item, df_item_record, list_item_in_batch

def end ():

    return
