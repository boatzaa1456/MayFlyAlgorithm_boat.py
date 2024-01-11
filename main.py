import pandas as pd
import random

# โหลดข้อมูลจากไฟล์ CSV ที่มีข้อมูลการสั่งซื้อ (Order) และกำหนดชื่อคอลัมน์ให้แต่ละ Order
df_Order = pd.read_csv('input_location_item_2l-20-45-0.csv', header=None)
df_Order.columns = [f'Order {i}' for i in range(df_Order.shape[1])]

# เปลี่ยนรูปแบบ DataFrame จากแนวนอนเป็นแนวตั้ง และกรองเฉพาะข้อมูลที่ไม่เป็นศูนย์
transformed_df = df_Order.melt(var_name='Order', value_name='item').query('item != 0')
transformed_df['Order'] = transformed_df['Order'].str.extract('(\d+)').astype(int)

# โหลดข้อมูลสินค้าและวันที่ส่งมอบจากไฟล์ CSV
df_item_sas_random = pd.read_csv('df_item_sas_random.csv')
df_duedate = pd.read_csv('duedate_2l-20-45-0.csv', header=None)
df_duedate['Order'] = range(1, len(df_duedate) + 1)
df_duedate.columns = ['DueDate', 'Order']

# รวมข้อมูลที่โหลดมาเข้าด้วยกันเพื่อให้ได้ DataFrame ที่มีข้อมูลครบถ้วน
merged_df = (transformed_df.merge(df_item_sas_random, on='item', how='left')
                           .merge(df_duedate, on='Order', how='left'))

# ปรับเรียงลำดับคอลัมน์ใหม่ โดยย้ายคอลัมน์ 'Order' ไปอยู่ท้ายสุด
column_order = [col for col in merged_df.columns if col != 'Order'] + ['Order']
df_Order_items_Duedate = merged_df[column_order]

# แสดงตัวอย่างข้อมูลหลังจากประมวลผลเรียบร้อย
print(df_Order_items_Duedate.head())

# คัดดัชนีของแถวที่มี 'item' ใน final_merged_df
# คัดลอกดัชนีของแถวที่มีข้อมูลสินค้าใน DataFrame
item_index = df_Order_items_Duedate.index[df_Order_items_Duedate['item'].notnull()].tolist()

# ทำการสุ่มเรียงลำดับดัชนีและเก็บไว้ในลิสต์ col_sol ทั้งหมด 20 ครั้ง
col_sol_1 = []
for _ in range(10):
    shuffled_indices = item_index.copy()
    random.shuffle(shuffled_indices)
    col_sol_1.append(shuffled_indices)

col_sol_2 = []
for _ in range(10):
    shuffled_indices = item_index.copy()
    random.shuffle(shuffled_indices)
    col_sol_2.append(shuffled_indices)
# แสดงลิสต์ของดัชนีที่ถูกสุ่ม
print(col_sol_1)
print(col_sol_2)