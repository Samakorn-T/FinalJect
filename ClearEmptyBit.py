import os
import pandas as pd

# ตั้งค่า input และ output
input_folder = 'Prepare_datavisual'
input_file = 'Expanded_Fingerprints_Data.xlsx'
output_file = 'Filtered_Fingerprints_Data.xlsx'
input_path = os.path.join(input_folder, input_file)
output_path = os.path.join(input_folder, output_file)

# ตรวจสอบว่ามีไฟล์ input หรือไม่
if os.path.exists(input_path):
    # โหลดไฟล์ Excel
    df = pd.read_excel(input_path)
    
    # เลือกคอลัมน์ที่ขึ้นต้นด้วย "FP_"
    fp_columns = [col for col in df.columns if col.startswith("FP_")]
    
    # ตรวจสอบว่าคอลัมน์ไหนมีค่าเป็น 0 ทั้งหมด และลบออก
    columns_to_keep = [col for col in fp_columns if not (df[col] == 0).all()]
    df_filtered = df.drop(columns=[col for col in fp_columns if col not in columns_to_keep])
    
    # เซฟไฟล์ Excel ใหม่
    df_filtered.to_excel(output_path, index=False)
    print(f"File Saved to: {output_path}")
else:
    raise FileNotFoundError(f"ไThe file {input_path} does not exist.")