import os
import csv

# 根文件夹和子文件夹列表
base_folder = "REFUGE"
subfolders = ["REFUGE_Test", "REFUGE_Train", "REFUGE_Validation"]

# 遍历每个子文件夹并生成CSV
for subfolder in subfolders:
    csv_filename = f"{subfolder}.csv"
    csv_path = os.path.join(base_folder, csv_filename)

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image', 'mask'])

        image_folder = os.path.join(base_folder, subfolder)
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

        for image_file in image_files:
            image_path = os.path.join(subfolder, image_file)
            mask_file = image_file.replace('.jpg', '.bmp')
            mask_path = os.path.join(subfolder, mask_file)
            csv_writer.writerow([image_path, mask_path])

    print(f"CSV file '{csv_filename}' generated.")

print("CSV generation complete.")
