import zipfile
import os

zip_folder_path = os.getcwd() + "\\data"
extract_folder_path = os.getcwd() + "\\extracted"

# 确保目标解压文件夹存在
os.makedirs(extract_folder_path, exist_ok=True)

# 获取所有 zip 文件的列表
zip_files = [f for f in os.listdir(zip_folder_path) if f.endswith('.zip')]

# 遍历所有 zip 文件并解压到根目录
for zip_file in zip_files:
    zip_file_path = os.path.join(zip_folder_path, zip_file)

    # 解压 zip 文件，并将文件直接放置到根目录下
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # 获取原始文件名并去掉路径信息
            filename = os.path.basename(member)
            if not filename:
                continue

            # 构造目标文件路径
            target_path = os.path.join(extract_folder_path, filename)

            # 打开目标文件并写入解压内容
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    print(f"解压完成: {zip_file} 到 {extract_folder_path}")
