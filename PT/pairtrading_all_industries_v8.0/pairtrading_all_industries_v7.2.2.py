import zipfile
zip_path = "v7.2_净值数据_20260416_155432.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    print("ZIP文件内容：", z.namelist())