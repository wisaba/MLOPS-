import os
from PIL import Image

data_dir = r"E:\MlOps\Practices\Assignement1\kagglecatsanddogs_5340\PetImages"
folders = ['Cat', 'Dog']

removed_count = 0

for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            try:
                # تلاش برای باز کردن و بررسی سلامت عکس
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                # اگر عکس خراب بود، آن را پاک کن
                print(f"فایل خراب پیدا و حذف شد: {file_path}")
                os.remove(file_path)
                removed_count += 1

print(f"پاکسازی تمام شد! {removed_count} عکس خراب حذف شدند.")
