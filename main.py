import tarfile
import os

tar_file = "images_dataSAT.tar"   # file is in parent/current directory
extract_path = "."                # same directory

if not os.path.exists(tar_file):
    raise FileNotFoundError("Dataset tar file not found")

with tarfile.open(tar_file, "r") as tar:
    tar.extractall(path=extract_path)

print("Dataset extracted successfully")

# Define directories
extract_dir = "."

base_dir = os.path.join(extract_dir, 'images_dataSAT')
dir_non_agri = os.path.join(base_dir, 'class_0_non_agri')
dir_agri = os.path.join(base_dir, 'class_1_agri')

agri_images_paths = []
for image in os.listdir(dir_agri):
    agri_images_paths.append(os.path.join(dir_agri, image))

agri_images_paths.sort()