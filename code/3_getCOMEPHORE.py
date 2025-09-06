import os
import tarfile
import rasterio
from rasterio.merge import merge
import numpy as np
from glob import glob

input_dir = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\comephore\downloadData"
output_dir = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace\comephore2\comephoreOutput"
os.makedirs(output_dir, exist_ok=True)


for tar_path in glob(os.path.join(input_dir, "*.tar")):
    print(f"Processing: {tar_path}")
    # unzip
    with tarfile.open(tar_path) as tar:
        tar.extractall(output_dir)

for year_folder in glob(os.path.join(output_dir, "*")):
    if not os.path.isdir(year_folder):
        continue
    year = os.path.basename(year_folder)
    rr_files = sorted(glob(os.path.join(year_folder, "*_RR.gtif")))

    if not rr_files:
        continue

    with rasterio.open(rr_files[0]) as src0:
        meta = src0.meta.copy()
        meta.update(count=len(rr_files))

        out_path = os.path.join(output_dir, f"{year}_summer_RR.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            for idx, rr_file in enumerate(rr_files, start=1):
                with rasterio.open(rr_file) as src:
                    dst.write(src.read(1), idx)
    print(f"Saved {out_path}")
