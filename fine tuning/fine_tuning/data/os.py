import os
import csv
import re

root = "/Users/1zero8/Desktop/fine_tuning/data/5m"   # <-- change this
output_csv = "output1.csv"

RE_DEPTH = re.compile(r'Average Depth:\s*([0-9.]+)')
RE_BOX   = re.compile(r'Bounding Box Area:\s*([0-9]+)')

rows = []

for folder, subdirs, files in os.walk(root):
    folder_name = os.path.basename(folder)
    folder_name = float(folder_name.replace("m", "").strip())*1000
    for file in files:
        if file.endswith(".txt"):

            txt_path = os.path.join(folder, file)
            filename = os.path.splitext(file)[0]

            with open(txt_path, "r") as f:
                content = f.read()

            # Extract info
            depth_match = RE_DEPTH.search(content)
            box_match = RE_BOX.search(content)

            if depth_match and box_match:
                depth_mm = float(depth_match.group(1))
                box_area = int(box_match.group(1))

                # folder_name is placed LAST
                rows.append([filename, box_area, int(depth_mm), folder_name])

# Write CSV
with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "box_area", "cam_depth", "true_depth"])
    writer.writerows(rows)

print(f"✅ Done! Extracted {len(rows)} entries into {output_csv}")
