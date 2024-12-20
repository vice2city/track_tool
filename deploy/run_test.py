import csv
import os
import random
import subprocess
import threading

import mot2coco

def read_stream(stream, prefix):
    """
    读取流并打印每一行。

    Args:
        stream: 要读取的流 (例如 process.stdout 或 process.stderr)。
        prefix: 输出行的前缀 (例如 "STDOUT" 或 "STDERR")。
    """
    for line in stream:
        if 'mmengine - INFO' in line or prefix == 'STDERR':
        # if prefix == 'STDERR':
            print(f"{prefix}: {line.strip()}")

in_folder = '/code/data/datasets/SAT-MTB_ship/'
video_names = os.listdir(os.path.join(in_folder, 'test'))
data_list = random.sample(video_names, 3)

mot2coco.get_annotation(in_folder, '/code/data/datasets/SAT-MTB_ship/annotations', data_list)

command = 'python tools/test_tracking.py custom_configs/bytetrack_train.py --checkpoint /code/data/checkpoints/bytetrack_plane.pth --work-dir /code/data/outputs/bytetrack_plane/test'
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True
)

stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT"))
stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR"))

stdout_thread.start()
stderr_thread.start()

stdout_thread.join()
stderr_thread.join()

return_code = process.wait()
print(f"Process finished with return code: {return_code}")

file_path = '/data/code/outputs/bytetrack_plane/metric_results/default-tracker/pedestrian_detailed.csv'
column_indices = [0, 2, 4, 5, 11, 23, 24, 25, 26]
with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile)  # 创建一个 csv reader 对象
    for row in reader:
        selected_columns = [row[i] for i in column_indices if i < len(row)]
        print(', '.join(selected_columns))  # 将每一行数据用逗号连接并打印
