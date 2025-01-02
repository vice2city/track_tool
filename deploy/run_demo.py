import subprocess
import threading

import video_combine


def read_stream(stream, prefix):
    """
    读取流并打印每一行。

    Args:
        stream: 要读取的流 (例如 process.stdout 或 process.stderr)。
        prefix: 输出行的前缀 (例如 "STDOUT" 或 "STDERR")。
    """
    for line in stream:
        if 'mmengine - INFO' in line or prefix == 'STDERR':
            print(f"{prefix}: {line.strip()}")


video_path1 = '/code/data/demo/demo1.mp4'
video_path2 = '/code/data/demo/demo2.mp4'
img_folder1 = '/code/data/datasets/SAT-MTB_ship/test/19/img'
img_folder2 = '/code/data/datasets/SAT-MTB_ship/test/19/img3'
checkpoint_path1 = '/code/data/checkpoints/bytetrack_ship.pth'
checkpoint_path2 = '/code/data/checkpoints/bytetrack_ship3.pth'
output_path = '/code/data/demo.mp4'

command1 = f'python demo/mot_demo.py {img_folder1} custom_configs/bytetrack_demo.py --checkpoint {checkpoint_path1} --score-thr 0.5 --device cuda:0 --out {video_path1} --fps 30'
command2 = f'python demo/mot_demo.py {img_folder2} custom_configs/bytetrack_demo.py --checkpoint {checkpoint_path2} --score-thr 0.5 --device cuda:9 --out {video_path2} --fps 30'

for c in [command1, command2]:
    process = subprocess.Popen(
        c,
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

merged_video = video_combine.merge_videos(video_path1, video_path2, 30, 90)
# 保存拼接后的视频
merged_video.write_videofile(output_path)
