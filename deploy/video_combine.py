from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_videos(video1_path, video2_path, m, angle):
    """
  将两个 MP4 视频拼接在一起。

  Args:
    angle:
    video1_path: 第一个视频的路径。
    video2_path: 第二个视频的路径。
    m:  两个视频重叠的帧数。

  Returns:
    拼接后的视频剪辑对象。
  """

    # 加载视频
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    # 旋转视频2
    video2 = video2.rotate(angle, expand=False)

    # 计算帧率
    fps1 = video1.fps
    fps2 = video2.fps

    # 计算时间
    t1 = video1.duration / 2 + m / fps1
    t2 = video2.duration / 2 + m / fps2

    # 截取视频片段
    clip1 = video1.subclip(0, t1)
    clip2 = video2.subclip(video2.duration - t2, video2.duration)

    # 拼接视频
    final_clip = concatenate_videoclips([clip1, clip2])

    return final_clip


index = 19
video1_path = f"/data1/zhuhongchun/outputs/bytetrack_ship/demo/demo{index}.mp4"
video2_path = f"/data1/zhuhongchun/outputs/bytetrack_ship3/demo/demo{index}.mp4"
m = 30

merged_video = merge_videos(video1_path, video2_path, m, 90)

# 保存拼接后的视频
merged_video.write_videofile(f"data/demo{index}c.mp4")
