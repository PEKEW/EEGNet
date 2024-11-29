from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2 as cv
import numpy as np
import json
import csv
from typing import Dict, Any, List
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math

FPS = 30


@dataclass
class Frame:
    """一个frame表示一帧数据，一个log中的一行数据，30个frame组成一个slice
    """
    is_sickness: str  # 0 or 1 表示当前帧是否为报告的sickness帧
    time_: float
    pos: str
    speed: str
    acceleration: str
    rotation_speed: str


class LogProcessor:
    def __init__(self, data_clip_info: Dict[str, Dict[str, int]]) -> None:
        self.data_clip_info = data_clip_info
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def process(self) -> None:
        norm_logs = {}
        for sub_name, clip_info in self.data_clip_info.items():
            sets = self._process_subject(sub_name, clip_info)
            if sets:
                norm_logs[sub_name] = sets
        self._save_norm_logs(norm_logs)

    def _get_log_file_path(self, sub_name: str) -> str:
        return os.path.join(self.current_dir, 'Logs', f'{sub_name}Log.txt')

    def _process_subject(self, sub_name: str, clip_info: Dict[str, int]) -> Dict[str, Dict[str, Frame]]:
        log_file_name = self._get_log_file_path(sub_name)
        try:
            with open(log_file_name, 'r') as f:
                lines = f.readlines()
        except IOError as e:
            print(f'open {log_file_name} failed: {e}')
            return None

        start_time = self._find_start_time(lines)
        if start_time is None:
            print(f'find start time failed for {sub_name}')
            return None

        return self._process_log_lines(lines, start_time, clip_info)

    @staticmethod
    def _find_start_time(lines: List[str]) -> float:
        return next((float(line.split(',')[1].split(':')[1].split(' ')[1])
                    for line in lines
                    if line.split(',')[0].isdigit()), None)

    def _process_log_lines(self, lines: List[str], start_time: float, clip_info: Dict[str, int]) \
            -> Dict[str, Dict[str, Dict[str, str]]]:
        sets = defaultdict(dict)
        slice_idx = 0
        frame_idx = 0
        last_sickness_tag = None
        complete_sickness = False
        is_slice_changed = False
        # 使用绝对时间计算
        expected_slice_count = math.ceil(clip_info['end'] - clip_info['begin'])
        tmp = start_time
        slice_range = []
        for _ in range(expected_slice_count):
            slice_range.append((tmp, tmp + 1))
            tmp += 1
        for line in lines:
            record = line.split(',')
            if len(record) <= 1:
                continue
            if 'Sickness0' in record[1]:
                last_sickness_tag = record[1].split(':')[1][2:-2]
                complete_sickness = False
            elif 'Sickness1' in record[1]:
                current_sickness_tag = record[1].split(':')[1][2:-2]
                if last_sickness_tag == current_sickness_tag:
                    complete_sickness = True

            # 处理数据行
            if not record[0].isdigit():
                continue
            frame = self._get_data_info_from_record(record, 0)
            for idx, (start, end) in enumerate(slice_range):
                if frame.time_ >= start and frame.time_ < end:
                    if slice_idx != idx:
                        slice_idx = idx
                        is_slice_changed = True
                    slice_idx = idx
                    break
            if is_slice_changed:
                frame_idx = 0
                is_slice_changed = False
                complete_sickness = False
            frame.time_ -= start_time
            sets[f'slice_{slice_idx}'][f'frame_{frame_idx}'] = {
                'complete_sickness': complete_sickness,
                **frame.__dict__
            }
            frame_idx += 1

        return dict(sets)

    @staticmethod
    def _get_data_info_from_record(record: List[str], time_shift: float) -> Frame:
        is_sickness = record[0]
        _time = float(record[1].split(':')[1].split(' ')[1]) - time_shift
        pos_x = record[2].split(':')[1][2:]
        pos_y = record[3][1:]
        pos_z = record[4][1:-2]
        pos = f"({pos_x},{pos_y},{pos_z})"
        speed = record[5].split(':')[1][1:]
        acceleration = record[6].split(':')[1][1:]
        rotation_speed = record[7].split(':')[1][1:-2]
        return Frame(is_sickness, _time, pos, speed, acceleration, rotation_speed)

    @staticmethod
    def _save_norm_logs(norm_logs: Dict[str, Dict[str, Dict[str, Frame]]]) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        norm_logs_dir = os.path.join(current_dir, 'norm_logs.json')
        try:
            with open(norm_logs_dir, 'w') as f:
                json.dump(norm_logs, f, default=lambda o: o.__dict__)
        except IOError as e:
            print('write norm_logs.json failed:', e)


def log_process(data_clip_info: Dict[str, Dict[str, int]]) -> None:
    processor = LogProcessor(data_clip_info)
    processor.process()


data_clip_info = {
    'TYR': {  # 263 - 10 = 253
        'begin': 10,
        'end': 263
    },
    'XSJ': {  # 196 - 26 = 170
        'begin': 26,
        'end': 196
    },
    'CM': {  # ica
        'begin': 78,  # 367-78 = 289
        'end': 367
    },
    'TX': {  # 227 - 2 = 225
        'begin': 2,
        'end': 227
    },
    'HZ': {  # ICA # 237 - 4 = 233
        'begin': 4,
        'end': 237
    },
    'CYL': {  # ICA  # 292 - 4 = 288
        'begin': 4,
        'end': 292
    },
    'GKW': {  # ICA # 168 - 22 = 146
        'begin': 22,
        'end': 168
    },
    'LMH': {  # 310 - 7 = 303
        'begin': 7,
        'end': 310
    },
    'WJX': {  # 233 - 3 = 230
        'begin': 3,
        'end': 233
    },

    'CWG': {  # ICA  # 209 - 3 = 206
        'begin': 3,
        'end': 209
    },
    'SHQ': {  # 122 - 2 = 120
        'begin': 2,
        'end': 122
    },
    'YHY': {  # 268 - 2 = 266
        'begin': 2,
        'end': 268
    },
    'LZX': {  # 244 - 11 = 233
        'begin': 11,
        'end': 244
    },
    'LJ': {  # 252 - 3 = 249
        'begin': 3,
        'end': 252
    },
    'WZT': {  # 254 - 5 = 249
        'begin': 5,
        'end': 254
    }
}


def down_sample(videoPath: str, outPath: str, fps: int = 30) -> None:
    """down sample video

    Args:
        fps (int, optional): down sample target. Defaults to 30.
    """
    videoClip = VideoFileClip(videoPath)
    newClip = videoClip.fl_time(lambda t: t, keep_duration=True)
    newClip.write_videofile(outPath, fps=fps)


def clip_video(videoPath: str, outPath: str, startTime: int, endTime: int) -> None:
    """clip video
    Args:
        videoPath (str): video path
        outPath (str): output path
        startTime (int): start time
        endTime (int): end time
    """
    videoClip = VideoFileClip(videoPath)
    newClip = videoClip.subclip(startTime, endTime).without_audio()
    newClip.write_videofile(outPath)


def cal_optical_flow(videoRootPath: str, videoID: str) -> None:
    """cal optical flow
    **** video must cliped before this func!**** 
    Args:
        videoRootPath (str): video path
        videoID (str): subject id
    """
    vd = cv.VideoCapture(videoRootPath)
    _, frame1 = vd.read()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, 'OpticalFlows', f'{
                             videoID}frame_0_original.png')
    cv.imwrite(file_name, frame1)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    idx = 0
    while True:
        print(f'Processing {idx}th frame')
        ret, frame2 = vd.read()
        if not ret:
            print('Video End, total frames:', idx+1)
            break
        file_name = os.path.join(current_dir, 'OpticalFlows', f'{
                                 videoID}frame_{idx}_original.png')
        cv.imwrite(file_name, frame2)
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(
            prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        file_name = os.path.join(current_dir, 'OpticalFlows', f'{
                                 videoID}frame_{idx}_optical.png')
        cv.imwrite(file_name, bgr)
        prvs = next
        idx += 1
    vd.release()


def video_process_multi(data_clip_info: Dict[str, Dict[str, int]]) -> None:
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        futures = [executor.submit(process_subject, sub_name, clip_info)
                   for sub_name, clip_info in data_clip_info.items()]

        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                print(f'Completed {completed} / {total}')
            except Exception as e:
                print(f'Error: {e}')
                import traceback
                traceback.print_exc()


def process_subject(sub_name: str, clip_info: Dict[str, int]) -> None:
    try:
        print(f'Processing {sub_name} (1st Clip (norm))')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_video = os.path.join(current_dir, 'Videos', f'{sub_name}.mp4')
        downsmapled_video = os.path.join(
            current_dir, 'Videos', f'down_{sub_name}.mp4')
        normalized_video = os.path.join(
            current_dir, 'Videos', f'norm_{sub_name}.mp4')
        os.makedirs('Videos', exist_ok=True)
        down_sample(input_video, downsmapled_video)
        clip_video(str(downsmapled_video), str(normalized_video),
                   clip_info['begin'], clip_info['end'])
        clip_duration = clip_info['end'] - clip_info['begin']
        process_video_slices(sub_name, normalized_video, clip_duration)

        process_optical_flow(sub_name, clip_duration)
    except Exception as e:
        print(f'Error processing {sub_name}: {e}')
        raise
    finally:
        import gc
        gc.collect()


def process_video_slices(sub_name: str, video_path: Path, clip_duration: int) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clip_dir = os.path.join(current_dir, 'ClipVideo')
    os.makedirs(clip_dir, exist_ok=True)
    for second in range(clip_duration):
        print(f'slicing video: {second+1} / {clip_duration}')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_clip = os.path.join(current_dir, 'ClipVideo', f'norm_{
                                   sub_name}_{second}.mp4')
        clip_video(str(video_path), str(output_clip), second, second+1)


def process_optical_flow(sub_name: str, clip_duration: int) -> None:
    optical_dir = os.path.join(current_dir, 'OpticalFlows')
    os.makedirs(optical_dir, exist_ok=True)
    for clip_id in range(clip_duration):
        in_ = os.path.join(current_dir, 'ClipVideo',
                           f'norm_{sub_name}_{clip_id}.mp4')
        cal_optical_flow(in_, f'sub_{sub_name}_sclice_{clip_id}_')


def label_process() -> Dict[str, Dict[str, int]]:
    def process_slice(slice: Dict[str, Dict[str, str]]) -> int:
        any_complete_sickness = any(
            frame['complete_sickness'] for frame in slice.values())
        sickness_frames = sum(1 for frame in slice.values()
                              if frame['is_sickness'] == '1')
        total_frames = len(slice)
        if total_frames == 0:
            return 0

        is_positive = (
            sickness_frames / len(slice) >= 0.1 or
            any_complete_sickness
        )

        # print(f"- Is positive: {is_positive}")

        return 1 if is_positive else 0

    result = defaultdict(dict)
    pos_label_cnt = 0
    neg_label_cnt = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_logs_path = os.path.join(current_dir, "norm_logs.json")
    labels_path = os.path.join(current_dir, "labels.json")

    try:
        with open(norm_logs_path, "r") as f:
            norm_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading norm_logs.json: {e}")
        return {}

    cnt = 0
    for subj, sets in norm_logs.items():
        for slice_id, slice in sets.items():
            cnt += 1
            label = process_slice(slice)
            result[subj][slice_id] = label
            if label == 1:
                pos_label_cnt += 1
            else:
                neg_label_cnt += 1

    try:
        with open(labels_path, 'w') as f:
            json.dump(result, f)
    except IOError as e:
        print(f"Error writing to labels.json: {e}")

    print(f'Positive label count: {
          pos_label_cnt}, Negative label count: {neg_label_cnt}. {cnt}')
    return result


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    directories = ['Videos', 'ClipVideo', 'OpticalFlows', 'Logs']
    for directory in directories:
        dir_path = os.path.join(current_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
    video_process_multi(data_clip_info)
    log_process(data_clip_info)
    label_process()
