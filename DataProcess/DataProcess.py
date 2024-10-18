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

FPS = 30

@dataclass
class Frame:
    complete_sickness: bool
    is_sickness: str
    time: float
    pos: str
    speed: str
    acceleration: str
    rotation_speed: str

# todo LabelProcessor
# todo videoProcessor
# todo test single thread (video process)
# todo test multi thread
# todo remove fucking Path keywords

class LogProcessor:
    def __init__(self, data_clip_info: Dict[str, Dict[str, int]]) -> None:
        self.data_clip_info = data_clip_info

    def process(self) -> None:
        norm_logs = {}
        for sub_name, clip_info in self.data_clip_info.items():
            sets = self._process_subject(sub_name, clip_info)
            if sets:
                norm_logs[sub_name] = sets
        self._save_norm_logs(norm_logs)

    def _process_subject(self, sub_name: str, clip_info: Dict[str, int]) -> Dict[str, Dict[str, Frame]]:
        log_file_name = f'Logs/{sub_name}Log.txt'
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

        return self._process_log_lines(lines, start_time, clip_info)  # Fixed method name

    @staticmethod
    def _find_start_time(lines: List[str]) -> float:
        return next((float(line.split(',')[1].split(':')[1].split(' ')[1])
                    for line in lines
                    if line.split(',')[0].isdigit()), None)

    def _process_log_lines(self, lines: List[str], start_time: float, clip_info: Dict[str, int]) -> Dict[str, Dict[str, Frame]]:
        sets = defaultdict(dict)
        slice_idx, frame_idx = 0, 0
        last_sickness_tag = None
        complete_sickness = False

        for line in lines:
            record = line.split(',')
            if len(record) > 1:
                if 'Sickness0' in record[1]:
                    last_sickness_tag = record[1].split(':')[1][2:-2]
                elif 'Sickness1' in record[1]:
                    current_sickness_tag = record[1].split(':')[1][2:-2]
                    if last_sickness_tag == current_sickness_tag:
                        complete_sickness = True
                        continue

            if record[0].isdigit():
                frame = self._get_data_info_from_record(record, start_time, complete_sickness)
                sets[f'slice_{slice_idx}'][f'frame_{frame_idx}'] = frame
                frame_idx += 1

            if frame_idx == FPS:
                slice_idx += 1
                last_sickness_tag = None
                frame_idx = 0
                complete_sickness = False
                if slice_idx >= clip_info['end'] - clip_info['begin']:
                    break
        return dict(sets)

    @staticmethod
    def _get_data_info_from_record(record: List[str], time_shift: float, complete_sickness: bool = False) -> Frame:
        is_sickness = record[0]
        _time = float(record[1].split(':')[1].split(' ')[1]) - time_shift
        pos_x = record[2].split(':')[1][2:]
        pos_y = record[3][1:]
        pos_z = record[4][1:-2]
        pos = f"({pos_x},{pos_y},{pos_z})"
        speed = record[5].split(':')[1][1:]
        acceleration = record[6].split(':')[1][1:]
        rotation_speed = record[7].split(':')[1][1:-2]
        return Frame(complete_sickness, is_sickness, _time, pos, speed, acceleration, rotation_speed)

    @staticmethod
    def _save_norm_logs(norm_logs: Dict[str, Dict[str, Dict[str, Frame]]]) -> None:
        try:
            with open('norm_logs.json', 'w') as f:
                json.dump(norm_logs, f, default=lambda o: o.__dict__)
        except IOError as e:
            print('write norm_logs.json failed:', e)

def log_process(data_clip_info: Dict[str, Dict[str, int]]) -> None:
    processor = LogProcessor(data_clip_info)
    processor.process()


data_clip_info = {
        'TYR': {
            'begin': 10,
            'end': 263
        },
        'XSJ': {
            'begin': 26,
            'end': 196
        },
        'CM': {
            'begin': 44,
            'end': 334
        },
        'TX': {
            'begin': 2,
            'end': 227
        },
        'HZ':{
            'begin': 4,
            'end': 237
        },
        'CYL':{
            'begin': 4,
            'end': 292
        },
        #todo re clip
        'GKW': { 
            'begin': 22,
            'end': 168
        },
        'LMH':{
            'begin': 7,
            'end': 310
        },
        'WJX': {
            'begin': 3,
            'end': 233
        }
}


def down_sample(videoPath:str, outPath:str, fps:int=30) -> None:
    """down sample video

    Args:
        fps (int, optional): down sample target. Defaults to 30.
    """
    videoClip = VideoFileClip(videoPath)
    newClip = videoClip.fl_time(lambda t: t, keep_duration=True)
    newClip.write_videofile(outPath, fps=fps)


def clipVideo(videoPath:str, outPath: str, startTime:int, endTime:int) -> None:
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


def cal_optical_flow(videoRootPath:str, videoID :str) -> None:
    """cal optical flow
    **** video must cliped before this func!**** 
    Args:
        videoRootPath (str): video path
        videoID (str): subject id
    """
    vd = cv.VideoCapture(videoRootPath)
    _, frame1 = vd.read()
    cv.imwrite(f'Frames/subj_{videoID}_frame0.png', frame1)
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
        cv.imwrite(f'OpticalFlows/{videoID}_frame{idx}_original.png', frame2)
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imwrite(f'OpticalFlows/{videoID}_frame{idx}_optical.png', bgr)
        prvs = next
        idx += 1
    vd.release()

def video_process(data_clip_info: Dict[str, Dict[str, int]]) -> None:
    for sub_name, clip_info in data_clip_info.items():
        print(f'Processing {sub_name} (1st Clip (norm))')
        # input_video = Path(f'Videos/{sub_name}.mp4')
        # downsmapled_video = Path(f'Videos/down_{sub_name}.mp4')
        # normalized_video = Path(f'Videos/norm_{sub_name}.mp4')
        input_video = f'Videos/{sub_name}.mp4'
        downsmapled_video = f'Videos/down_{sub_name}.mp4'
        normalized_video = f'Videos/norm_{sub_name}.mp4'

        down_sample(input_video, downsmapled_video)
        normalized_video = clipVideo(str(downsmapled_video), str(normalized_video), clip_info['begin'], clip_info['end'])
        clip_duration = clip_info['end'] - clip_info['begin']
        for second in range(clip_duration):
            print(f'slicing video: {second+1} / {clip_duration}')
            # output_clip = Path(f'ClipVideo/norm_{sub_name}_{second}.mp4')
            output_clip = f'ClipVideo/norm_{sub_name}_{second}.mp4'
            clipVideo(str(normalized_video), str(output_clip), second, second+1)
        print("cal optical flow")
        for clip_id in range(clip_duration):
            # cal_optical_flow(Path(f'ClipVideo/norm_{sub_name}_{clip_id}.mp4'), f'sub_{sub_name}_sclice_{clip_id}_')
            cal_optical_flow(f'ClipVideo/norm_{sub_name}_{clip_id}.mp4', f'sub_{sub_name}_sclice_{clip_id}_')
        write_dataset_csv(sub_name, clip_duration)

def video_process_multi(data_clip_info: Dict[str, Dict[str, int]]) -> None:
    with ThreadPoolExecutor() as executor:
        future = [executor.submit(process_subject, sub_name, clip_info)
                for sub_name, clip_info in data_clip_info.items()]
        for future in as_completed(future):
            try:
                future.result()
            except Exception as e:
                print(f'Error: {e}')
def process_subject(sub_name: str, clip_info: Dict[str, int]) -> None:
    print(f'Processing {sub_name} (1st Clip (norm))')
    # input_video = Path(f'Videos/{sub_name}.mp4')
    # downsmapled_video = Path(f'Videos/down_{sub_name}.mp4')
    # normalized_video = Path(f'Videos/norm_{sub_name}.mp4')
    input_video = f'Videos/{sub_name}.mp4'
    downsmapled_video = f'Videos/down_{sub_name}.mp4'
    normalized_video = f'Videos/norm_{sub_name}.mp4'


    down_sample(input_video, downsmapled_video)
    normalized_video = clipVideo(str(downsmapled_video), str(normalized_video), clip_info['begin'], clip_info['end'])
    clip_duration = clip_info['end'] - clip_info['begin']
    process_video_slices(sub_name, normalized_video, clip_duration)
    process_optical_flow(sub_name, clip_duration)
    write_dataset_csv(sub_name, clip_duration)

def process_video_slices(sub_name: str, video_path: Path, clip_duration: int) -> None:
    for second in range(clip_duration):
        print(f'slicing video: {second+1} / {clip_duration}')
        # output_clip = Path(f'ClipVideo/norm_{sub_name}_{second}.mp4')
        output_clip = f'ClipVideo/norm_{sub_name}_{second}.mp4'
        clipVideo(str(video_path), str(output_clip), second, second+1)

def process_optical_flow(sub_name: str, clip_duration: int) -> None:
    for clip_id in range(clip_duration):
        cal_optical_flow(f'ClipVideo/norm_{sub_name}_{clip_id}.mp4', f'sub_{sub_name}_sclice_{clip_id}_')

def write_dataset_csv(sub_name: str, clip_duration: int) -> None:
    csv_path = Path('datasets.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csv_path.stat*(0).st_size == 0:
            writer.writerow(['subj', 'Slice', 'Frame', 'OriginalPath', 'OpticalPath'])
        for clip_id in range(clip_duration):
            for frame in range(FPS):
                writer.writerow([
                    f'sub_{sub_name}',
                    f'_sclice_{clip_id}',
                    f'_frame_{frame}',
                    f'_originalPath_OpticalFlows/sub_{sub_name}_sclice_{clip_id}_frame{frame}_original.png',
                    f'_OpticalPath_OpticalFlows/sub_{sub_name}_sclice_{clip_id}_frame{frame}_optical.png'
                ])



def label_process() -> Dict[str, Dict[str, int]]:
    def process_slice(frames: Dict) -> int:
        sickness_cnt = 0
        for data in frames.values():
            if data['completeSickness']:
                return 1
            if data['isSickness'] == '1':
                sickness_cnt += 1
            if sickness_cnt >= FPS//2:
                return 1
        return 0
    result = defaultdict(dict)
    pos_label_cnt = 0
    neg_label_cnt = 0

    try:
        with open("normLogs.json", "r") as f:
            norm_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading normLogs.json: {e}")
        return {}

    cnt = 0
    for subj, sets in norm_logs.items():
        for slice_id, frames in sets.items():
            cnt += 1
            label = process_slice(frames)
            result[subj][slice_id] = label
            if label == 1:
                pos_label_cnt += 1
            else:
                neg_label_cnt += 1

    try:
        with open('labels.json', 'w') as f:
            json.dump(result, f)
    except IOError as e:
        print(f"Error writing to labels.json: {e}")

    print(f'Positive label count: {pos_label_cnt}, Negative label count: {neg_label_cnt}. {cnt}')
    return result




if __name__ == '__main__':
    # video_process(data_clip_info)
    video_process_multi(data_clip_info)
    log_process(data_clip_info)
    label_process()