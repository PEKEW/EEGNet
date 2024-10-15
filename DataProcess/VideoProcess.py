from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2 as cv
import numpy as np

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


def calOpticalFlow(videoRootPath:str, videoID :str) -> None:
    """cal optical flow
    **** video must cliped before this func!**** 
    Args:
        videoRootPath (str): video path
        videoID (str): subject id
    """
    opticalFlowImgList = []
    vd = cv.VideoCapture(videoRootPath)
    _, frame1 = vd.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    idx = 0
    while True:
        print(f'Processing {idx}th frame')
        ret, frame2 = vd.read()
        if not ret:
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        opticalFlowImgList.append(bgr)
        idx += 1
        cv.imwrite(f'OpticalFlows/subj_{videoID}_frame{idx}.png', bgr)
    vd.release()


def videoProcess():
    firstClipInfo = {
        'test' :{
            'begin': 10,
            'end': 20
        },
        # 'TYR': {
        #     'begin': 10,
        #     'end': 20
        # },
        # 'WZK': { # this guy need to redo the exp !
        #     'begin': 10,
        #     'end': 20
        # },
        # 'XSJ': {
        #     'begin': 10,
        #     'end': 20
        # },
        # 'CM': {
        #     'begin': 10,
        #     'end': 20
        # },
        # 'WZT': {
        #     'begin': 10,
        #     'end': 20
        # }
    }
    for subName, clipInfo in firstClipInfo.items():
        print(f'Processing {subName} (1st Clip (norm))')
        clipVideo(f'Videos/{subName}.mp4', f'ClipVideo/norm_{subName}.mp4', clipInfo['begin'], clipInfo['end'])
        # todo
        # 1. cut norm video to parts, which lens is 1s (1s-2s, 2s-3s, 3s-4s, ...)
        # 2. save the part with 30fps ? (30 imgs)
        # 3. cal optical flow for each part
        # 4. save these imgs to a folder / how struct this data? (part ID, frame in each part, of each subj) 
        

if __name__ == '__main__':
    videoProcess()