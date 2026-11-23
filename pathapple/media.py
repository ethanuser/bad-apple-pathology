"""Sample the pinned constant-frame-rate source at explicit, reproducible times."""
import json
import subprocess
from fractions import Fraction

import cv2
import numpy as np

from .inputs import ASSETS

VIDEO = ASSETS / 'badapple.mp4'


def frames(start, seconds, fps):
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        spec = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=r_frame_rate,nb_frames', '-of', 'json', str(VIDEO)]))['streams'][0]
    except FileNotFoundError:
        # Clusters may lack FFmpeg. OpenCV's bundled decoder reports the same
        # 30000/1001 rate and 6,955 frames for the pinned file.
        spec = dict(r_frame_rate=str(Fraction(cap.get(cv2.CAP_PROP_FPS)).limit_denominator(1001)),
                    nb_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    rate = Fraction(spec['r_frame_rate'])
    n = round(seconds * fps)
    if n < 1 or fps <= 0 or start < 0 or abs(seconds * fps - n) > 1e-6:
        raise ValueError('Use a positive, whole number of output frames and nonnegative start.')
    indices = [int((Fraction(str(start)) + Fraction(i, fps)) * rate) for i in range(n)]
    if indices[-1] >= int(spec['nb_frames']):
        raise ValueError('Excerpt exceeds the source duration.')
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    current, result = indices[0] - 1, []
    try:
        for idx in indices:
            while current < idx:
                if not cap.grab():
                    raise ValueError(f'Could not decode source frame {idx}')
                current += 1
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f'Could not retrieve source frame {idx}')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 1444x1080 source has two extra columns per side.
            left = (gray.shape[1] - gray.shape[0] * 4 // 3) // 2
            view = gray[:, left:left + gray.shape[0] * 4 // 3]
            result.append(cv2.resize(view, (384, 288), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    return result, indices, str(rate)


def frame_count():
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
