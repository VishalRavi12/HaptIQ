"""Download the official MediaPipe Hand Landmarker model bundle.

Source: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
"""
from pathlib import Path
from urllib.request import urlretrieve

url = ('https://storage.googleapis.com/mediapipe-models/'
       'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')
dest = Path(__file__).parent / 'assets' / 'hand_landmarker.task'
dest.parent.mkdir(exist_ok=True)
if dest.exists() and dest.stat().st_size > 1000:
    print('Model already present:', dest)
else:
    print('Downloading official Hand Landmarker model...')
    urlretrieve(url, dest)
    print('Model downloaded:', dest)
