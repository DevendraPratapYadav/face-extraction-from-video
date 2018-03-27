# Face extraction and crop from video
Extract cropped face video segments from an input video using OpenCV and Python

## This project is currently in progress

### Dependencies:
1. OpenCV
2. Python 3.5
3. pyannote-video

### Input : 
1. Video file to be processed is placed in 'videoSample' folder.
2. Perform face detection and tracking on video using pyannote-video to generate a txt file containing face bounding box at each frame.

Format : Each line in txt file contains 'time xmin ymin xmax ymax correlationType'

Eg. '0.000 0 0.514 0.200 0.665 0.469 detection'

Sample file: videoSample/tbbt.mp4.track.txt.

### Output : 
Cropped face videos are written to 'croppedFaces' folder.

### Run using:

python faceCrop.py <videoFile\> <face_detections_txt_file\> <videoFolder\>   

Example
 ```sh
   python faceCrop.py "tbbt.mp4" "tbbt.mp4.track.txt" videoSample
 ```
