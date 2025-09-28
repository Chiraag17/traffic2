Real-Time Traffic Light Detector
This project uses Python and OpenCV to detect red, yellow, and green traffic lights in a video stream. It can process both pre-recorded video files and live webcam feeds. The script identifies traffic lights based on their color, size, and shape, then generates an annotated output video, a CSV log of detections, and a final summary report.

Features
Color-Based Detection: Identifies traffic lights by filtering colors in the HSV color space.

Video and Webcam Support: Accepts a video file path as an argument or defaults to using the primary webcam if no input is provided.

Contour Filtering: Refines detections by analyzing the area and circularity of potential objects to reduce false positives.

Multiple Outputs:

Annotated Video (.avi): A video file showing the original stream with bounding boxes and labels drawn around detected lights.

Detection Log (.csv): A frame-by-frame log indicating whether a red, yellow, or green light was detected.

Summary Report (.txt): A text file summarizing the total frames processed and the percentage of frames where each color was detected.

Real-Time FPS Display: Overlays the current processing speed (Frames Per Second) on the output video.

Setup and Installation
Clone the repository:

git clone [https://github.com/your-username/traffic2.git](https://github.com/your-username/traffic2.git)
cd traffic2

Install dependencies: This script relies on OpenCV and NumPy. You can install them using pip.

pip install opencv-python numpy

How to Use
You can run the script from your terminal. There are two ways to use it:

1. To process a video file:
Use the --video argument to specify the path to your video file.

python traffic_light_detector.py --video traffic.mp4

The script will generate output files based on the input video's name (e.g., traffic_output.avi, traffic_output.csv, and traffic_report.txt).

2. To use a live webcam feed:
Run the script without any arguments.

python traffic_light_detector.py

The script will open your default webcam. Press the ESC key to stop the program. The output files will be named webcam_output.avi, webcam_output.csv, etc.

Understanding the Output
After running the script, you will find three new files in your directory:

*_output.avi: The processed video with detections highlighted.

*_output.csv: A data file with columns frame, red, yellow, green. A 1 indicates a detection in that frame, and a 0 indicates no detection.

*_report.txt: A summary of the entire run, like this:

Traffic Light Detection Report
====================================
Generated at: 2023-10-27T10:30:00.123456
Source: traffic.mp4
Total frames processed: 1500

RED: 450 frames (30.00%)
YELLOW: 150 frames (10.00%)
GREEN: 600 frames (40.00%)

Output video: traffic_output.avi
CSV log: traffic_output.csv

Customization
You can fine-tune the detection algorithm by modifying the constants at the top of the traffic_light_detector.py script:

HSV_RANGES: Adjust the HSV color values to improve detection in different lighting conditions.

MIN_CONTOUR_AREA / MAX_CONTOUR_AREA: Change the minimum or maximum pixel area to detect lights that are closer or farther away.

CIRCULARITY_MIN / CIRCULARITY_MAX: Modify these values to make the shape detection more or less strict.
