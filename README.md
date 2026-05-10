# AI Fight Detection System using YOLOv8-Pose

I developed this project for a recent E-Summit competition to address the limitations of standard security camera software. While most systems can detect a person, they struggle to understand the context of human interaction. This project uses pose estimation to identify physical altercations in real-time by analyzing how individual skeletons overlap and interact over time.

### Project Overview and Logic

The core of this system relies on the YOLOv8-Pose model, which is much more efficient than running a standard detector alongside a separate pose library. By capturing 17 keypoints for every person in the frame, the script can calculate the spatial relationship between individuals with high precision.

One of the biggest challenges in fight detection is avoiding false positives, such as two people simply walking past each other. To solve this, I implemented a logic system based on Intersection over Union (IoU) and a temporal grace period. The system does not trigger an alarm the moment a collision is detected. Instead, it requires a sustained collision of at least 1.5 seconds. This ensures that brief, accidental contact is ignored while actual physical engagements are logged and flagged.

### System Features

1. YOLOv8-Pose Integration. The system uses the Nano version of the YOLOv8-Pose model to ensure it can run on a standard laptop without a high-end GPU. It processes the feed at a 640x360 resolution to maintain a high frame rate.
2. Visual and Audio Alerts. When a potential fight is detected, the on-screen display shifts to a warning state. If the collision continues, a red alert HUD appears, and the system triggers an audible beep using the Pygame mixer library.
3. Automated Evidence Collection. During an active alert, the system automatically captures up to five screenshots as visual evidence. It also logs the start time, end time, and duration of the event into a local CSV file for later review.
4. Efficiency Controls. To keep the processing load low, the script is designed to process every fourth frame. This prevents the hardware from overheating during long monitoring sessions while still providing enough data points to detect rapid movements.

### Setup and Requirements

1. Python Environment. This project requires Python 3.8 or higher. It is recommended to use a virtual environment to manage dependencies.
2. Necessary Libraries. You will need to install the following packages via pip:
* ultralytics
* opencv-python
* numpy
* pygame


3. Model Weights. The system is configured to use yolov8n-pose.pt. The first time you run the script, the model weights will automatically download from the Ultralytics repository.

### Running the Application

1. Clone the repository to your local machine using the git clone command.
2. Ensure your webcam is connected or update the camera source line in the script to point to your specific IP camera stream.
3. Launch the detection system by running: python fight_detection_FINAL.py.
4. While the application is running, you can press the S key to take a manual screenshot of the current frame or press the Q key to safely exit the program.

### Data Management

The system is designed to keep your repository clean. I have configured a .gitignore file to ensure that the screenshots folder and the CSV logs remain on your local machine and are not uploaded to GitHub. The script includes a check at startup to automatically create the screenshots directory if it does not already exist on your desktop, ensuring the program does not crash during its first run.

**Author:** silverbackish
**Contact:** jhapriyanshu698@gmail.com
