# Fabric QA Inspector

Automated computer vision quality assurance system for textile manufacturing. This SDET final project combines a custom YOLO object detection model with OpenCV to detect fabric defects, log results to CSV, and display a real-time inspection dashboard.

## Features

- **Persistent GUI workflow** — launch once from the command line; inspect many images without restarting
- **Folder browser** — select a folder and click any listed image to process it instantly
- **Dynamic input handling** — supports images (`.jpg`, `.jpeg`, `.png`) and videos (`.mp4`, `.avi`, `.mov`)
- **Region of Interest (ROI)** — defects are only flagged when their bounding box center falls inside a central horizontal tracking zone (30%–80% of frame height)
- **Anti-double-counting** — video mode uses a 30-frame cooldown per defect class to prevent duplicate CSV entries
- **CSV data pipeline** — automatic log creation and append with timestamp, file name, frame number, defect type, and confidence
- **HUD overlay** — live system status, defect counter, ROI guides, and YOLO bounding boxes



## Prerequisites

- Python 3.9 or newer
- A display environment (for the OpenCV window and tkinter file dialog)
- Your trained YOLO weights file: `best.pt`



## Setup

1. Clone or download this project folder.
2. Create and activate a virtual environment (recommended):
  ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
  ```
3. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
4. Place your trained model weights in the project root:
  ```
   YOLO Fabric Inspector/
   ├── best.pt              <-- your trained weights here
   ├── fabric_inspector.py
   └── requirements.txt
  ```
   **GPU note:** Ultralytics installs PyTorch automatically. For CUDA GPU acceleration, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/) for your platform before or after installing requirements.



## Usage

Run the inspector from the project directory:

```bash
python fabric_inspector.py
```

The GUI opens and stays running so you can inspect multiple files in one session.

### GUI Workflow


| Button / Action   | What it does                                                                       |
| ----------------- | ---------------------------------------------------------------------------------- |
| **Select Image**  | Pick a single image; results appear in the main panel immediately                  |
| **Select Folder** | Pick a folder; all images are listed in the sidebar — click any file to inspect it |
| **Select Video**  | Opens video playback in a separate OpenCV window                                   |


After each image inspection, pick another image or click a different file in the folder list — no need to restart the program.

### Controls


| Mode  | Action                                                      |
| ----- | ----------------------------------------------------------- |
| Video | Press `q` in the video window to stop and return to the GUI |
| GUI   | Close the main window to exit the application               |




## Output

On first run, the script creates `fabric_qa_log.csv` in the project directory.

### CSV Schema


| Column       | Description                                     | Example             |
| ------------ | ----------------------------------------------- | ------------------- |
| Timestamp    | Human-readable date and time of the log entry   | 2026-07-14 10:30:45 |
| File_Name    | Base name of the inspected source file          | fabric_roll_01.mp4  |
| Frame_Number | Video frame index, or `Static Image` for photos | 142                 |
| Defect_Type  | YOLO class name of the detected defect          | tear                |
| Confidence   | Model confidence score (0–1)                    | 0.872               |




### Example row

```csv
2026-07-14 10:30:45,fabric_roll_01.mp4,142,tear,0.872
```

Subsequent runs append new rows without overwriting existing data.

## ROI and Cooldown Behavior

**Region of Interest:** A horizontal band spanning the full frame width between 30% and 80% of the image height. Only detections whose bounding box *center point* falls inside this zone trigger alerts and CSV logging.

**Cooldown (video only):** After a defect class is logged, the same class is suppressed for the next 30 frames. If a *different* defect class appears during the cooldown window, it is logged immediately.

**Image mode:** No cooldown is applied. Each in-ROI detection on the single frame is logged once.

## Project Structure

```
YOLO Fabric Inspector/
├── best.pt                 # Trained YOLO weights (you provide)
├── fabric_inspector.py     # Main inspection script
├── fabric_qa_log.csv       # Auto-generated defect log
├── requirements.txt        # Python dependencies
└── README.md               # This file
```



## Troubleshooting


| Issue                        | Solution                                                                |
| ---------------------------- | ----------------------------------------------------------------------- |
| `Model weights not found`    | Ensure `best.pt` is in the same folder as `fabric_inspector.py`         |
| `Unsupported file extension` | Use `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, or `.mov`                  |
| `Could not open video file`  | Verify the video is not corrupted and codecs are supported              |
| File dialog does not appear  | Confirm tkinter is installed (included with standard Python on Windows) |


