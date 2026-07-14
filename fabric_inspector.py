"""
Fabric QA Inspector — Automated computer vision quality assurance pipeline
for textile manufacturing. Combines YOLO object detection with OpenCV for
real-time defect detection, CSV logging, and HUD visualization.

Launch from the command line; a persistent GUI lets you inspect multiple
images or browse a folder without restarting the program.
"""

import csv
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
CSV_PATH = os.path.join(SCRIPT_DIR, "fabric_qa_log.csv")
WINDOW_NAME = "Fabric QA Inspector — Video"
GUI_TITLE = "Fabric QA Inspector"

# Central horizontal ROI band (30%–80% of frame height, full width)
ROI_TOP_RATIO = 0.30
ROI_BOTTOM_RATIO = 0.80

# Video-only cooldown: suppress duplicate logs for the same class
COOLDOWN_FRAMES = 30

# Supported file extensions (lowercase, with leading dot)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

# BGR color constants
COLOR_CLEAR = (0, 255, 0)       # Green
COLOR_DEFECT = (0, 0, 255)      # Bright Red
COLOR_ROI_GUIDE = (180, 180, 180)
COLOR_HUD_BG = (30, 30, 30)
COLOR_HUD_TEXT = (255, 255, 255)
COLOR_BOX_DEFAULT = (0, 255, 255)  # Yellow for out-of-ROI detections


# ---------------------------------------------------------------------------
# File type routing
# ---------------------------------------------------------------------------

def get_input_type(file_path):
    """
    Detect whether the selected file is an image or a video by inspecting
    its extension. Extensions are normalized to lowercase so '.MP4' and '.mp4'
    are treated identically.

    Routing logic:
      - Extensions in IMAGE_EXTENSIONS  -> return 'image'
      - Extensions in VIDEO_EXTENSIONS  -> return 'video'
      - Anything else                     -> return None (unsupported)
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def list_images_in_folder(folder_path):
    """Return sorted absolute paths for supported images in a folder."""
    if not os.path.isdir(folder_path):
        return []

    images = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path) and get_input_type(full_path) == "image":
            images.append(full_path)

    return sorted(images, key=lambda path: os.path.basename(path).lower())


# ---------------------------------------------------------------------------
# CSV data pipeline
# ---------------------------------------------------------------------------

def ensure_csv_exists():
    """Create fabric_qa_log.csv with a header row when it does not exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["Timestamp", "File_Name", "Frame_Number", "Defect_Type", "Confidence"]
            )


def append_csv_row(file_name, frame_number, defect_type, confidence):
    """Append a single validated defect record to the CSV log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([timestamp, file_name, frame_number, defect_type, confidence])


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def compute_roi_bounds(frame_height, frame_width):
    """Return (y_top, y_bottom) pixel coordinates for the horizontal ROI band."""
    y_top = int(frame_height * ROI_TOP_RATIO)
    y_bottom = int(frame_height * ROI_BOTTOM_RATIO)
    return y_top, y_bottom


def bbox_center_in_roi(x1, y1, x2, y2, y_top, y_bottom):
    """
    Calculate the center point of a bounding box and check whether it falls
    inside the ROI. A defect triggers alerts/logging ONLY when its center
    is physically within the horizontal tracking zone.
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    in_roi = y_top <= center_y <= y_bottom
    return in_roi, int(center_x), int(center_y)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_roi_guides(frame, y_top, y_bottom):
    """Draw faint horizontal lines marking the ROI boundaries."""
    width = frame.shape[1]
    cv2.line(frame, (0, y_top), (width, y_top), COLOR_ROI_GUIDE, 1)
    cv2.line(frame, (0, y_bottom), (width, y_bottom), COLOR_ROI_GUIDE, 1)


def draw_hud(frame, system_status, status_color, total_defects):
    """Render a HUD panel showing system status and the defect counter."""
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 420, 80

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        COLOR_HUD_BG,
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (80, 80, 80),
        1,
    )

    status_label = f"System Status: {system_status}"
    cv2.putText(
        frame,
        status_label,
        (panel_x + 12, panel_y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    counter_label = f"Total Defects: {total_defects}"
    cv2.putText(
        frame,
        counter_label,
        (panel_x + 12, panel_y + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_HUD_TEXT,
        2,
        cv2.LINE_AA,
    )


def draw_detections(frame, results, y_top, y_bottom, class_names):
    """
    Render YOLO bounding boxes with class labels and confidence percentages.
    Returns a list of in-ROI defect dicts: {class_name, confidence}.
    """
    in_roi_defects = []
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return in_roi_defects

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        in_roi, _, _ = bbox_center_in_roi(x1, y1, x2, y2, y_top, y_bottom)
        box_color = COLOR_DEFECT if in_roi else COLOR_BOX_DEFAULT

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        label = f"{class_name} {confidence * 100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1 - 8, label_size[1] + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0] + 4, label_y + 4),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if in_roi:
            in_roi_defects.append(
                {"class_name": class_name, "confidence": confidence}
            )

    return in_roi_defects


def update_system_state(in_roi_defects):
    """Set display state based on whether any defect center is inside the ROI."""
    if in_roi_defects:
        return "DEFECT DETECTED", COLOR_DEFECT
    return "System Clear", COLOR_CLEAR


# ---------------------------------------------------------------------------
# Cooldown and logging (video)
# ---------------------------------------------------------------------------

def should_log_defect(class_name, cooldown_remaining, last_logged_class):
    """Video-only anti-double-counting gate."""
    if cooldown_remaining > 0 and class_name == last_logged_class:
        return False
    return True


def log_defects(
    in_roi_defects,
    file_name,
    frame_number,
    cooldown_remaining,
    last_logged_class,
    total_defects,
    apply_cooldown,
):
    """Write eligible defects to CSV and update session counters."""
    for defect in in_roi_defects:
        class_name = defect["class_name"]
        confidence = defect["confidence"]

        if apply_cooldown and not should_log_defect(
            class_name, cooldown_remaining, last_logged_class
        ):
            continue

        append_csv_row(
            file_name,
            frame_number,
            class_name,
            f"{confidence:.3f}",
        )
        total_defects += 1
        last_logged_class = class_name
        cooldown_remaining = COOLDOWN_FRAMES if apply_cooldown else 0

    return total_defects, last_logged_class, cooldown_remaining


# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def inspect_image(file_path, model, session_defect_total):
    """
    Run YOLO on a single image and return the annotated frame plus metadata.
    Does not open any windows — intended for the persistent GUI workflow.
    """
    frame = cv2.imread(file_path)
    if frame is None:
        return None

    file_name = os.path.basename(file_path)
    class_names = model.names

    height, width = frame.shape[:2]
    y_top, y_bottom = compute_roi_bounds(height, width)

    results = model(frame, verbose=False)
    in_roi_defects = draw_detections(frame, results, y_top, y_bottom, class_names)

    image_defects = 0
    image_defects, _, _ = log_defects(
        in_roi_defects,
        file_name,
        "Static Image",
        cooldown_remaining=0,
        last_logged_class=None,
        total_defects=image_defects,
        apply_cooldown=False,
    )

    session_defect_total += image_defects
    system_status, status_color = update_system_state(in_roi_defects)
    draw_roi_guides(frame, y_top, y_bottom)
    draw_hud(frame, system_status, status_color, session_defect_total)

    return {
        "frame": frame,
        "file_name": file_name,
        "file_path": file_path,
        "system_status": system_status,
        "image_defects": image_defects,
        "session_defect_total": session_defect_total,
    }


def process_video(file_path, model, on_complete=None):
    """
    Video pipeline in a dedicated OpenCV window.
    Uses destroyWindow instead of destroyAllWindows so the main GUI stays open.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        if on_complete:
            on_complete(False, f"Could not open video file: {file_path}")
        return 0

    file_name = os.path.basename(file_path)
    class_names = model.names

    total_defects = 0
    cooldown_remaining = 0
    last_logged_class = None
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        height, width = frame.shape[:2]
        y_top, y_bottom = compute_roi_bounds(height, width)

        results = model(frame, verbose=False)
        in_roi_defects = draw_detections(frame, results, y_top, y_bottom, class_names)

        total_defects, last_logged_class, cooldown_remaining = log_defects(
            in_roi_defects,
            file_name,
            frame_number,
            cooldown_remaining,
            last_logged_class,
            total_defects,
            apply_cooldown=True,
        )

        system_status, status_color = update_system_state(in_roi_defects)
        draw_roi_guides(frame, y_top, y_bottom)
        draw_hud(frame, system_status, status_color, total_defects)

        cv2.imshow(WINDOW_NAME, frame)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)

    if on_complete:
        on_complete(True, f"Video complete. Defects logged: {total_defects}")

    return total_defects


# ---------------------------------------------------------------------------
# Persistent GUI application
# ---------------------------------------------------------------------------

class FabricInspectorApp:
    """
    Main application window. The YOLO model loads once at startup; users can
    pick individual images, browse a folder, or launch video inspection
    without closing and restarting the program.
    """

    SIDEBAR_WIDTH = 280

    def __init__(self, model):
        self.model = model
        self.session_defect_total = 0
        self.current_folder = None
        self.folder_images = []
        self.photo_image = None
        self.processing = False
        self.video_thread = None

        self.root = tk.Tk()
        self.root.title(GUI_TITLE)
        self.root.geometry("1200x760")
        self.root.minsize(960, 620)
        self.root.configure(bg="#1e1e1e")

        self._build_layout()
        self._set_status("Ready — select an image or folder to begin.")

    def _build_layout(self):
        """Create sidebar controls and the main inspection canvas."""
        sidebar = tk.Frame(self.root, width=self.SIDEBAR_WIDTH, bg="#2b2b2b")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        title = tk.Label(
            sidebar,
            text="Fabric QA Inspector",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg="#2b2b2b",
            anchor="w",
        )
        title.pack(fill=tk.X, padx=14, pady=(16, 8))

        subtitle = tk.Label(
            sidebar,
            text="Continuous inspection workflow",
            font=("Segoe UI", 9),
            fg="#aaaaaa",
            bg="#2b2b2b",
            anchor="w",
        )
        subtitle.pack(fill=tk.X, padx=14, pady=(0, 16))

        self.btn_select_image = ttk.Button(
            sidebar,
            text="Select Image",
            command=self.on_select_image,
        )
        self.btn_select_image.pack(fill=tk.X, padx=14, pady=4)

        self.btn_select_folder = ttk.Button(
            sidebar,
            text="Select Folder",
            command=self.on_select_folder,
        )
        self.btn_select_folder.pack(fill=tk.X, padx=14, pady=4)

        self.btn_select_video = ttk.Button(
            sidebar,
            text="Select Video",
            command=self.on_select_video,
        )
        self.btn_select_video.pack(fill=tk.X, padx=14, pady=4)

        folder_label = tk.Label(
            sidebar,
            text="Folder Images",
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#2b2b2b",
            anchor="w",
        )
        folder_label.pack(fill=tk.X, padx=14, pady=(18, 6))

        self.folder_path_var = tk.StringVar(value="No folder selected")
        folder_path_label = tk.Label(
            sidebar,
            textvariable=self.folder_path_var,
            font=("Segoe UI", 8),
            fg="#888888",
            bg="#2b2b2b",
            anchor="w",
            wraplength=self.SIDEBAR_WIDTH - 28,
            justify=tk.LEFT,
        )
        folder_path_label.pack(fill=tk.X, padx=14, pady=(0, 8))

        list_frame = tk.Frame(sidebar, bg="#2b2b2b")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.image_listbox = tk.Listbox(
            list_frame,
            activestyle="none",
            bg="#383838",
            fg="#ffffff",
            selectbackground="#0078d4",
            selectforeground="#ffffff",
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 10),
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.image_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.image_listbox.bind("<Double-Button-1>", self.on_listbox_double_click)

        info_frame = tk.Frame(sidebar, bg="#333333")
        info_frame.pack(fill=tk.X, padx=14, pady=(8, 14))

        self.system_status_var = tk.StringVar(value="System Status: —")
        self.session_defects_var = tk.StringVar(value="Session Defects: 0")
        self.current_file_var = tk.StringVar(value="Current File: —")

        for var in (
            self.system_status_var,
            self.session_defects_var,
            self.current_file_var,
        ):
            tk.Label(
                info_frame,
                textvariable=var,
                font=("Segoe UI", 9),
                fg="#dddddd",
                bg="#333333",
                anchor="w",
                wraplength=self.SIDEBAR_WIDTH - 40,
                justify=tk.LEFT,
            ).pack(fill=tk.X, padx=10, pady=4)

        display_frame = tk.Frame(self.root, bg="#111111")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            display_frame,
            bg="#111111",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#252525",
            anchor="w",
            padx=12,
            pady=6,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _set_status(self, message):
        self.status_var.set(message)
        print(message)

    def _set_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_select_image.config(state=state)
        self.btn_select_folder.config(state=state)
        self.btn_select_video.config(state=state)
        self.image_listbox.config(state=state)

    def _on_canvas_resize(self, _event):
        """Re-render the current image when the display area is resized."""
        if hasattr(self, "_current_frame_bgr") and self._current_frame_bgr is not None:
            self._render_frame(self._current_frame_bgr)

    def _render_frame(self, frame_bgr):
        """Scale an annotated OpenCV frame to fit the canvas."""
        self._current_frame_bgr = frame_bgr

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        image_width, image_height = image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = min(scale, 1.0)

        if scale < 1.0:
            new_size = (
                max(int(image_width * scale), 1),
                max(int(image_height * scale), 1),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo_image,
            anchor=tk.CENTER,
        )

    def _update_info_panel(self, result):
        self.system_status_var.set(f"System Status: {result['system_status']}")
        self.session_defects_var.set(
            f"Session Defects: {result['session_defect_total']}"
        )
        self.current_file_var.set(f"Current File: {result['file_name']}")

    def _populate_folder_list(self, folder_path):
        """Load all supported images from the selected folder into the listbox."""
        self.current_folder = folder_path
        self.folder_images = list_images_in_folder(folder_path)

        self.image_listbox.delete(0, tk.END)
        for image_path in self.folder_images:
            self.image_listbox.insert(tk.END, os.path.basename(image_path))

        display_name = os.path.basename(folder_path) or folder_path
        self.folder_path_var.set(
            f"{display_name} ({len(self.folder_images)} images)"
        )

    def _inspect_image_path(self, file_path):
        """Run inspection and update the GUI display."""
        if self.processing:
            return

        if get_input_type(file_path) != "image":
            messagebox.showerror(
                "Unsupported File",
                "Please select a supported image file "
                f"({', '.join(sorted(IMAGE_EXTENSIONS))}).",
            )
            return

        self.processing = True
        self._set_controls_enabled(False)
        self._set_status(f"Processing: {os.path.basename(file_path)}")

        def worker():
            result = inspect_image(file_path, self.model, self.session_defect_total)

            def finish():
                self.processing = False
                self._set_controls_enabled(True)

                if result is None:
                    messagebox.showerror(
                        "Read Error",
                        f"Could not read image:\n{file_path}",
                    )
                    self._set_status("Failed to read image.")
                    return

                self.session_defect_total = result["session_defect_total"]
                self._update_info_panel(result)
                self._render_frame(result["frame"])
                self._set_status(
                    f"Inspected {result['file_name']} — "
                    f"{result['image_defects']} defect(s) logged this image."
                )

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def on_select_image(self):
        """Open a file picker for a single image without restarting the app."""
        file_path = filedialog.askopenfilename(
            title="Select Image for Fabric Inspection",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self._inspect_image_path(file_path)

    def on_select_folder(self):
        """Browse a folder and list all inspectable images in the sidebar."""
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Fabric Images",
        )
        if not folder_path:
            return

        self._populate_folder_list(folder_path)

        if self.folder_images:
            self._set_status(
                f"Loaded {len(self.folder_images)} image(s) from folder. "
                "Click or double-click a file to inspect."
            )
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(0)
            self.image_listbox.activate(0)
            self._inspect_image_path(self.folder_images[0])
        else:
            self._set_status("No supported images found in the selected folder.")

    def on_listbox_select(self, _event):
        """Inspect the highlighted image when the selection changes."""
        selection = self.image_listbox.curselection()
        if not selection or not self.folder_images:
            return

        index = selection[0]
        if 0 <= index < len(self.folder_images):
            self._inspect_image_path(self.folder_images[index])

    def on_listbox_double_click(self, _event):
        """Double-click also triggers inspection (same as single select)."""
        self.on_listbox_select(_event)

    def on_select_video(self):
        """Launch video inspection in a separate OpenCV window."""
        if self.video_thread and self.video_thread.is_alive():
            messagebox.showinfo(
                "Video In Progress",
                "A video is already playing. Press 'q' in the video window to stop.",
            )
            return

        file_path = filedialog.askopenfilename(
            title="Select Video for Fabric Inspection",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        if get_input_type(file_path) != "video":
            messagebox.showerror(
                "Unsupported File",
                "Please select a supported video file "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))}).",
            )
            return

        self._set_status(f"Playing video: {os.path.basename(file_path)} (press q to stop)")

        def worker():
            def on_complete(_success, message):
                self.root.after(0, lambda: self._set_status(message))

            process_video(file_path, self.model, on_complete=on_complete)

        self.video_thread = threading.Thread(target=worker, daemon=True)
        self.video_thread.start()

    def run(self):
        """Start the tkinter event loop."""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Start the Fabric QA Inspector GUI from the command line.

    The YOLO model and CSV log initialize once; the window stays open so you
    can inspect multiple images or browse a folder without restarting.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at '{MODEL_PATH}'.")
        print("Place your trained 'best.pt' file in the same directory as this script.")
        return

    ensure_csv_exists()

    print(f"Loading YOLO model from '{MODEL_PATH}'...")
    model = YOLO(MODEL_PATH)
    print("Launching Fabric QA Inspector GUI...")

    app = FabricInspectorApp(model)
    app.run()


if __name__ == "__main__":
    main()
