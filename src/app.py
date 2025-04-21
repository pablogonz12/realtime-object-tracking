"""
Computer Vision Project - Main Application
This file combines the functionality from GUI.py, main.py, and display_video.py
into a single unified application with both GUI and CLI interfaces.
"""

import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import curses
import time  # Import time module for FPS control and timing
from pathlib import Path

# Add project root to PYTHONPATH so sibling packages (data_sets, inference) are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import traceback  # Import traceback for detailed error logging
import threading  # Import threading for video preview
from PIL import Image, ImageTk  # Import PIL for image processing
from ttkthemes import ThemedTk # Import ThemedTk
import tkinter.simpledialog as simpledialog  # for input dialogs
from data_sets.dataset_manager import DatasetManager
from inference.evaluate_models import ModelEvaluator, COCO_VAL_TOTAL_IMAGES
import pandas as pd # Import pandas for displaying results table

# Import model wrappers directly from consolidated models file
from models.models import ModelManager # Removed unused YOLOWrapper, RTDETRWrapper, SAMWrapper imports

# Default model paths for auto-download
DEFAULT_MODELS = {
    'faster-rcnn': 'models/pts/fasterrcnn_resnet50_fpn.pt',
    'yolo-seg': 'models/pts/yolov8n-seg.pt',
    'rtdetr': 'models/pts/rtdetr-l.pt'
}

def display_video_with_model(video_path, model_manager):
    """
    Displays video with object detection/segmentation overlays

    Args:
        video_path (str): Path to video file
        model_manager (ModelManager): Initialized model manager
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video {video_path}") # Use messagebox for GUI errors
        print(f"Error: Cannot open video {video_path}")
        return

    print(f"Starting video processing with {model_manager.model_type} model on device {model_manager.device}...") # Show device
    frame_count = 0
    processing_errors = 0
    max_errors_to_show = 5 # Limit number of error popups

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading the video.")
            break

        frame_count += 1
        print(f"\nProcessing frame {frame_count}...") # Print frame number

        # Perform prediction based on model type
        try:
            annotated_frame = None # Initialize annotated_frame
            if model_manager.model_type == "sam":
                print("  Model type is SAM. Generating input points...")
                # For SAM, create a grid of prompt points
                h, w = frame.shape[:2]

                # Create a 3x3 grid of points in the central region
                grid_size = 3
                y_points = np.linspace(h // 4, 3 * h // 4, grid_size, dtype=int)
                x_points = np.linspace(w // 4, 3 * w // 4, grid_size, dtype=int)

                input_points = np.array([[x, y] for y in y_points for x in x_points])

                # Generate labels (all foreground points)
                input_labels = np.ones(len(input_points), dtype=np.int32)
                print(f"  Generated {len(input_points)} input points.")

                # Run prediction with points and labels
                print("  Calling model_manager.predict for SAM...")
                results = model_manager.predict(frame, input_points=input_points, input_labels=input_labels) # Pass labels
                print("  model_manager.predict for SAM returned.")

                # Unpack results (expecting masks, scores, annotated_frame)
                if results and len(results) == 3:
                    masks, scores, annotated_frame = results
                    if annotated_frame is None: # Handle case where annotation might fail
                         annotated_frame = frame.copy()
                         cv2.putText(annotated_frame, "Annotation Error", (30, 60),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"  SAM processing successful for frame {frame_count}.")
                else:
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "No segmentation results", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"  SAM processing returned unexpected results or no results for frame {frame_count}.")
            else:
                print(f"  Model type is {model_manager.model_type}. Calling model_manager.predict...")
                # For YOLO and RT-DETR (expecting detections, segmentations, annotated_frame)
                results = model_manager.predict(frame)
                if results and len(results) == 3:
                    detections, segmentations, annotated_frame = results
                    if annotated_frame is None: # Handle case where annotation might fail
                         annotated_frame = frame.copy()
                         cv2.putText(annotated_frame, "Annotation Error", (30, 60),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"  {model_manager.model_type} processing successful for frame {frame_count}.")
                else:
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "No detection results", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"  {model_manager.model_type} processing returned unexpected results or no results for frame {frame_count}.")

            # Display the annotated frame
            cv2.imshow("Video with Object Recognition", annotated_frame)

        except Exception as e:
            processing_errors += 1
            print(f"Error processing frame {frame_count}:")
            print(traceback.format_exc())
            # Show the original frame if processing fails
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Error: {str(e)[:50]}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Video with Object Recognition", error_frame)
            # Show error popup only for the first few errors to avoid spamming
            if processing_errors <= max_errors_to_show:
                 messagebox.showwarning("Processing Error", f"Error processing frame {frame_count}:\n{e}\n\nCheck console for details.")
            if processing_errors == max_errors_to_show + 1:
                 messagebox.showwarning("Processing Error", "Further processing errors will only be logged to the console.")


        # Break the loop if 'q' is pressed or window is closed
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("User pressed 'q'. Exiting...")
            break
        # Check if the window was closed
        try:
            # This will raise an error if the window is closed
            if cv2.getWindowProperty("Video with Object Recognition", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Exiting...")
                break
        except cv2.error:
            # Error means window is closed
            print("Window closed by user. Exiting...")
            break


    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

def terminal_gui():
    """
    Terminal-based GUI for selecting video and model
    
    Returns:
        tuple: (selected_video_path, selected_model_path)
    """
    def draw_menu(stdscr, current_row):
        stdscr.clear()
        stdscr.addstr("Select a video:\n", curses.A_BOLD)
        for idx, video in enumerate(videos):
            if idx == current_row:
                stdscr.addstr(f"> {video}\n", curses.color_pair(1))
            else:
                stdscr.addstr(f"  {video}\n")

        stdscr.addstr("\nSelect a model:\n", curses.A_BOLD)
        for idx, model in enumerate(models):
            if idx + len(videos) == current_row:
                stdscr.addstr(f"> {model}\n", curses.color_pair(1))
            else:
                stdscr.addstr(f"  {model}\n")

        stdscr.refresh()

    videos = [
        "data_sets/video_data/samples/bottle-detection.mp4",
        "data_sets/video_data/samples/car-detection.mp4",
        "data_sets/video_data/samples/classroom.mp4",
        "data_sets/video_data/samples/fruit-and-vegetable-detection.mp4",
        "data_sets/video_data/samples/one-by-one-person-detection.mp4",
        "data_sets/video_data/samples/people-detection.mp4",
        "data_sets/video_data/samples/person-bicycle-car-detection.mp4",
        "data_sets/video_data/samples/store-aisle-detection.mp4",
        "data_sets/video_data/samples/worker-zone-detection.mp4",
    ]

    models = [
        "models/yolov8n-seg.pt",
        "yolov8n.pt",  # Root level model
        "models/yolov8n.pt",  # Also check in models directory
    ]

    def main(stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

        current_row = 0
        draw_menu(stdscr, current_row)

        while True:
            key = stdscr.getch()

            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(videos) + len(models) - 1:
                current_row += 1
            elif key == ord("q"):
                return None, None
            elif key == ord("\n"):
                if current_row < len(videos):
                    selected_video = videos[current_row]
                    selected_model = models[0]  # Default to the first model
                else:
                    selected_video = videos[0]  # Default to the first video
                    selected_model = models[current_row - len(videos)]
                return selected_video, selected_model

            draw_menu(stdscr, current_row)

    try:
        selected_video, selected_model = curses.wrapper(main)
        return selected_video, selected_model
    except Exception as e:
        print(f"Error in terminal GUI: {e}")
        print("Falling back to command line interface")
        return None, None

class GraphicalGUI:
    """
    Tkinter-based graphical user interface for the Computer Vision application.
    Provides controls for model selection, video input, processing, and evaluation.
    """
    def __init__(self):
        # Use ThemedTk for modern themes. Try 'arc', 'breeze', 'radiance', etc.
        self.root = ThemedTk(theme="arc") 
        self.root.title("Computer Vision Project")
        self.root.geometry("1100x700")
        self.root.minsize(800, 500)  # Prevent window from becoming too small
        
        # Initialize variables that could be accessed early
        self.status_var = tk.StringVar(value="Ready")
        self.model_type_var = tk.StringVar(value="yolo-seg")  # Define model type variable
        self.gpu_var = tk.StringVar(value="auto")
        self.gpu_status_var = tk.StringVar(value="Checking GPU...")
        self.model_info_var = tk.StringVar(value="")
        self.current_dir_var = tk.StringVar(value=str(Path.home()))  # Use Path object for home dir
        self.custom_video_path = tk.StringVar()
        self.custom_model_path = tk.StringVar()
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.preview_img = None  # Will store PhotoImage for preview
        
        # --- State Variables ---
        self.video_thread = None  # Will hold the thread for video processing
        self.stop_preview = False  # Flag to stop preview threads
        self.pause_processing = False  # Flag to pause processing
        self.processing_active = False  # Flag to track if processing is currently active
        
        # Initialize statistics attributes
        self.frame_count = 0
        self.processing_time = 0.0
        self.detection_counts = {}
        
        # Define model requirements - moved this BEFORE using it in _update_model_info
        self.model_requirements = {
            "faster-rcnn": {
                "description": "Two-stage detection with ResNet50 FPN backbone (TorchVision)",
                "ram": "8GB",
                "vram": "4GB",
                "cpu_cores": "4",
                "pytorch": "1.10+"
            },
            "yolo-seg": {
                "description": "Single-stage detection + segmentation (Ultralytics)",
                "ram": "4GB",
                "vram": "2GB",
                "cpu_cores": "2",
                "pytorch": "1.10+"
            },
            "rtdetr": {
                "description": "Real-time detection transformer (Ultralytics)",
                "ram": "6GB",
                "vram": "3GB",
                "cpu_cores": "4",
                "pytorch": "1.10+"
            },
            "sam": {
                "description": "Segment Anything Model (Meta AI)",
                "ram": "16GB",
                "vram": "8GB",
                "cpu_cores": "8",
                "pytorch": "1.12+"
            }
        }
        
        # Initialize data - now model_requirements is defined before this is called
        self.available_samples = self.get_sample_videos()
        self._update_model_info()
        
        # Check for GPU availability at startup
        self.check_gpu_availability()
        
        # Setup UI for closing app properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup the UI and menu
        self._setup_ui()
        self._init_menu()
        
    def _setup_ui(self):
        """Set up the main UI layout and components."""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 0))
        
        control_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(control_frame, weight=1)
        control_frame.columnconfigure(0, weight=1)
        
        preview_frame = ttk.Frame(main_paned, padding=(0, 10, 10, 10))
        main_paned.add(preview_frame, weight=2)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        
        self._setup_model_section(control_frame)
        self._setup_input_section(control_frame)
        self._setup_action_section(control_frame)
        self._setup_preview_panel(preview_frame)
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)
        
    def _setup_model_section(self, parent):
        """Setup the model selection and info widgets."""
        model_group = ttk.LabelFrame(parent, text="Model Configuration", padding=10)
        model_group.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        model_group.columnconfigure(1, weight=1)
        
        ttk.Label(model_group, text="Type:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        model_type_dropdown = ttk.Combobox(model_group, textvariable=self.model_type_var,
                                           values=list(DEFAULT_MODELS.keys()), state="readonly", width=15)
        model_type_dropdown.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        model_type_dropdown.bind("<<ComboboxSelected>>", self._on_model_type_change)
        
        add_model_btn = ttk.Button(model_group, text="+ Custom", width=10, command=self._add_custom_model)
        add_model_btn.grid(row=0, column=2, sticky="e", padx=(5, 0))
        
        ttk.Label(model_group, text="Device:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        gpu_dropdown = ttk.Combobox(model_group, textvariable=self.gpu_var,
                                    values=["auto", "cpu", "cuda"], state="readonly", width=10)
        gpu_dropdown.grid(row=1, column=1, sticky="w", pady=(5, 0))
        gpu_dropdown.bind("<<ComboboxSelected>>", self._on_gpu_change)
        
        gpu_status_label = ttk.Label(model_group, textvariable=self.gpu_status_var, anchor="w")
        gpu_status_label.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(80, 0), pady=(5, 0))
        
        model_info_label = ttk.Label(model_group, textvariable=self.model_info_var, wraplength=350,
                                     anchor="nw", justify=tk.LEFT, padding=(0, 5, 0, 0))
        model_info_label.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
    def _setup_input_section(self, parent):
        """Setup the video input selection widgets using a Notebook."""
        input_group = ttk.LabelFrame(parent, text="Video Input", padding=10)
        input_group.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        input_group.rowconfigure(0, weight=1)
        input_group.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        self.file_notebook = ttk.Notebook(input_group)
        self.file_notebook.grid(row=0, column=0, sticky="nsew")
        
        sample_frame = ttk.Frame(self.file_notebook, padding=5)
        self.file_notebook.add(sample_frame, text="Sample Videos")
        sample_frame.rowconfigure(0, weight=1)
        sample_frame.columnconfigure(0, weight=1)
        
        sample_list_frame = ttk.Frame(sample_frame)
        sample_list_frame.grid(row=0, column=0, sticky="nsew")
        sample_list_frame.rowconfigure(0, weight=1)
        sample_list_frame.columnconfigure(0, weight=1)
        
        sample_scrollbar = ttk.Scrollbar(sample_list_frame, orient=tk.VERTICAL)
        sample_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.sample_listbox = tk.Listbox(sample_list_frame, height=8, yscrollcommand=sample_scrollbar.set, exportselection=False)
        self.sample_listbox.grid(row=0, column=0, sticky="nsew")
        sample_scrollbar.config(command=self.sample_listbox.yview)
        
        for _, display_name in self.available_samples:
            self.sample_listbox.insert(tk.END, display_name)
        self.sample_listbox.bind('<<ListboxSelect>>', self._on_sample_select)
        
        explorer_frame = ttk.Frame(self.file_notebook, padding=5)
        self.file_notebook.add(explorer_frame, text="File Explorer")
        explorer_frame.rowconfigure(1, weight=1)
        explorer_frame.columnconfigure(0, weight=1)
        
        dir_nav_frame = ttk.Frame(explorer_frame)
        dir_nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        dir_nav_frame.columnconfigure(0, weight=1)
        
        up_dir_btn = ttk.Button(dir_nav_frame, text="Up", width=4, command=self._go_up_directory)
        up_dir_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        current_dir_entry = ttk.Entry(dir_nav_frame, textvariable=self.current_dir_var, state="readonly")
        current_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_dir_btn = ttk.Button(dir_nav_frame, text="Browse", command=self._browse_directory)
        browse_dir_btn.pack(side=tk.LEFT)
        
        explorer_panes = ttk.PanedWindow(explorer_frame, orient=tk.HORIZONTAL)
        explorer_panes.grid(row=1, column=0, sticky="nsew")
        
        dir_list_frame = ttk.Frame(explorer_panes, padding=2)
        explorer_panes.add(dir_list_frame, weight=1)
        dir_list_frame.rowconfigure(0, weight=1)
        dir_list_frame.columnconfigure(0, weight=1)
        
        dir_scrollbar = ttk.Scrollbar(dir_list_frame, orient=tk.VERTICAL)
        dir_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.dir_listbox = tk.Listbox(dir_list_frame, height=8, yscrollcommand=dir_scrollbar.set, exportselection=False)
        self.dir_listbox.grid(row=0, column=0, sticky="nsew")
        dir_scrollbar.config(command=self.dir_listbox.yview)
        self.dir_listbox.bind("<Double-1>", self._change_directory)
        
        file_list_frame = ttk.Frame(explorer_panes, padding=2)
        explorer_panes.add(file_list_frame, weight=2)
        file_list_frame.rowconfigure(0, weight=1)
        file_list_frame.columnconfigure(0, weight=1)
        
        file_scrollbar = ttk.Scrollbar(file_list_frame, orient=tk.VERTICAL)
        file_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.file_listbox = tk.Listbox(file_list_frame, height=8, yscrollcommand=file_scrollbar.set, exportselection=False)
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        file_scrollbar.config(command=self.file_listbox.yview)
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)
        
        self._update_explorer()
        
    def _setup_action_section(self, parent):
        """Setup the action buttons (Run, Pause, Stop, Webcam)."""
        action_frame = ttk.Frame(parent, padding=5)
        action_frame.grid(row=2, column=0, sticky="ew")
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=1)
        action_frame.columnconfigure(3, weight=1)
        
        self.webcam_button = ttk.Button(action_frame, text="Start Webcam", command=self._toggle_webcam_detection)
        self.webcam_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.processing_buttons_frame = ttk.Frame(action_frame)
        self.processing_buttons_frame.grid(row=0, column=1, columnspan=3, sticky="e")
        
        self.run_button = ttk.Button(self.processing_buttons_frame, text="Run Model", command=self._run_model)
        self.pause_button = ttk.Button(self.processing_buttons_frame, text="Pause", command=self._pause_processing)
        self.stop_button = ttk.Button(self.processing_buttons_frame, text="Stop", command=self._stop_processing)
        
        self._update_buttons(processing=False)
        
    def _setup_preview_panel(self, parent):
        """Setup the video preview canvas."""
        preview_group = ttk.LabelFrame(parent, text="Preview / Output", padding=5)
        preview_group.grid(row=0, column=0, sticky="nsew")
        preview_group.rowconfigure(0, weight=1)
        preview_group.columnconfigure(0, weight=1)
        
        self.preview_canvas = tk.Canvas(preview_group, bg="#2B2B2B", highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas_image_ref = None
        
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)
        self._display_message_on_canvas("Select a video or start webcam")
        
    def _update_buttons(self, processing):
        """Update button states based on processing status."""
        self.processing_active = processing
        if processing:
            self.run_button.pack_forget()
            self.pause_button.pack(side=tk.LEFT, padx=(0, 5))
            self.stop_button.pack(side=tk.LEFT)
            self.webcam_button.config(state=tk.DISABLED)
        else:
            self.pause_button.pack_forget()
            self.stop_button.pack_forget()
            self.run_button.pack(side=tk.LEFT)
            self.webcam_button.config(state=tk.NORMAL, text="Start Webcam")
            self.pause_processing = False
            self.pause_button.config(text="Pause")
            
    def _display_message_on_canvas(self, message, title="Info"):
        """ Safely display a text message centered on the canvas (from any thread). """
        self.root.after(0, self._display_message_on_canvas_threadsafe, message, title)
        
    def _display_message_on_canvas_threadsafe(self, message, title):
        """ Internal method to display message (runs in main thread). """
        self.preview_canvas.delete("all")
        
        # Get the actual canvas dimensions (with fallbacks if not yet rendered)
        width = self.preview_canvas.winfo_width() or 400
        height = self.preview_canvas.winfo_height() or 300
        
        # Create text centered based on actual canvas dimensions
        self.preview_canvas.create_text(
            width // 2,
            height // 2,
            text=message,
            fill="white", 
            font=("Arial", 12), 
            justify=tk.CENTER
        )
        
    def _display_image_on_canvas(self, image_np, message=None):
        """ Safely display an OpenCV image (BGR) centered on the canvas (from any thread). """
        self.root.after(0, self._display_image_on_canvas_threadsafe, image_np, message)
        
    def _display_image_on_canvas_threadsafe(self, image_np, message):
        """ Internal method to display image (runs in main thread). """
        self.preview_canvas.delete("all")
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        self._canvas_image_ref = ImageTk.PhotoImage(image_pil)
        self.preview_canvas.create_image(
            self.preview_canvas.winfo_width() // 2,
            self.preview_canvas.winfo_height() // 2,
            image=self._canvas_image_ref,
            anchor=tk.CENTER
        )
        if message:
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() - 30,
                text=message,
                fill="white", font=("Arial", 10), justify=tk.CENTER,
                anchor=tk.CENTER
            )
            
    def _on_canvas_resize(self, event):
        """ Handle canvas resize events to redraw content centered. """
        pass
        
    def _init_menu(self):
        """Initialize the menu bar with Dataset and Evaluate options"""
        menubar = tk.Menu(self.root)
        
        # Dataset menu
        dataset_menu = tk.Menu(menubar, tearoff=0)
        dataset_menu.add_command(label="Download Full COCO", command=self.download_coco_full)
        dataset_menu.add_command(label="Download COCO Subset...", command=self.download_coco_subset)
        dataset_menu.add_separator()
        dataset_menu.add_command(label="Decompress Datasets", command=self.decompress_datasets)
        dataset_menu.add_command(label="Compress Datasets", command=self.compress_datasets)
        dataset_menu.add_separator()
        dataset_menu.add_command(label="Delete All Datasets", command=self.delete_datasets_gui)
        menubar.add_cascade(label="Dataset", menu=dataset_menu)
        
        # Evaluation menu
        eval_menu = tk.Menu(menubar, tearoff=0)
        eval_menu.add_command(label="Run Evaluation...", command=self.run_evaluation_gui)
        menubar.add_cascade(label="Evaluate", menu=eval_menu)
        
        # Attach to root
        self.root.config(menu=menubar)
        
    def download_coco_full(self):
        """Download the full COCO val2017 dataset"""
        dm = DatasetManager()
        self.status_var.set("Downloading full COCO dataset...")
        self.root.update()
        dm.setup_coco(None)
        messagebox.showinfo("Dataset Manager", "Full COCO dataset download and setup complete.")
        self.status_var.set("Ready")
        
    def download_coco_subset(self):
        """Prompt for subset size and download COCO subset"""
        size = simpledialog.askinteger("COCO Subset", "Enter number of images for subset:", minvalue=1)
        if size:
            dm = DatasetManager()
            self.status_var.set(f"Downloading COCO subset ({size} images)...")
            self.root.update()
            dm.setup_coco(size)
            messagebox.showinfo("Dataset Manager", f"COCO subset of {size} images ready.")
            self.status_var.set("Ready")
            
    def delete_datasets_gui(self):
        """Confirm and delete all datasets via DatasetManager"""
        if messagebox.askyesno("Delete Datasets", "Are you sure you want to delete all image datasets? This cannot be undone."):
            dm = DatasetManager()
            dm.delete_datasets()
            messagebox.showinfo("Dataset Manager", "All datasets deleted.")
            self.status_var.set("Ready")
            
    def run_evaluation_gui(self):
        """Prompt for evaluation parameters and run evaluation in background"""
        # Ask for number of images
        num = simpledialog.askinteger("Run Evaluation", f"Number of images (1-{COCO_VAL_TOTAL_IMAGES}):", minvalue=1, maxvalue=COCO_VAL_TOTAL_IMAGES)
        if not num:
            return
        no_vis = messagebox.askyesno("Visualizations", "Skip saving visualizations? (Yes to skip)")
        
        # Run evaluation in a thread
        def eval_task():
            self.status_var.set(f"Evaluating models on {num} images...")
            self.root.update() # Update status immediately
            
            # --- Progress Callback Definition ---
            def update_progress_display(model_type, current_image, total_images):
                """Callback function to update the preview canvas during evaluation."""
                progress_percent = (current_image / total_images) * 100 if total_images > 0 else 0
                message = (
                    f"Evaluating: {model_type.upper()}\n"
                    f"Image {current_image} of {total_images}\n"
                    f"Progress: {progress_percent:.1f}%"
                )
                # Use root.after to ensure GUI update happens in the main thread:
                self.root.after(0, self.show_preview_message, message)
            # --- End Progress Callback Definition ---
            
            # Initial message before starting
            self.show_preview_message(f"Starting evaluation on {num} images...")
            evaluator = ModelEvaluator()
            try:
                # Pass the callback function to run_evaluation
                evaluator.run_evaluation(
                    max_images=num, 
                    save_visualizations=not no_vis, 
                    progress_callback=update_progress_display 
                )
                results = evaluator.results # Capture results
                # Display final results in the GUI
                self.root.after(0, self.show_evaluation_results, results)
                messagebox.showinfo("Evaluation Complete", 
                                    "Model evaluation finished.\nResults are displayed in the preview panel.\n"
                                    "Detailed reports and charts saved in the 'inference/results' folder.")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                traceback.print_exc()
                messagebox.showerror("Evaluation Error", f"An error occurred during evaluation:\n{e}")
                self.show_preview_message(f"Evaluation failed.\nError: {e}") # Show error in preview
            finally:
                self.status_var.set("Ready") # Reset status bar
                
        threading.Thread(target=eval_task, daemon=True).start()
        
    def decompress_datasets(self):
        """Decompress datasets for use"""
        dm = DatasetManager()
        self.status_var.set("Decompressing datasets...")
        self.root.update()
        dm.decompress_datasets()
        messagebox.showinfo("Dataset Manager", "Datasets decompressed successfully.")
        self.status_var.set("Ready")
        
    def compress_datasets(self):
        """Compress datasets to save space"""
        dm = DatasetManager()
        self.status_var.set("Compressing datasets...")
        self.root.update()
        dm.compress_datasets()
        messagebox.showinfo("Dataset Manager", "Datasets compressed successfully.")
        self.status_var.set("Ready")
        
    def get_available_videos(self):
        """Get list of available video samples"""
        video_dir = Path("data_sets/video_data/samples")
        default_video = "data_sets/video_data/samples/people-detection.mp4" # Define default
        if not video_dir.is_dir(): # Check if it's a directory
            print(f"Warning: Video directory not found: {video_dir}")
            # Create the directory if it doesn't exist? Or just return default?
            # os.makedirs(video_dir, exist_ok=True) # Optionally create it
            return [default_video] # Return default if dir doesn't exist
        videos = sorted([str(f) for f in video_dir.glob("*.mp4")]) # Sort for consistency
        return videos if videos else [default_video] # Return default if dir is empty
        
    def _on_model_type_change(self, event):
        """Handle model type change"""
        model_type = self.model_type_var.get()
        # Update model info text with description and requirements
        if model_type in self.model_requirements:
            req = self.model_requirements[model_type]
            self.model_info_var.set(
                f"{model_type.upper()}: {req['description']}\n\n"
                f"Minimum Requirements:\n"
                f"• RAM: {req['ram']}\n"
                f"• VRAM: {req['vram']}\n"
                f"• CPU: {req['cpu_cores']}\n"
                f"• PyTorch: {req['pytorch']}"
            )
        else:
            # Provide fallback description
            self.model_info_var.set(f"{model_type.upper()}: Model type info not available")
        # Update default model path for advanced settings (if implemented)
        if self.show_advanced_var.get():
            self.custom_model_path.set(DEFAULT_MODELS.get(model_type, ""))
        self.root.update_idletasks()  # Force update to resize window properly
        
    def _browse_video(self):
        """Open file dialog to select custom video"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if file_path:
            self.custom_video_path.set(file_path)
            self.start_video_preview(file_path)
            
    def _browse_model(self):
        """Open file dialog to select custom model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.pt *.pth *.onnx"), ("All files", "*.*")] # Added onnx
        )
        if file_path:
            self.custom_model_path.set(file_path)
            
    def _browse_config(self):
        """Open file dialog to select config file (Currently unused)"""
        file_path = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("Config files", "*.yaml *.yml *.json"), ("All files", "*.*")]
        )
        # if file_path:
        #     self.config_path.set(file_path) # Config path currently not used by wrappers
            
    def _run_model(self):
        """Run the model with selected parameters"""
        # If we're already processing, don't start again
        if self.processing_active:
            return
        try:
            # Get model type
            model_type = self.model_type_var.get()
            # Determine video path based on selection method
            video_path = self.get_selected_video_path()
            if not video_path:
                # Error message is already shown by get_selected_video_path
                return
            # Get model path (custom or default)
            if self.show_advanced_var.get() and self.custom_model_path.get():
                model_path = self.custom_model_path.get()
            else:
                model_path = DEFAULT_MODELS.get(model_type)
                if not model_path:
                     messagebox.showerror("Error", f"Default model path not defined for {model_type}.")
                     return
            # Config path is currently not used by the wrappers
            config_path = None # args.config_path
            # --- Basic Path Validations ---
            if not os.path.exists(video_path):
                messagebox.showerror("Error", f"Video file not found or not specified:\n{video_path}")
                return
            if not model_path:
                 messagebox.showerror("Error", "Model path not specified.")
                 return
            # --- Model Initialization and Validation ---
            self.status_var.set(f"Initializing {model_type} model...")
            self.root.update()
            model_manager = None # Initialize to None
            try:
                model_manager = ModelManager(model_type, model_path, config_path)
                if model_manager.model_wrapper is None or not hasattr(model_manager.model_wrapper, 'model') or model_manager.model_wrapper.model is None:
                     raise ValueError("Model wrapper or underlying model failed to initialize.")
            except Exception as e:
                print(traceback.format_exc()) # Log detailed error
                messagebox.showerror("Model Initialization Error",
                                     f"Failed to load the {model_type} model from:\n{model_path}\n\n"
                                     f"Error: {e}\n\n"
                                     "Please ensure the model file is correct and dependencies are installed. "
                                     "Check console for more details.")
                self.status_var.set("Error: Model initialization failed.")
                return
            # --- Run Processing ---
            self.status_var.set(f"Running {model_type} on {os.path.basename(video_path)}...")
            self.processing_active = True
            # Update button states
            self._update_buttons(True)
            # Process the video in the preview panel
            self.process_video_with_model(video_path, model_manager)
        except Exception as e:
            print(traceback.format_exc()) # Log detailed error
            messagebox.showerror("Runtime Error", f"An error occurred during processing:\n{e}")
            self.status_var.set("Error occurred. See console for details.")
            # Reset processing state
            self.processing_active = False
            # Restore button state
            self._update_buttons(False)
            
    def get_selected_video_path(self):
        """Get the selected video path based on the active tab and selection"""
        # Get the active tab
        active_tab = self.file_notebook.index(self.file_notebook.select())
        # Sample Videos tab (index 0)
        if active_tab == 0:
            selection = self.sample_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a sample video.")
                return None
            # Get the full path from the stored list using the index
            return self.available_samples[selection[0]][0]
        # File Explorer tab (index 1)
        elif active_tab == 1:
            selection = self.file_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a video file from the explorer.")
                return None
            # Construct path using the current directory and selected filename
            selected_file = self.file_listbox.get(selection[0])
            file_path = os.path.join(self.current_dir_var.get(), selected_file)
            return file_path
        # Advanced settings custom path
        elif self.show_advanced_var.get() and self.custom_video_path.get():
            return self.custom_video_path.get()
        # Fallback error
        messagebox.showerror("Error", "No video selection method available.")
        return None
        
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
        
    def get_sample_videos(self):
        """Get list of available sample videos with clean display names"""
        sample_dir = Path("data_sets/video_data/samples")
        if not sample_dir.is_dir():
            print(f"Warning: Sample directory not found: {sample_dir}")
            return []
        samples = []
        for video_path in sorted(sample_dir.glob("*.mp4")):
            # Create a tuple of (full_path, clean_display_name)
            # Remove extension and replace hyphens with spaces for better display
            display_name = video_path.stem.replace('-', ' ').title()
            samples.append((str(video_path), display_name))
        return samples
        
    def _browse_directory(self):
        """Open file dialog to select a directory"""
        directory = filedialog.askdirectory(initialdir=self.current_dir_var.get())
        if directory:
            self.current_dir_var.set(directory)
            self._update_explorer()
            
    def _go_up_directory(self):
        """Navigate up one directory level"""
        current_dir = Path(self.current_dir_var.get())
        parent_dir = current_dir.parent
        self.current_dir_var.set(str(parent_dir))
        self._update_explorer()
        
    def _change_directory(self, event):
        """Change to the selected directory"""
        selection = self.dir_listbox.curselection()
        if selection:
            selected_dir = self.dir_listbox.get(selection[0])
            new_dir = Path(self.current_dir_var.get()) / selected_dir
            self.current_dir_var.set(str(new_dir))
            self._update_explorer()
            
    def _update_explorer(self):
        """Update the file explorer with the current directory contents"""
        try:
            current_dir = Path(self.current_dir_var.get())
            if not current_dir.is_dir():
                messagebox.showerror("Error", f"Invalid directory: {current_dir}")
                return
            # Update directory listbox
            self.dir_listbox.delete(0, tk.END)
            # Add parent directory entry if not at root
            if current_dir.parent != current_dir:
                self.dir_listbox.insert(tk.END, "..")
            # Add subdirectories
            for item in sorted(current_dir.iterdir()):
                if item.is_dir():
                    self.dir_listbox.insert(tk.END, item.name)
            # Update file listbox with video files (multiple formats)
            self.file_listbox.delete(0, tk.END)
            video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
            for ext in video_extensions:
                for item in sorted(current_dir.glob(f"*{ext}")):
                    self.file_listbox.insert(tk.END, item.name)
            # Update status bar with directory info
            file_count = len(list(self.file_listbox.get(0, tk.END)))
            self.status_var.set(f"Directory: {current_dir} | {file_count} video files found")
        except Exception as e:
            print(f"Error updating file explorer: {e}")
            messagebox.showerror("Error", f"Failed to read directory: {e}")
        # Removed self.apply_theme() call
        
    def _add_custom_model(self):
        """Open file dialog to select and add a custom model to the dropdown"""
        file_path = filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[("Model files", "*.pt *.pth *.onnx"), ("All files", "*.*")]
        )
        if not file_path:
            return  # User cancelled
        # Get model name from path
        model_name = Path(file_path).name
        model_type = "custom"  # Default type for custom models
        # Try to determine model type from filename
        if "yolo" in model_name.lower():
            if "seg" in model_name.lower():
                model_type = "yolo-seg"
            else:
                model_type = "yolo"
        elif "rtdetr" in model_name.lower():
            model_type = "rtdetr"
        elif "rcnn" in model_name.lower() or "faster" in model_name.lower():
            model_type = "faster-rcnn"
        # Add the model to DEFAULT_MODELS if it doesn't exist
        global DEFAULT_MODELS
        if model_name not in DEFAULT_MODELS.values():
            DEFAULT_MODELS[model_name] = file_path
        # Get current dropdown values
        model_dropdown = None
        for widget in self.root.winfo_children():
            for subwidget in widget.winfo_children():
                if isinstance(subwidget, ttk.Frame):  # Main frame
                    for child in subwidget.winfo_children():
                        if isinstance(child, ttk.Combobox):
                            model_dropdown = child
                            break
                    if model_dropdown:
                        break
        if model_dropdown:
            # Get current values and add new model if not present
            values = list(model_dropdown['values'])
            if model_name not in values:
                values.append(model_name)
                model_dropdown['values'] = values
            # Select the new model
            model_dropdown.set(model_name)
        # Update model type and info
        self.model_type_var.set(model_name)
        self.model_info_var.set(f"Custom Model: {model_name} ({model_type})")
        # Show message to confirm
        messagebox.showinfo("Model Added", 
                           f"Custom model '{model_name}' has been added to the dropdown.\n\n"
                           f"Path: {file_path}\n"
                           f"Detected Type: {model_type}")
        # Store the model path for later use
        self.custom_model_path.set(file_path)
        
    def start_video_preview(self, video_path):
        """Start a thread to preview the selected video"""
        if self.video_thread and self.video_thread.is_alive():
            self.stop_preview = True
            self.video_thread.join()
        self.stop_preview = False
        self.video_thread = threading.Thread(target=self.update_video_preview, args=(video_path,))
        self.video_thread.start()
        
    def update_video_preview(self, video_path):
        """Update the video preview panel with frames from the selected video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.preview_label.config(text="Cannot open video")
            return
        while not self.stop_preview:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320, 240))
            self.preview_img = tk.PhotoImage(image=tk.Image.fromarray(frame))
            self.preview_label.config(image=self.preview_img, text="")
            if self.stop_preview:
                break
            self.root.update_idletasks()
            self.root.after(30)
        cap.release()
        
    def on_closing(self):
        """Handle application shutdown properly"""
        # Stop any running video preview threads
        self.stop_preview = True
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
            try:
                self.video_thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error stopping video thread: {e}")
        # Destroy the window
        self.root.destroy()
        
    def _on_sample_select(self, event):
        """Handle selection of a sample video"""
        selection = self.sample_listbox.curselection()
        if not selection:
            return
        # Get the selected sample video path
        video_path = self.available_samples[selection[0]][0]
        self.show_video_preview(video_path)
        
    def _on_file_select(self, event):
        """Handle selection of a file in the file explorer"""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        # Get the selected file path
        selected_file = self.file_listbox.get(selection[0])
        file_path = os.path.join(self.current_dir_var.get(), selected_file)
        self.show_video_preview(file_path)
        
    def show_video_preview(self, video_path):
        """Show preview of the selected video in the preview panel"""
        # Stop any existing preview
        self.stop_preview = True
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        self.stop_preview = False
        # Clear previous content
        self.preview_canvas.delete("all")
        # Display loading message
        self.preview_message = self.preview_canvas.create_text(
            self.preview_canvas.winfo_width() // 2, 
            self.preview_canvas.winfo_height() // 2,
            text="Loading preview...",
            fill="white", font=("Arial", 12), justify=tk.CENTER
        )
        self.root.update_idletasks()
        # Start preview in a separate thread
        self.video_thread = threading.Thread(
            target=self.video_preview_thread,
            args=(video_path,),
            daemon=True
        )
        self.video_thread.start()
        
    def video_preview_thread(self, video_path):
        """Thread function to display video preview frames"""
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.show_preview_message(f"Cannot open video:\n{video_path}")
                return
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Calculate resize ratio to fit in the canvas
            canvas_width = self.preview_canvas.winfo_width() or 400
            canvas_height = self.preview_canvas.winfo_height() or 300
            # Make sure we have valid dimensions
            if canvas_width <= 10:
                canvas_width = 400
            if canvas_height <= 10:
                canvas_height = 300
            # Calculate scaling ratio
            ratio = min(canvas_width / width, canvas_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                self.show_preview_message("Error reading video frame")
                cap.release()
                return
            # Display the first frame as a preview with message
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            # Convert to PhotoImage format
            self.preview_image = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            # Update canvas on the main thread
            self.root.after(0, self.update_preview_canvas, self.preview_image, 
                           f"Press 'Run Model' to process video\n{os.path.basename(video_path)}\n"
                           f"{width}x{height}, {fps:.1f} FPS, {frame_count} frames")
            cap.release()
        except Exception as e:
            print(f"Error in video preview thread: {e}")
            self.show_preview_message(f"Error previewing video:\n{str(e)}")
            
    def update_preview_canvas(self, image, message_text):
        """Update the preview canvas with an image and message (called from main thread)"""
        try:
            # Clear canvas
            self.preview_canvas.delete("all")
            # Show image
            self.preview_canvas.create_image(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                image=image,
                anchor=tk.CENTER
            )
            # Add message overlay at the bottom
            self.preview_message = self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() - 30,
                text=message_text,
                fill="white", font=("Arial", 10), justify=tk.CENTER,
                anchor=tk.CENTER
            )
            # Add semi-transparent overlay behind text for readability
            bbox = self.preview_canvas.bbox(self.preview_message)
            if bbox:
                padding = 10
                rect = self.preview_canvas.create_rectangle(
                    bbox[0] - padding, bbox[1] - padding,
                    bbox[2] + padding, bbox[3] + padding,
                    fill="black", outline="", stipple="gray25", 
                    tags="overlay"
                )
                self.preview_canvas.tag_lower(rect, self.preview_message)
        except Exception as e:
            print(f"Error updating preview canvas: {e}")
            
    def process_video_with_model(self, video_path, model_manager):
        """Process video with object detection/segmentation overlays and display in preview panel
        
        Args:
            video_path (str): Path to video file
            model_manager (ModelManager): Initialized model manager
        """
        # Stop any existing preview
        self.stop_preview = True
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        self.stop_preview = False
        # Clear previous content
        self.preview_canvas.delete("all")
        # Display processing message
        self.preview_message = self.preview_canvas.create_text(
            self.preview_canvas.winfo_width() // 2, 
            self.preview_canvas.winfo_height() // 2,
            text=f"Processing video with {model_manager.model_type}...",
            fill="white", font=("Arial", 12), justify=tk.CENTER
        )
        self.root.update_idletasks()
        # Show pause and stop buttons
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        # Start processing in a separate thread
        self.video_thread = threading.Thread(
            target=self._process_video_thread,
            args=(video_path, model_manager),
            daemon=True
        )
        self.video_thread.start()
        
    def _process_video_thread(self, video_path, model_manager):
        """Thread function to process and display video frames"""
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.show_preview_message(f"Cannot open video:\n{video_path}")
                return
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # --- Performance Optimizations ---
            # 1. Frame Skipping: Process every Nth frame
            frame_skip = 2 # Process every 2nd frame (adjust as needed)
            # 2. Processing Resolution: Resize before inference
            process_width = 640 # Standard processing width
            process_height = 480 # Standard processing height
            # 3. UI Update Rate Limitations ---
            target_display_fps = min(15, fps / frame_skip) # Limit display FPS
            frame_interval_ms = int(1000 / target_display_fps) if target_display_fps > 0 else 100 # Min 100ms interval
            # --- End Optimizations ---
            # Store canvas dimensions for resizing display frames
            canvas_width = self.preview_canvas.winfo_width() or 400  
            canvas_height = self.preview_canvas.winfo_height() or 300
            # Calculate scaling ratio to fit display frame in the canvas
            display_ratio = min(canvas_width / process_width, canvas_height / process_height)
            display_width = int(process_width * display_ratio)
            display_height = int(process_height * display_ratio)
            frame_index = 0 # Keep track of original frame index
            processed_frame_count = 0 # Keep track of frames actually processed
            processing_errors = 0
            last_update_time = 0
            # Statistics tracking
            start_time = time.time()
            pause_time = 0
            total_pause_duration = 0
            detection_counts = {}
            print(f"Starting video processing with {model_manager.model_type} model...")
            model_device = getattr(model_manager, 'device', 'unknown')
            print(f"Model device: {model_device}")
            while not self.stop_preview:
                # Handle pausing
                if self.pause_processing:
                    if pause_time == 0:                        
                        pause_start_time = time.time() # Record when pause started
                        self.root.after(0, lambda: self.status_var.set(f"Paused at frame {frame_index}/{total_frames}"))
                        pause_time = pause_start_time # Set pause_time to non-zero
                    time.sleep(0.1)  # Reduce CPU usage while paused
                    continue
                elif pause_time > 0:  # Just resumed
                    total_pause_duration += (time.time() - pause_time)
                    pause_time = 0
                    self.root.after(0, lambda count=frame_index, total=total_frames: 
                                   self.status_var.set(f"Resumed processing at frame {count}/{total}"))
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream.")
                    break # Exit loop if frame read fails
                frame_index += 1
                # --- Apply Frame Skipping ---
                if frame_index % frame_skip != 0:
                    continue # Skip this frame
                processed_frame_count += 1
                current_time = int(time.time() * 1000)
                # Update status periodically based on processed frames
                if processed_frame_count % 5 == 0: # Update status less frequently
                    self.root.after(0, lambda count=frame_index, total=total_frames: 
                                   self.status_var.set(f"Processing frame {count}/{total} ({processed_frame_count} processed)"))
                try:
                    # --- Resize frame BEFORE processing ---
                    frame_resized_for_processing = cv2.resize(frame, (process_width, process_height))
                    # Process frame with model
                    # Assuming model_manager methods handle the resized frame
                    if model_manager.model_type == "sam":
                        # SAM might need specific handling (e.g., embeddings first)
                        # This part needs to be adapted based on how SAM is integrated
                        # For now, assume a generic process_frame method exists
                        annotated_frame, detections = model_manager.process_frame(frame_resized_for_processing)
                    else:
                        # Standard object detection/segmentation
                        # predict now returns (detections, segmentations, annotated_frame)
                        detections, segmentations, annotated_frame = model_manager.predict(frame_resized_for_processing) # Pass resized frame
                        # Update detection counts using the returned detections list
                        for det in detections:
                            class_name = det.get('class_name', 'Unknown')
                            if class_name != 'Unknown':
                                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    # Limit display update rate to keep UI responsive
                    if (current_time - last_update_time) >= frame_interval_ms:
                        # Resize the *annotated* frame for display
                        frame_resized_for_display = cv2.resize(annotated_frame, (display_width, display_height))
                        frame_rgb = cv2.cvtColor(frame_resized_for_display, cv2.COLOR_BGR2RGB)
                        # Create PhotoImage and update canvas on main thread
                        pil_img = Image.fromarray(frame_rgb)
                        self.root.after(0, self._update_video_canvas, 
                                       pil_img, frame_index, total_frames, model_manager.model_type)
                        # Update timestamp
                        last_update_time = current_time
                    # No need for time.sleep(0.001) if UI updates are throttled
                except Exception as e:
                    processing_errors += 1
                    print(f"Error processing frame {frame_index}: {e}")
                    traceback.print_exc()
                    if processing_errors <= 5:  # Limit error messages
                        error_msg = f"Error processing frame {frame_index}: {str(e)[:50]}"
                        self.root.after(0, lambda msg=error_msg: 
                                      messagebox.showwarning("Processing Error", msg))
            # Release resources
            cap.release()
            print("Video processing complete.")
            # Calculate final statistics
            processing_time = time.time() - start_time - total_pause_duration
            actual_fps = processed_frame_count / processing_time if processing_time > 0 else 0
            # Prepare statistics summary
            stats = {
                "frames_processed": processed_frame_count, # Report processed frames
                "total_frames": total_frames, # Report total frames in video
                "processing_time": processing_time,
                "actual_fps": actual_fps, # FPS based on processed frames
                "errors": processing_errors,
                "detections": detection_counts,
                "model_type": model_manager.model_type,
                "model_device": model_device
            }
            # Update status when done
            self.root.after(0, lambda: self.status_var.set("Processing complete"))
            self.root.after(0, lambda s=stats: self.show_completion_stats(s))
            # Reset processing state when done
            self.root.after(0, lambda: setattr(self, 'processing_active', False))
            # Reset button states
            self.root.after(0, lambda: self._update_buttons(False))
        except Exception as e:
            print(f"Error in video processing thread: {e}")
            traceback.print_exc()
            self.root.after(0, lambda err=str(e): self.show_preview_message(
                f"Error during video processing:\n{err}"))
            self.root.after(0, lambda err=str(e): messagebox.showerror(
                "Video Processing Error", f"An error occurred during video processing:\n{err}"))
            # Reset processing state on error
            self.root.after(0, lambda: setattr(self, 'processing_active', False))
            # Reset button states on error
            self.root.after(0, lambda: self._update_buttons(False))
            
    def _update_video_canvas(self, pil_img, frame_count, total_frames, model_type):
        """Update the video canvas with a processed frame from the model (called from main thread)"""
        try:
            # Clear canvas
            self.preview_canvas.delete("all")
            # Convert PIL image to PhotoImage
            self.processed_frame = ImageTk.PhotoImage(image=pil_img)
            # Calculate position to center the image
            x_center = self.preview_canvas.winfo_width() // 2
            y_center = self.preview_canvas.winfo_height() // 2
            # Create image on canvas
            self.preview_canvas.create_image(
                x_center, y_center,
                image=self.processed_frame,
                anchor=tk.CENTER
            )
            # Add frame counter overlay - adjust color
            progress_text = f"Frame {frame_count}/{total_frames}"
            self.preview_canvas.create_text(
                10, 20, text=progress_text, 
                fill="#CCCCCC", font=("Arial", 10), anchor=tk.W # Lighter color
            )
            # Add model type overlay - adjust color
            self.preview_canvas.create_text(
                10, 40, text=f"Model: {model_type}", 
                fill="#CCCCCC", font=("Arial", 10), anchor=tk.W # Lighter color
            )
        except Exception as e:
            print(f"Error updating video canvas: {e}")
            
    def _pause_processing(self):
        """Pause or resume video processing"""
        self.pause_processing = not self.pause_processing
        self.pause_button.config(text="Resume" if self.pause_processing else "Pause")
        
    def _stop_processing(self):
        """Stop video processing"""
        self.stop_preview = True
        self.pause_processing = False
        self.status_var.set("Processing stopped")
        # Reset processing state
        self.processing_active = False
        # Update buttons back to normal
        self._update_buttons(False)
        
    def show_completion_stats(self, stats):
        """Show a summary of processing statistics"""
        try:
            # Clear canvas
            self.preview_canvas.delete("all")
            # Calculate canvas dimensions for layout
            canvas_width = self.preview_canvas.winfo_width() or 400
            canvas_height = self.preview_canvas.winfo_height() or 300
            # Background rectangle for better readability (adjust color for theme)
            # Use a slightly lighter dark color that fits 'arc' theme
            bg_color = "#383838" # Example color, adjust as needed
            self.preview_canvas.create_rectangle(
                20, 20, canvas_width - 20, canvas_height - 20,
                fill=bg_color, outline="#555555", width=1
            )
            # Create a header with summary info (adjust color)
            header_text = f"Processing Results: {stats['model_type'].upper()}"
            header_color = "#77AADD" # Example color
            self.preview_canvas.create_text(
                canvas_width // 2, 40,
                text=header_text,
                fill=header_color, font=("Arial", 16, "bold"),
                anchor=tk.CENTER
            )
            # Create processing statistics section (adjust color)
            y_pos = 80
            section_header_color = "#DDDDDD"
            self.preview_canvas.create_text(
                40, y_pos,
                text="Processing Statistics:",
                fill=section_header_color, font=("Arial", 12, "bold"),
                anchor=tk.W
            )
            y_pos += 25
            # Format duration as minutes:seconds
            minutes = int(stats['processing_time'] // 60)
            seconds = stats['processing_time'] % 60
            duration_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
            # Safely handle division by zero or missing stats for frame percentage
            try:
                frame_total = stats.get('total_frames', 0)
                frame_processed = stats.get('frames_processed', 0)
                frame_pct = (frame_processed / frame_total * 100) if frame_total > 0 else 0.0
                frame_stat_str = (
                    f"• Video frames: {frame_processed}/{frame_total}"
                    + (f" ({frame_pct:.1f}%)" if frame_total > 0 else "")
                )
            except Exception:
                frame_processed = stats.get('frames_processed', 0)
                frame_total = stats.get('total_frames', 0)
                frame_stat_str = f"• Video frames: {frame_processed}/{frame_total}"
            proc_stats = [
                frame_stat_str,
                f"• Processing time: {duration_str}",
                f"• Processing speed: {stats.get('actual_fps', 0):.1f} FPS",
                f"• Hardware: {stats.get('model_device', 'N/A')}",
                f"• Errors encountered: {stats.get('errors', 0)}"
            ]
            
            stat_color = "#FFFFFF" # White for stats text
            for stat in proc_stats:
                self.preview_canvas.create_text(
                    50, y_pos,
                    text=stat,
                    fill=stat_color, font=("Arial", 11),
                    anchor=tk.W
                )
                y_pos += 22
            # Create detections summary section if we have detections
            if stats['detections']:
                y_pos += 10
                self.preview_canvas.create_text(
                    40, y_pos,
                    text="Detection Summary:",
                    fill=section_header_color, font=("Arial", 12, "bold"),
                    anchor=tk.W
                )
                y_pos += 25
                # Sort detections by count (descending)
                sorted_detections = sorted(
                    stats['detections'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                total_detections = sum(count for _, count in sorted_detections)
                # Display top detections (limit to 10)
                max_detections_to_show = 5 # Limit display for space
                for i, (class_name, count) in enumerate(sorted_detections[:max_detections_to_show]):
                    percentage = (count / total_detections * 100) if total_detections > 0 else 0
                    # Use brighter colors suitable for dark background
                    colors = ["#87CEFA", "#90EE90", "#FFD700", "#FFA07A", "#DDA0DD", 
                             "#7FFFD4", "#FFB6C1", "#F0E68C", "#FF69B4", "#DEB887"]
                    color = colors[i % len(colors)]
                    self.preview_canvas.create_text(
                        50, y_pos,
                        text=f"• {class_name}: {count} ({percentage:.1f}%)",
                        fill=color, font=("Arial", 11),
                        anchor=tk.W
                    )
                    y_pos += 22
                # If we have more than max_detections_to_show detection classes, show a summary for the rest
                if len(sorted_detections) > max_detections_to_show:
                    remaining = sum(count for _, count in sorted_detections[max_detections_to_show:])
                    percentage = (remaining / total_detections * 100) if total_detections > 0 else 0
                    self.preview_canvas.create_text(
                        50, y_pos,
                        text=f"• Other classes ({len(sorted_detections) - max_detections_to_show}): {remaining} ({percentage:.1f}%)",
                        fill="#BBBBBB", font=("Arial", 11), # Lighter gray
                        anchor=tk.W
                    )
                    y_pos += 22 # Add space after 'Other classes'
            else:
                y_pos += 20
                self.preview_canvas.create_text(
                    canvas_width // 2, y_pos,
                    text="No detection data collected",
                    fill="#BBBBBB", font=("Arial", 11), # Lighter gray
                    anchor=tk.CENTER
                )
                y_pos += 22 # Add space even if no detections
            # Add a footer with prompt to continue (adjust color)
            footer_color = "#77AADD" # Same as header
            self.preview_canvas.create_text(
                canvas_width // 2, canvas_height - 30,
                text="Processing complete. Select another video or run evaluation.",
                fill=footer_color, font=("Arial", 10),
                anchor=tk.CENTER
            )
        except Exception as e:
            print(f"Error displaying completion stats: {e}")
            self.show_preview_message(f"Error displaying completion statistics:\n{str(e)}")
            
    def show_evaluation_results(self, results):
        """Show a summary of model evaluation results on the canvas"""
        try:
            # Clear canvas
            self.preview_canvas.delete("all")
            if not results:
                self.show_preview_message("No evaluation results available.")
                return
            # Calculate canvas dimensions for layout
            canvas_width = self.preview_canvas.winfo_width() or 600 # Use larger default
            canvas_height = self.preview_canvas.winfo_height() or 500 # Use larger default
            # Background rectangle
            bg_color = "#383838"
            self.preview_canvas.create_rectangle(
                10, 10, canvas_width - 10, canvas_height - 10,
                fill=bg_color, outline="#555555", width=1
            )
            # Header
            header_text = "Model Evaluation Results"
            header_color = "#77AADD"
            self.preview_canvas.create_text(
                canvas_width // 2, 30,
                text=header_text,
                fill=header_color, font=("Arial", 16, "bold"),
                anchor=tk.CENTER
            )
            # Prepare data for display
            model_types = list(results.keys())
            metrics_to_display = {
                "fps": "FPS",
                "mean_inference_time": "Avg Inference (s)",
                "total_detections": "Total Detections",
                "unique_classes_detected": "Unique Classes",
                "AP_IoU=0.50:0.95": "mAP [0.5:0.95]",
                "AP_IoU=0.50": "AP@0.50",
                "successful_inferences": "Success %" # Note: This reflects processing success, not detection accuracy.
            }
            data = {"Metric": list(metrics_to_display.values())}
            for model in model_types:
                model_data = []
                metrics = results.get(model, {})
                coco_metrics = metrics.get("coco_metrics", {}) or {} # Handle None
                total_images = metrics.get("total_images", 1) # Avoid division by zero
                for key, display_name in metrics_to_display.items():
                    value = None
                    if key in metrics:
                        value = metrics[key]
                    elif key in coco_metrics:
                        value = coco_metrics[key]
                    # Format values
                    if value is None:
                        formatted_value = "N/A"
                    elif key == "successful_inferences":
                        # Clarification: This metric indicates the percentage of images 
                        # processed without runtime errors, not detection accuracy.
                        success_count = metrics.get("successful_inferences", 0)
                        formatted_value = f"{(success_count / total_images * 100):.1f}%" if total_images > 0 else "0.0%"
                    elif isinstance(value, float):
                        if key == "fps":
                            formatted_value = f"{value:.1f}"
                        elif "time" in key:
                            formatted_value = f"{value:.3f}"
                        else: # mAP/AP
                            formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = str(value)
                    model_data.append(formatted_value)
                data[model.upper()] = model_data
            # Use pandas to create a string representation of the table
            try:
                df = pd.DataFrame(data)
                table_str = df.to_string(index=False, justify='center')
            except ImportError:
                # Fallback if pandas is not available (less neat)
                table_str = "Pandas not found. Cannot display table.\n\n"
                for model in model_types:
                    table_str += f"--- {model.upper()} ---\n"
                    for key, display_name in metrics_to_display.items():
                         value = results.get(model, {}).get(key) or \
                                 (results.get(model, {}).get("coco_metrics", {}) or {}).get(key)
                         table_str += f"{display_name}: {value}\n"
                    table_str += "\n"
            # Display the table string on the canvas
            # Use a fixed-width font for better table alignment
            self.preview_canvas.create_text(
                canvas_width // 2,
                70, # Start below header
                text=table_str,
                fill="#FFFFFF", font=("Courier New", 10), # Fixed-width font
                anchor=tk.N, # Anchor to top-center
                justify=tk.LEFT # Left justify lines
            )
            # Add footer
            footer_color = "#AAAAAA"
            self.preview_canvas.create_text(
                canvas_width // 2, canvas_height - 20,
                text="Evaluation complete. Detailed reports in 'inference/results'.",
                fill=footer_color, font=("Arial", 10),
                anchor=tk.CENTER
            )
        except Exception as e:
            print(f"Error displaying evaluation results: {e}")
            self.show_preview_message(f"Error displaying evaluation results:\n{str(e)}")
            
    def start_webcam_detection(self):
        """Start or stop real-time detection using the webcam."""
        if self.processing_active:
            # If already active, stop the webcam detection
            self.stop_webcam_detection()
            return
        try:
            # Get model type
            model_type = self.model_type_var.get()
            # Get model path (custom or default)
            if self.show_advanced_var.get() and self.custom_model_path.get():
                model_path = self.custom_model_path.get()
            else:
                model_path = DEFAULT_MODELS.get(model_type)
            if not model_path:
                messagebox.showerror("Error", "No model selected or available.")
                return
            # Initialize the model manager
            self.status_var.set(f"Initializing {model_type} model...")
            self.root.update()
            model_manager = ModelManager(model_type, model_path)
            # Initialize statistics tracking and reset flags
            self.frame_count = 0
            self.processing_time = 0.0
            self.detection_counts = {}
            self.total_frames = 0
            self.stop_preview = False
            # Open webcam here to ensure we can access it before starting the thread
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                messagebox.showerror("Error", "Cannot access webcam. Please check connections and permissions.")
                return
            # Start webcam processing in a separate thread
            self.processing_active = True
            self.video_thread = threading.Thread(
                target=self._process_webcam_thread,
                args=(model_manager,),
                daemon=True
            )
            self.video_thread.start()
            # Update the button text to "Stop Camera"
            self.webcam_button.config(text="Stop Camera")
        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("Runtime Error", f"An error occurred during webcam initialization:\n{e}")
            self.processing_active = False
            if hasattr(self, 'webcam') and self.webcam is not None:
                self.webcam.release()
                self.webcam = None
                
    def stop_webcam_detection(self):
        """Stop the real-time webcam detection and display statistics."""
        if not self.processing_active:
            return
        self.stop_preview = True  # Signal the thread to stop
        # Wait for the thread to stop without blocking the main thread
        def wait_for_thread():
            if self.video_thread and self.video_thread.is_alive():
                self.root.after(100, wait_for_thread)  # Check again after 100ms
            else:
                self.processing_active = False
                self.status_var.set("Webcam detection stopped.")
                # Update the button text back to "Start Camera"
                self.webcam_button.config(text="Start Camera")
                # Display statistics after stopping the webcam
                stats = {
                    "frames_processed": self.frame_count,
                    "total_frames": getattr(self, 'total_frames', self.frame_count),  # Default to frame_count if not set
                    "processing_time": self.processing_time,
                    "actual_fps": self.frame_count / self.processing_time if self.processing_time > 0 else 0,
                    "detections": self.detection_counts,
                    "model_type": self.model_type_var.get(),
                    "model_device": "cpu"  # Assuming CPU for now; update if GPU is used
                }
                self.show_completion_stats(stats)
        wait_for_thread()
        
    def _toggle_webcam_detection(self):
        """Toggle webcam detection on and off."""
        if self.processing_active:
            self.stop_webcam_detection()
        else:
            self.start_webcam_detection()
        
    def _process_webcam_thread(self, model_manager):
        """Thread function to process webcam frames in real-time with optimizations."""
        try:
            # Use the already opened webcam object if available
            if not hasattr(self, 'webcam') or self.webcam is None or not self.webcam.isOpened():
                 self.webcam = cv2.VideoCapture(0) # Open the default webcam
                 if not self.webcam.isOpened():
                     self.show_preview_message("Cannot access webcam")
                     self.processing_active = False # Ensure state is reset
                     self.root.after(0, lambda: self.webcam_button.config(text="Start Camera")) # Reset button
                     return
            cap = self.webcam # Use the instance variable
            # Get webcam properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # --- Performance Optimizations ---
            # 1. Processing Resolution (already implemented via resize)
            process_width = 640
            process_height = 480
            # 2. Frame Skipping (already implemented)
            frame_skip = 2  # Process every Nth frame
            # 3. UI Update Rate Limitations ---
            webcam_fps = cap.get(cv2.CAP_PROP_FPS) or 30 # Estimate if needed
            target_display_fps = min(15, webcam_fps / frame_skip) # Limit display FPS
            frame_interval_ms = int(1000 / target_display_fps) if target_display_fps > 0 else 100 # Min 100ms interval
            # --- End Optimizations ---
            # Calculate resize ratio for display
            canvas_width = self.preview_canvas.winfo_width() or 400
            canvas_height = self.preview_canvas.winfo_height() or 300
            display_ratio = min(canvas_width / process_width, canvas_height / process_height)
            display_width = int(process_width * display_ratio)
            display_height = int(process_height * display_ratio)
            # Reset statistics
            self.frame_count = 0 # Processed frames
            self.total_frames = 0 # Total frames read
            self.detection_counts = {}
            start_time = time.time()
            last_update_time = 0
            processing_errors = 0
            model_device = getattr(model_manager, 'device', 'unknown')
            print(f"Starting webcam processing with {model_manager.model_type} on {model_device}")
            while not self.stop_preview:
                ret, frame = cap.read()
                if not ret:
                    print("Webcam stream ended or error reading frame.")
                    break # Exit loop if frame read fails
                self.total_frames += 1  # Count total frames read
                # Apply frame skipping
                if self.total_frames % frame_skip != 0:
                    continue  # Skip frames
                self.frame_count += 1  # Count processed frames
                current_time = int(time.time() * 1000)
                # Update status periodically
                if self.frame_count % 10 == 0:
                     self.root.after(0, lambda count=self.frame_count: 
                                    self.status_var.set(f"Webcam processing frame {count}"))
                try:
                    # Resize frame for processing
                    frame_resized_for_processing = cv2.resize(frame, (process_width, process_height))
                    # Perform prediction and get annotated frame directly
                    detections, segmentations, annotated_frame = model_manager.predict(frame_resized_for_processing)
                    # Update detection counts (optional for webcam, but good practice)
                    for det in detections:
                        class_name = det.get('class_name', 'Unknown')
                        if class_name != 'Unknown':
                            self.detection_counts[class_name] = self.detection_counts.get(class_name, 0) + 1
                    # Limit display update rate
                    if (current_time - last_update_time) >= frame_interval_ms:
                        # Resize annotated frame for display
                        frame_resized_for_display = cv2.resize(annotated_frame, (display_width, display_height))
                        frame_rgb = cv2.cvtColor(frame_resized_for_display, cv2.COLOR_BGR2RGB)
                        # Update canvas on main thread
                        pil_img = Image.fromarray(frame_rgb)
                        # Use _update_video_canvas for consistency
                        self.root.after(0, self._update_video_canvas, 
                                       pil_img, self.frame_count, self.total_frames, model_manager.model_type) 
                        last_update_time = current_time
                except Exception as e:
                    processing_errors += 1
                    print(f"Error processing webcam frame {self.frame_count}: {e}")
                    # Don't show message box for webcam errors to avoid spamming
                    if processing_errors > 10 and processing_errors % 50 == 0: # Log occasionally
                         print("Multiple webcam processing errors occurred.")
                    # Continue processing next frame
            # Calculate total processing time
            self.processing_time = time.time() - start_time
            # Release webcam ONLY if we opened it in this thread (or handle externally)
            # The current logic in start/stop suggests releasing it in stop_webcam_detection
            # cap.release() # Avoid releasing here if managed externally
            print(f"Webcam processing stopped. Processed {self.frame_count} frames.")
            # Statistics are shown by stop_webcam_detection after the thread finishes
        except Exception as e:
            print(f"Fatal error in webcam processing thread: {e}")
            traceback.print_exc()
            self.root.after(0, lambda err=str(e): self.show_preview_message(f"Webcam Error:\n{err}"))
        finally:
            # Ensure state is reset even if errors occur
            self.processing_active = False
            # Let stop_webcam_detection handle button reset and stats display
            
    def check_gpu_availability(self):
        """Check if GPU is available and update the status with detailed information."""
        try:
            import torch
            import platform
            
            selected_device = self.gpu_var.get()
            
            # Get CPU information regardless of selected device
            cpu_info = f"CPU: {platform.processor()}"
            if not cpu_info or "unknown" in cpu_info.lower():
                # Try alternate methods to get CPU info
                try:
                    import cpuinfo
                    cpu_info = f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}"
                except:
                    if platform.system() == "Linux":
                        try:
                            with open('/proc/cpuinfo', 'r') as f:
                                for line in f:
                                    if line.startswith('model name'):
                                        cpu_info = f"CPU: {line.split(':', 1)[1].strip()}"
                                        break
                        except:
                            cpu_info = f"CPU: {platform.machine()}"
                    else:
                        cpu_info = f"CPU: {platform.machine()}"
            
            # First check YOLO detection logs
            # If we've seen YOLO timing logs, we know GPU is working even if PyTorch doesn't detect it
            import os
            yolo_gpu_detected = False
            gpu_name = "GPU"
            
            # Check for NVIDIA driver and GPU name
            try:
                import subprocess
                nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                           capture_output=True, text=True, timeout=2)
                if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
                    gpu_name = nvidia_smi.stdout.strip()
                    print(f"NVIDIA GPU detected via nvidia-smi: {gpu_name}")
                    yolo_gpu_detected = True
            except Exception as e:
                print(f"No NVIDIA GPU detected via nvidia-smi: {e}")
                
                # Try AMD GPU detection if NVIDIA fails
                try:
                    rocm_smi = subprocess.run(['rocm-smi', '--showproductname'], 
                                            capture_output=True, text=True, timeout=2)
                    if rocm_smi.returncode == 0 and "GPU" in rocm_smi.stdout:
                        for line in rocm_smi.stdout.splitlines():
                            if "GPU" in line and ":" in line:
                                gpu_name = line.split(":", 1)[1].strip()
                                print(f"AMD GPU detected via rocm-smi: {gpu_name}")
                                yolo_gpu_detected = True
                                break
                except Exception:
                    pass
            
            # Get CUDA availability info from PyTorch
            cuda_available = torch.cuda.is_available()
            print(f"PyTorch CUDA availability: {cuda_available}")
            
            # Force a specific device if requested
            if selected_device == "cpu":
                # CPU mode
                self.gpu_status_var.set(cpu_info)
                print(f"Using CPU mode: {cpu_info}")
                return
                
            if selected_device == "cuda":
                if cuda_available or yolo_gpu_detected:
                    # CUDA is available through PyTorch or YOLO with our GPU
                    if cuda_available:
                        # Get GPU name from PyTorch
                        device_count = torch.cuda.device_count()
                        current_device = torch.cuda.current_device()
                        device_name = torch.cuda.get_device_name(current_device)
                        self.gpu_status_var.set(f"GPU: {device_name}")
                    else:
                        # Using GPU through YOLO but not recognized by PyTorch
                        self.gpu_status_var.set(f"GPU: {gpu_name} (YOLO accelerated)")
                    return
                else:
                    # CUDA requested but truly not available
                    error_msg = f"CUDA requested but not available - using {cpu_info}"
                    self.gpu_status_var.set(error_msg)
                    print("CUDA requested but not available")
                    return
            
            # Auto mode - prefer GPU if available through any means
            if cuda_available:
                # PyTorch detects CUDA
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                self.gpu_status_var.set(f"GPU: {device_name} (auto-selected)")
            elif yolo_gpu_detected:
                # PyTorch doesn't detect CUDA but YOLO can use the GPU
                self.gpu_status_var.set(f"GPU: {gpu_name} (YOLO accelerated)")
            else:
                # No GPU available through any means, use CPU
                self.gpu_status_var.set(f"{cpu_info} (auto-selected)")
                
        except ImportError as e:
            self.gpu_status_var.set(f"PyTorch not installed: {str(e)[:50]}")
            print("PyTorch is not installed - GPU acceleration unavailable")
        except Exception as e:
            self.gpu_status_var.set(f"Error checking GPU: {str(e)[:50]}")
            print(f"Error checking GPU availability: {e}")
            
    def show_preview_message(self, message):
        """Display a message in the preview panel (helper for other methods)"""
        self._display_message_on_canvas(message)
        
    def _on_gpu_change(self, event):
        """Handle GPU selection change."""
        selected_device = self.gpu_var.get()
        if selected_device == "auto":
            print("Setting device selection to automatic (PyTorch default)")
            os.environ.pop("CV_FORCE_DEVICE", None)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif selected_device == "cpu":
            print("Forcing CPU usage")
            os.environ["CV_FORCE_DEVICE"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
        elif selected_device == "cuda":
            print("Forcing CUDA GPU usage if available")
            os.environ["CV_FORCE_DEVICE"] = "cuda"
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # Enable all GPUs
        self.status_var.set(f"Device set to: {selected_device}")
        # Re-check GPU to show updated status
        self.check_gpu_availability()
        
    def _update_model_info(self):
        """Update the model information text based on current selection."""
        model_type = self.model_type_var.get()
        if model_type in self.model_requirements:
            req = self.model_requirements[model_type]
            self.model_info_var.set(
                f"{model_type.upper()}: {req['description']}\n\n"
                f"Minimum Requirements:\n"
                f"• RAM: {req['ram']}\n"
                f"• VRAM: {req['vram']}\n"
                f"• CPU: {req['cpu_cores']}\n"
                f"• PyTorch: {req['pytorch']}"
            )
        else:
            # Provide fallback description
            self.model_info_var.set(f"{model_type.upper()}: Model type info not available")
        
        # Update default model path for advanced settings (if implemented)
        if hasattr(self, 'show_advanced_var') and self.show_advanced_var.get():
            if hasattr(self, 'custom_model_path'):
                self.custom_model_path.set(DEFAULT_MODELS.get(model_type, ""))
        
# Removed validate_model function as initialization handles this now

def main():
    """
    Main function that processes command line arguments or launches GUI
    """
    parser = argparse.ArgumentParser(description="Computer Vision Project")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode")
    # parser.add_argument("--terminal-ui", action="store_true", help="Run with terminal-based UI (Currently disabled)") # Disabled TUI
    parser.add_argument("--model-type", type=str, choices=["faster-rcnn", "yolo-seg", "rtdetr"], 
                          help="Type of model to use (required for CLI)")
    parser.add_argument("--model-path", type=str, help="Path to the model weights file (optional for CLI, uses default)")
    # parser.add_argument("--config-path", type=str, help="Path to the configuration file (if required)") # Config path not used
    parser.add_argument("--video-path", type=str, help="Path to the input video file (required for CLI)")
    args = parser.parse_args()
    # Command-line mode
    if args.cli:
        if not args.model_type or not args.video_path:
            parser.error("--cli requires --model-type and --video-path arguments.")
        # Use default model if not specified, otherwise use provided path
        model_path = args.model_path or DEFAULT_MODELS.get(args.model_type)
        if not model_path:
             print(f"Error: Default model path not found for type '{args.model_type}'. Please specify --model-path.")
             sys.exit(1)
        if not os.path.exists(args.video_path):
             print(f"Error: Video file not found: {args.video_path}")
             sys.exit(1)
        # Config path not currently used
        config_path = None # args.config_path
        print(f"Running CLI mode with Model: {args.model_type}, Path: {model_path}, Video: {args.video_path}")
        try:
            model_manager = ModelManager(args.model_type, model_path, config_path)
            if model_manager.model_wrapper is None or not hasattr(model_manager.model_wrapper, 'model') or model_manager.model_wrapper.model is None:
                 print(f"Error: Failed to initialize model {args.model_type} from {model_path}")
                 sys.exit(1)
            display_video_with_model(args.video_path, model_manager)
        except Exception as e:
            print(f"An error occurred during CLI execution:")
            print(traceback.format_exc())
            sys.exit(1)
    # Terminal UI mode (Currently disabled)
    # elif args.terminal_ui:
    #     try:
    #         video_path, model_path = terminal_gui()
    #         if video_path and model_path:
    #             # Determine model type based on path? Or default to yolo?
    #             # For now, assume yolo if using TUI default models
    #             model_type = 'yolo' # Default assumption
    #             if 'rtdetr' in model_path: model_type = 'rtdetr'
    #             elif 'sam' in model_path: model_type = 'sam'
    #             model_manager = ModelManager(model_type, model_path)
    #             display_video_with_model(video_path, model_manager)
    #     except Exception as e:
    #         print(f"Error running terminal UI: {e}")
    #         print("Falling back to graphical UI")
    #         gui = GraphicalGUI()
    #         gui.run()
    # GUI mode (default)
    else:
        # Check if display is available (basic check for headless environments)
        try:
            # Test with ThemedTk instead of tk.Tk
            root = ThemedTk(theme="arc") 
            root.withdraw() # Don't show the root window immediately
            root.destroy() # Clean up
        except tk.TclError as e:
             print("Error: Could not initialize Tkinter display.")
             print("If running in a headless environment, please use the --cli option.")
             print(f"(Tkinter error: {e})")
             sys.exit(1)
        gui = GraphicalGUI()
        gui.run()
        
if __name__ == "__main__":
    # Ensure the models/pts directory exists
    os.makedirs("models/pts", exist_ok=True)
    main()