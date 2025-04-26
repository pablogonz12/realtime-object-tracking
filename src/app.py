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
import time  # Import time module for FPS control and timing
from pathlib import Path
import shutil # Import shutil for directory deletion

# Add project root to PYTHONPATH so sibling packages (data_sets, inference) are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import traceback  # Import traceback for detailed error logging
import threading  # Import threading for video preview
from PIL import Image, ImageTk  # Import PIL for image processing
from ttkthemes import ThemedTk # Import ThemedTk
import tkinter.simpledialog as simpledialog  # for input dialogs
from data_sets.dataset_manager import DatasetManager
from src.evaluate_models import ModelEvaluator, COCO_VAL_TOTAL_IMAGES
from src.metrics_visualizer import MetricsVisualizer  # Import MetricsVisualizer
import pandas as pd # Import pandas for displaying results table

# Import model wrappers directly from consolidated models file
from src.models import ModelManager # Removed unused YOLOWrapper, RTDETRWrapper, SAMWrapper imports

# Default model paths for auto-download
DEFAULT_MODELS = {
    'yolov8n-seg': 'models/pts/yolov8n-seg.pt',
    'yolov8s-seg': 'models/pts/yolov8s-seg.pt',
    'yolov8m-seg': 'models/pts/yolov8m-seg.pt',
    'yolov8l-seg': 'models/pts/yolov8l-seg.pt',
    'yolov8x-seg': 'models/pts/yolov8x-seg.pt',
    'yolo11n-seg': 'models/pts/yolo11n-seg.pt',
    'yolo11s-seg': 'models/pts/yolo11s-seg.pt',
    'yolo11m-seg': 'models/pts/yolo11m-seg.pt',
    'yolo11l-seg': 'models/pts/yolo11l-seg.pt',
    'yolo11x-seg': 'models/pts/yolo11x-seg.pt',
    'yoloe-11s-seg': 'models/pts/yoloe-11s-seg.pt',
    'yoloe-11m-seg': 'models/pts/yoloe-11m-seg.pt',
    'yoloe-11l-seg': 'models/pts/yoloe-11l-seg.pt',
    'yoloe-v8s-seg': 'models/pts/yoloe-v8s-seg.pt',
    'yoloe-v8m-seg': 'models/pts/yoloe-v8m-seg.pt',
    'yoloe-v8l-seg': 'models/pts/yoloe-v8l-seg.pt',
    'yolov9c-seg': 'models/pts/yolov9c-seg.pt',
    'yolov9e-seg': 'models/pts/yolov9e-seg.pt'
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
            print(f"  Model type is {model_manager.model_type}. Calling model_manager.predict...")
            # For all our models (expecting detections, segmentations, annotated_frame)
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
    Terminal-based GUI for selecting video and model (Currently Disabled)

    Returns:
        tuple: (selected_video_path, selected_model_path)
    """
    # This function is currently disabled as per user request/code state.
    # If re-enabled, ensure curses is imported and handled correctly.
    print("Terminal UI is currently disabled.")
    return None, None
    # ... (original curses code removed for brevity) ...

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
        # self.preview_img = None # Removed, canvas handles images

        # --- State Variables ---
        self.video_thread = None  # Will hold the thread for video processing
        self.stop_preview = False  # Flag to stop preview threads
        self.pause_processing = False  # Flag to pause processing
        self.processing_active = False  # Flag to track if processing is currently active
        self.stop_event = threading.Event() # Event to signal stopping
        self.pause_event = threading.Event() # Event to signal pausing

        # Initialize statistics attributes
        self.frame_count = 0
        self.processing_time = 0.0
        self.detection_counts = {}

        # Define model requirements - moved this BEFORE using it in _update_model_info
        self.model_requirements = {
            "mask-rcnn": {
                "description": "Instance segmentation with ResNet50 FPN backbone (TorchVision)",
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
            }
            # Removed rtdetr-ins and sam requirements if they existed
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
        # Store reference to the dropdown
        self.model_type_dropdown = ttk.Combobox(model_group, textvariable=self.model_type_var,
                                                values=list(DEFAULT_MODELS.keys()), state="readonly", width=15)
        self.model_type_dropdown.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.model_type_dropdown.bind("<<ComboboxSelected>>", self._on_model_type_change)

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
        self.pause_button = ttk.Button(self.processing_buttons_frame, text="Pause", command=self.pause_processing)
        self.stop_button = ttk.Button(self.processing_buttons_frame, text="Stop", command=self.stop_processing)

        self._update_buttons(processing=False)

    def stop_processing(self):
        """Stop video processing"""
        self.stop_preview = True # Signal preview loops to stop
        self.pause_processing = False # Ensure pause is off

        # Signal the stop event to stop the video processing thread
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        # Clear pause event if set
        if hasattr(self, 'pause_event'):
            self.pause_event.clear()

        self.status_var.set("Processing stopped")
        # Reset processing state
        self.processing_active = False
        # Update buttons back to normal
        self._update_buttons(False)

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
        dataset_menu.add_command(label="Delete Generated Data", command=self.delete_generated_data)
        dataset_menu.add_command(label="Delete Downloaded Models", command=self.delete_downloaded_models)
        menubar.add_cascade(label="Dataset", menu=dataset_menu)

        # Evaluation menu
        eval_menu = tk.Menu(menubar, tearoff=0)
        eval_menu.add_command(label="Run Evaluation...", command=self.run_evaluation_gui)
        menubar.add_cascade(label="Evaluate", menu=eval_menu)

        # Metrics Visualization menu
        metrics_menu = tk.Menu(menubar, tearoff=0)
        metrics_menu.add_command(label="Show Performance Dashboard", command=self.show_metrics_dashboard)
        metrics_menu.add_command(label="Show Precision-Recall Curve", command=self.show_precision_recall_curve)
        metrics_menu.add_command(label="Show Reliability Analysis", command=self.show_reliability_analysis)
        metrics_menu.add_command(label="Show Performance Trends", command=self.show_performance_trends)
        metrics_menu.add_separator()
        metrics_menu.add_command(label="Run Best Model on Test Video", command=self.run_best_model_on_test)
        metrics_menu.add_separator()
        metrics_menu.add_command(label="Generate Comprehensive Report", command=self.generate_metrics_report)
        menubar.add_cascade(label="Metrics", menu=metrics_menu)

        # Attach to root
        self.root.config(menu=menubar)

    def download_coco_full(self):
        """Download the full COCO val2017 dataset with loading bar"""
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Downloading Dataset")
        progress_dialog.geometry("500x250")
        progress_dialog.transient(self.root)  # Make dialog modal
        progress_dialog.grab_set()  # Prevent interaction with main window
        progress_dialog.resizable(False, False)
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        # Center the dialog on the main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (500 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (250 // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Configure the grid
        progress_dialog.columnconfigure(0, weight=1)
        progress_dialog.rowconfigure(3, weight=1)
        
        # Dialog header
        header_label = ttk.Label(
            progress_dialog, 
            text="Downloading COCO Dataset", 
            font=("Arial", 14, "bold"),
            padding=(0, 10, 0, 20)
        )
        header_label.grid(row=0, column=0, sticky="ew")
        
        # Current operation label
        current_op_var = tk.StringVar(value="Initializing...")
        current_op_label = ttk.Label(
            progress_dialog, 
            textvariable=current_op_var,
            font=("Arial", 10),
            padding=(0, 0, 0, 5)
        )
        current_op_label.grid(row=1, column=0, sticky="ew")
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            progress_dialog,
            variable=progress_var,
            maximum=100,
            length=450,
            mode='determinate'
        )
        progress_bar.grid(row=2, column=0, padx=25, pady=10, sticky="ew")
        
        # Details text widget with scrollbar
        details_frame = ttk.Frame(progress_dialog)
        details_frame.grid(row=3, column=0, sticky="nsew", padx=25, pady=(10, 20))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        details_scroll = ttk.Scrollbar(details_frame)
        details_scroll.grid(row=0, column=1, sticky="ns")
        
        details_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            width=50,
            height=6,
            yscrollcommand=details_scroll.set,
            font=("Consolas", 9)
        )
        details_text.grid(row=0, column=0, sticky="nsew")
        details_scroll.config(command=self.file_listbox.yview)
        
        # Make text widget read-only but still allow copy
        details_text.configure(state="disabled")
        
        # Define the progress callback function
        def update_progress(stage, stage_progress, overall_progress, message):
            # Update on the main thread
            self.root.after(0, lambda: _update_ui(stage, stage_progress, overall_progress, message))
        
        def _update_ui(stage, stage_progress, overall_progress, message):
            # Update progress bar
            progress_var.set(overall_progress * 100)
            
            # Update operation label
            current_op_var.set(message)
            
            # Append to log
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"{message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Force update the UI
            progress_dialog.update_idletasks()
            
        # Add initial message to log
        details_text.configure(state="normal")
        details_text.insert(tk.END, "Starting download of COCO dataset...\n")
        details_text.configure(state="disabled")
        
        # Define a function for running the download in a separate thread
        def download_thread():
            try:
                dm = DatasetManager(progress_callback=update_progress)
                dm.setup_coco(None)
                
                # Show completion on the main thread
                self.root.after(0, lambda: _show_completion())
                
            except Exception as e:
                # Show error on the main thread
                error_message = f"Error: {str(e)}"
                self.root.after(0, lambda: _show_error(error_message))
        
        def _show_completion():
            # Update status
            self.status_var.set("Full COCO dataset setup complete.")
            
            # Enable dialog closing and add a complete message
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            details_text.configure(state="normal")
            details_text.insert(tk.END, "\n✓ Download and setup complete!\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        def _show_error(error_message):
            # Update status
            self.status_var.set(f"Error: {error_message}")
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Show error in details
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n❌ Error: {error_message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        # Start the download thread
        threading.Thread(target=download_thread, daemon=True).start()

    def download_coco_subset(self):
        """Prompt for subset size and download COCO subset with loading bar"""
        size = simpledialog.askinteger("COCO Subset", "Enter number of images for subset:", 
                                     minvalue=1, maxvalue=5000, initialvalue=100)
        if not size:
            return
            
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Downloading Dataset Subset")
        progress_dialog.geometry("500x250")
        progress_dialog.transient(self.root)  # Make dialog modal
        progress_dialog.grab_set()  # Prevent interaction with main window
        progress_dialog.resizable(False, False)
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        # Center the dialog on the main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (500 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (250 // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Configure the grid
        progress_dialog.columnconfigure(0, weight=1)
        progress_dialog.rowconfigure(3, weight=1)
        
        # Dialog header
        header_label = ttk.Label(
            progress_dialog, 
            text=f"Downloading COCO Subset ({size} images)", 
            font=("Arial", 14, "bold"),
            padding=(0, 10, 0, 20)
        )
        header_label.grid(row=0, column=0, sticky="ew")
        
        # Current operation label
        current_op_var = tk.StringVar(value="Initializing...")
        current_op_label = ttk.Label(
            progress_dialog, 
            textvariable=current_op_var,
            font=("Arial", 10),
            padding=(0, 0, 0, 5)
        )
        current_op_label.grid(row=1, column=0, sticky="ew")
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            progress_dialog,
            variable=progress_var,
            maximum=100,
            length=450,
            mode='determinate'
        )
        progress_bar.grid(row=2, column=0, padx=25, pady=10, sticky="ew")
        
        # Details text widget with scrollbar
        details_frame = ttk.Frame(progress_dialog)
        details_frame.grid(row=3, column=0, sticky="nsew", padx=25, pady=(10, 20))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        details_scroll = ttk.Scrollbar(details_frame)
        details_scroll.grid(row=0, column=1, sticky="ns")
        
        details_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            width=50,
            height=6,
            yscrollcommand=details_scroll.set,
            font=("Consolas", 9)
        )
        details_text.grid(row=0, column=0, sticky="nsew")
        details_scroll.config(command=details_text.yview)
        
        # Make text widget read-only but still allow copy
        details_text.configure(state="disabled")
        
        # Define the progress callback function
        def update_progress(stage, stage_progress, overall_progress, message):
            # Update on the main thread
            self.root.after(0, lambda: _update_ui(stage, stage_progress, overall_progress, message))
        
        def _update_ui(stage, stage_progress, overall_progress, message):
            # Update progress bar
            progress_var.set(overall_progress * 100)
            
            # Update operation label
            current_op_var.set(message)
            
            # Append to log
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"{message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Force update the UI
            progress_dialog.update_idletasks()
            
        # Add initial message to log
        details_text.configure(state="normal")
        details_text.insert(tk.END, f"Starting download of COCO subset with {size} images...\n")
        details_text.configure(state="disabled")
        
        # Define a function for running the download in a separate thread
        def download_thread():
            try:
                dm = DatasetManager(progress_callback=update_progress)
                dm.setup_coco(subset_size=size)
                
                # Show completion on the main thread
                self.root.after(0, lambda: _show_completion())
                
            except Exception as e:
                # Show error on the main thread
                error_message = f"Error: {str(e)}"
                self.root.after(0, lambda: _show_error(error_message))
        
        def _show_completion():
            # Update status
            self.status_var.set(f"COCO subset with {size} images setup complete.")
            
            # Enable dialog closing and add a complete message
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n✓ Subset with {size} images ready!\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        def _show_error(error_message):
            # Update status
            self.status_var.set(f"Error: {error_message}")
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Show error in details
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n❌ Error: {error_message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        # Start the download thread
        threading.Thread(target=download_thread, daemon=True).start()

    def decompress_datasets(self):
        """Decompress datasets with loading bar"""
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Decompressing Dataset")
        progress_dialog.geometry("500x250")
        progress_dialog.transient(self.root)  # Make dialog modal
        progress_dialog.grab_set()  # Prevent interaction with main window
        progress_dialog.resizable(False, False)
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        # Center the dialog on the main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (500 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (250 // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Configure the grid
        progress_dialog.columnconfigure(0, weight=1)
        progress_dialog.rowconfigure(3, weight=1)
        
        # Dialog header
        header_label = ttk.Label(
            progress_dialog, 
            text="Decompressing Dataset Files", 
            font=("Arial", 14, "bold"),
            padding=(0, 10, 0, 20)
        )
        header_label.grid(row=0, column=0, sticky="ew")
        
        # Current operation label
        current_op_var = tk.StringVar(value="Initializing...")
        current_op_label = ttk.Label(
            progress_dialog, 
            textvariable=current_op_var,
            font=("Arial", 10),
            padding=(0, 0, 0, 5)
        )
        current_op_label.grid(row=1, column=0, sticky="ew")
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            progress_dialog,
            variable=progress_var,
            maximum=100,
            length=450,
            mode='determinate'
        )
        progress_bar.grid(row=2, column=0, padx=25, pady=10, sticky="ew")
        
        # Details text widget with scrollbar
        details_frame = ttk.Frame(progress_dialog)
        details_frame.grid(row=3, column=0, sticky="nsew", padx=25, pady=(10, 20))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        details_scroll = ttk.Scrollbar(details_frame)
        details_scroll.grid(row=0, column=1, sticky="ns")
        
        details_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            width=50,
            height=6,
            yscrollcommand=details_scroll.set,
            font=("Consolas", 9)
        )
        details_text.grid(row=0, column=0, sticky="nsew")
        details_scroll.config(command=details_text.yview)
        
        # Make text widget read-only but still allow copy
        details_text.configure(state="disabled")
        
        # Add initial message to log
        details_text.configure(state="normal")
        details_text.insert(tk.END, "Starting decompression of dataset files...\n")
        details_text.configure(state="disabled")
        
        # Keep track of extraction steps
        extraction_steps = {
            "annotations": {"done": False, "weight": 0.3},
            "images": {"done": False, "weight": 0.7}
        }
        total_progress = 0.0
        
        # Define a thread to monitor the process
        def decompress_thread():
            try:
                # Get paths to check for compressed files
                dm = DatasetManager()
                img_compressed_file = dm.coco_dir / "val2017_compressed.zip"
                ann_compressed_file = dm.coco_dir / "annotations_compressed.zip"
                
                # Check if files exist
                if not img_compressed_file.exists() and not ann_compressed_file.exists():
                    update_ui("No compressed dataset files found.", 1.0)
                    _show_completion("No compressed files found.")
                    return
                
                # Run decompression process
                if ann_compressed_file.exists():
                    update_ui("Decompressing annotation files...", 0.1)
                    
                    # Create dummy progress updates as we're not using the updated extract method
                    for i in range(1, 11):
                        time.sleep(0.2)  # Simulate progress
                        update_ui(f"Decompressing annotations: {i*10}%", 
                                    (i/10) * extraction_steps["annotations"]["weight"])
                    
                    extraction_steps["annotations"]["done"] = True
                    total_progress = extraction_steps["annotations"]["weight"]
                
                if img_compressed_file.exists():
                    update_ui("Decompressing image files...", total_progress + 0.05)
                    
                    # Create dummy progress updates
                    for i in range(1, 11):
                        time.sleep(0.3)  # Simulate progress (takes longer)
                        done_weight = extraction_steps["annotations"]["weight"] if extraction_steps["annotations"]["done"] else 0
                        progress = done_weight + ((i/10) * extraction_steps["images"]["weight"])
                        update_ui(f"Decompressing images: {i*10}%", progress)
                    
                    extraction_steps["images"]["done"] = True
                
                # Actually run the decompression
                dm.decompress_datasets()
                
                # Final update
                update_ui("Decompression completed successfully.", 1.0)
                _show_completion("Datasets decompressed successfully.")
                
            except Exception as e:
                error_message = f"Error during decompression: {str(e)}"
                update_ui(error_message, 0.0)
                _show_error(error_message)
        
        def update_ui(message, progress):
            # Update on the main thread
            self.root.after(0, lambda m=message, p=progress: _update_ui_internal(m, p))
        
        def _update_ui_internal(message, progress):
            # Update progress bar
            progress_var.set(progress * 100)
            
            # Update operation label
            current_op_var.set(message)
            
            # Append to log
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"{message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Force update the UI
            progress_dialog.update_idletasks()
        
        def _show_completion(message="Decompression complete."):
            # Update status
            self.status_var.set(message)
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Add completion message
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n✓ {message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        def _show_error(error_message):
            # Update status
            self.status_var.set(f"Error: {error_message}")
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Show error in details
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n❌ Error: {error_message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        # Start the decompression thread
        threading.Thread(target=decompress_thread, daemon=True).start()

    def compress_datasets(self):
        """Compress datasets with loading bar"""
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Compressing Dataset")
        progress_dialog.geometry("500x250")
        progress_dialog.transient(self.root)  # Make dialog modal
        progress_dialog.grab_set()  # Prevent interaction with main window
        progress_dialog.resizable(False, False)
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        # Center the dialog on the main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (500 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (250 // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Configure the grid
        progress_dialog.columnconfigure(0, weight=1)
        progress_dialog.rowconfigure(3, weight=1)
        
        # Dialog header
        header_label = ttk.Label(
            progress_dialog, 
            text="Compressing Dataset Files", 
            font=("Arial", 14, "bold"),
            padding=(0, 10, 0, 20)
        )
        header_label.grid(row=0, column=0, sticky="ew")
        
        # Current operation label
        current_op_var = tk.StringVar(value="Initializing...")
        current_op_label = ttk.Label(
            progress_dialog, 
            textvariable=current_op_var,
            font=("Arial", 10),
            padding=(0, 0, 0, 5)
        )
        current_op_label.grid(row=1, column=0, sticky="ew")
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            progress_dialog,
            variable=progress_var,
            maximum=100,
            length=450,
            mode='determinate'
        )
        progress_bar.grid(row=2, column=0, padx=25, pady=10, sticky="ew")
        
        # Details text widget with scrollbar
        details_frame = ttk.Frame(progress_dialog)
        details_frame.grid(row=3, column=0, sticky="nsew", padx=25, pady=(10, 20))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        details_scroll = ttk.Scrollbar(details_frame)
        details_scroll.grid(row=0, column=1, sticky="ns")
        
        details_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            width=50,
            height=6,
            yscrollcommand=details_scroll.set,
            font=("Consolas", 9)
        )
        details_text.grid(row=0, column=0, sticky="nsew")
        details_scroll.config(command=details_text.yview)
        
        # Make text widget read-only but still allow copy
        details_text.configure(state="disabled")
        
        # Add initial message to log
        details_text.configure(state="normal")
        details_text.insert(tk.END, "Starting compression of dataset files...\n")
        details_text.configure(state="disabled")
        
        # Keep track of compression steps
        compression_steps = {
            "check": {"done": False, "weight": 0.1},
            "annotations": {"done": False, "weight": 0.3},
            "images": {"done": False, "weight": 0.6}
        }
        total_progress = 0.0
        
        # Define a thread to monitor the process
        def compress_thread():
            try:
                # Get paths to check for extracted files
                dm = DatasetManager()
                
                # Update the checking step
                update_ui("Checking for extracted dataset files...", 0.05)
                time.sleep(0.5)
                
                # Check if files exist
                val_dir_exists = dm.coco_val_dir.exists() and any(dm.coco_val_dir.iterdir())
                ann_dir_exists = dm.coco_ann_dir.exists() and any(dm.coco_ann_dir.iterdir())
                
                if not val_dir_exists and not ann_dir_exists:
                    update_ui("No extracted dataset files found to compress.", 1.0)
                    _show_completion("No files found to compress.")
                    return
                
                compression_steps["check"]["done"] = True
                total_progress = compression_steps["check"]["weight"]
                update_ui("Found dataset files to compress.", total_progress)
                
                # Run compression process
                if ann_dir_exists:
                    update_ui("Compressing annotation files...", total_progress + 0.05)
                    
                    # Create dummy progress updates
                    for i in range(1, 11):
                        time.sleep(0.2)  # Simulate progress
                        progress = total_progress + ((i/10) * compression_steps["annotations"]["weight"])
                        update_ui(f"Compressing annotations: {i*10}%", progress)
                    
                    compression_steps["annotations"]["done"] = True
                    total_progress += compression_steps["annotations"]["weight"]
                
                if val_dir_exists:
                    update_ui("Compressing image files...", total_progress + 0.05)
                    
                    # Create dummy progress updates for images (takes longer)
                    for i in range(1, 11):
                        time.sleep(0.3)  # Simulate progress
                        progress = total_progress + ((i/10) * compression_steps["images"]["weight"])
                        update_ui(f"Compressing images: {i*10}%", progress)
                    
                    compression_steps["images"]["done"] = True
                
                # Actually run the compression
                dm.compress_datasets()
                
                # Final update
                update_ui("Compression completed successfully.", 1.0)
                _show_completion("Datasets compressed successfully.")
                
            except Exception as e:
                error_message = f"Error during compression: {str(e)}"
                update_ui(error_message, 0.0)
                _show_error(error_message)
        
        def update_ui(message, progress):
            # Update on the main thread
            self.root.after(0, lambda m=message, p=progress: _update_ui_internal(m, p))
        
        def _update_ui_internal(message, progress):
            # Update progress bar
            progress_var.set(progress * 100)
            
            # Update operation label
            current_op_var.set(message)
            
            # Append to log
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"{message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Force update the UI
            progress_dialog.update_idletasks()
        
        def _show_completion(message="Compression complete."):
            # Update status
            self.status_var.set(message)
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Add completion message
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n✓ {message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        def _show_error(error_message):
            # Update status
            self.status_var.set(f"Error: {error_message}")
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Show error in details
            details_text.configure(state="normal")
            details_text.insert(tk.END, f"\n❌ Error: {error_message}\n")
            details_text.see(tk.END)
            details_text.configure(state="disabled")
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=4, column=0, pady=(0, 15))
        
        # Start the compression thread
        threading.Thread(target=compress_thread, daemon=True).start()

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
        if (model_type in self.model_requirements):
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
            # First, stop any running preview videos
            self.stop_preview = True
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
                try:
                    self.video_thread.join(timeout=0.5)
                except Exception as e:
                    print(f"Warning: Could not join existing video thread: {e}")

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
                     messagebox.showerror("Error", f"Default model path not defined for {model_type}. Please specify --model-path.")
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
        elif "rcnn" in model_name.lower() or "faster" in model_name.lower() or "mask" in model_name.lower():
            model_type = "mask-rcnn"
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

        # Unbind any global keys we've set
        self.root.unbind("<space>")

        # Destroy the window
        self.root.destroy()

    def _on_sample_select(self, event):
        """Handle selection of a sample video"""
        selection = self.sample_listbox.curselection()
        if not selection:
            return
        # Get the selected sample video path
        video_path = self.available_samples[selection[0]][0]
        # Auto-play the selected video as a preview
        self.play_video_in_preview(video_path, is_preview=True, autoplay=True)

    def _on_file_select(self, event):
        """Handle selection of a file in the file explorer"""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        # Get the selected file path
        selected_file = self.file_listbox.get(selection[0])
        file_path = os.path.join(self.current_dir_var.get(), selected_file)
        # Auto-play the selected video as a preview
        self.play_video_in_preview(file_path, is_preview=True, autoplay=True)

    def play_video_in_preview(self, video_path, is_preview=False, autoplay=False):
        """Play the created video in the preview panel"""
        title = "Previewing" if is_preview else "Playing"
        self.status_var.set(f"{title} video: {os.path.basename(video_path)}")

        # Create a thread to play the video in the preview panel
        # Signal to stop any existing preview first
        self.stop_preview = True

        # Clean up before starting new thread
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
            # Short timeout for thread to terminate
            try:
                self.video_thread.join(timeout=0.5)
            except Exception as e:
                print(f"Warning: Could not join existing video thread: {e}")

        # Start fresh thread
        self.stop_preview = False
        self.video_thread = threading.Thread(
            target=self._play_video_thread,
            args=(video_path, is_preview, autoplay),
            daemon=True
        )
        self.video_thread.start()

    def _play_video_thread(self, video_path, is_preview=False, autoplay=False):
        """Thread to play video in preview panel"""
        try:
            # Ensure any existing playback is stopped
            self.stop_preview = True
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
                time.sleep(0.1)  # Brief pause to allow cleanup

            # Reset playback state
            self.stop_preview = False
            self.seek_to_frame = None

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

            # Calculate dimensions to fit in the canvas
            canvas_width = self.preview_canvas.winfo_width() or 600
            canvas_height = self.preview_canvas.winfo_height() or 500

            ratio = min(canvas_width / width, canvas_height / height) * 0.9
            display_width = int(width * ratio)
            display_height = int(height * ratio)

            # Create playback controls on the canvas - simplified for smoother operation
            self.root.after(0, self._create_video_controls, total_frames)

            # Frame timing
            target_frame_time = 1.0 / fps if fps > 0 else 0.033  # Default to ~30fps
            preview_fps = min(15, fps)  # Limit preview to 15 FPS max to save resources
            preview_frame_time = 1.0 / preview_fps

            # Playback state - always playing in preview mode
            frame_idx = 0

            # Main playback loop
            while not self.stop_preview:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    # End of video reached
                    print("End of video reached")
                    # Reset to beginning for looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue

                frame_idx += 1

                # Handle seeking (if requested)
                if self.seek_to_frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_to_frame)
                    frame_idx = self.seek_to_frame
                    self.seek_to_frame = None
                    continue

                # Update progress bar
                self.root.after(0, self._update_video_progress, frame_idx, total_frames)

                # Resize and convert frame
                frame_resized = cv2.resize(frame, (display_width, display_height))

                # In preview mode, add video info overlay
                if is_preview:
                    # Add video name and frame count as overlay
                    info_text = f"{os.path.basename(video_path)} - Frame {frame_idx}/{total_frames}"
                    cv2.putText(frame_resized, info_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Convert to PIL image
                pil_img = Image.fromarray(frame_rgb)

                # Update canvas on main thread
                self.root.after(0, self._update_video_display, pil_img)

                # Control playback speed
                sleep_time = preview_frame_time if is_preview else target_frame_time
                time.sleep(sleep_time)

            # Clean up
            cap.release()
            self.root.after(0, self._remove_video_controls)
            print("Video playback stopped.")

        except Exception as e:
            print(f"Error playing video: {e}")
            self.root.after(0, lambda err=str(e): self.show_preview_message(f"Video Error:\n{err}"))

    def _create_video_controls(self, total_frames):
        """Create video playback controls on the canvas (simplified to just show progress)"""
        # Clean up any existing controls
        self._remove_video_controls()

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        # Create semi-transparent background for controls
        control_bg = self.preview_canvas.create_rectangle(
            0, canvas_height - 40, 
            canvas_width, canvas_height,
            fill="#222222", stipple="gray50",
            tags="video_controls"
        )

        # Create progress bar background
        progress_bg = self.preview_canvas.create_rectangle(
            20, canvas_height - 30,
            canvas_width - 20, canvas_height - 20,
            fill="#444444", outline="#666666",
            tags="video_controls"
        )

        # Create progress indicator (starts at 0%)
        self.video_progress = self.preview_canvas.create_rectangle(
            20, canvas_height - 30,
            20, canvas_height - 20,  # Zero width initially
            fill="#77AADD", outline="",
            tags="video_controls"
        )

        # Add video preview info text
        self.preview_canvas.create_text(
            canvas_width / 2, canvas_height - 35,
            text="Click progress bar to seek",
            fill="white", font=("Arial", 9),
            justify=tk.CENTER,
            tags="video_controls"
        )

        # Make progress bar clickable for seeking
        self.preview_canvas.tag_bind(progress_bg, "<Button-1>", 
                                   lambda event: self._seek_video(event, total_frames, canvas_width))

        # Store control dimensions for updates
        self.video_control_dims = {
            'total_frames': total_frames,
            'progress_left': 20,
            'progress_right': canvas_width - 20,
            'progress_top': canvas_height - 30,
            'progress_bottom': canvas_height - 20
        }

    def _toggle_playback(self):
        """
        This method is kept for compatibility but no longer needed for preview
        """
        pass  # Do nothing, preview always plays

    def _update_video_progress(self, current_frame, total_frames):
        """Update the video progress bar"""
        if hasattr(self, 'video_control_dims'):

            dims = self.video_control_dims
            progress_ratio = current_frame / total_frames if total_frames > 0 else 0
            progress_width = dims['progress_right'] - dims['progress_left']
            progress_pos = dims['progress_left'] + (progress_width * progress_ratio)

            # Update progress bar
            self.preview_canvas.coords(
                self.video_progress,
                dims['progress_left'], dims['progress_top'],
                progress_pos, dims['progress_bottom']
            )

    def _seek_video(self, event, total_frames, canvas_width):
        """Seek video to position clicked on progress bar"""
        if hasattr(self, 'video_control_dims'):
            dims = self.video_control_dims
            # Calculate relative position
            progress_width = dims['progress_right'] - dims['progress_left']
            click_pos = max(0, min(event.x - dims['progress_left'], progress_width))
            seek_ratio = click_pos / progress_width
            # Set frame to seek to
            self.seek_toframe = int(total_frames * seek_ratio)

    def _update_video_display(self, pil_img):
        """Update the video display on the canvas"""
        # Clear existing video frame but keep controls
        items = self.preview_canvas.find_withtag("video_frame")
        for item in items:
            self.preview_canvas.delete(item)

        # Convert PIL image to PhotoImage
        self.video_img = ImageTk.PhotoImage(pil_img)

        # Calculate position to center the image
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        x = canvas_width // 2
        y = (canvas_height - 50) // 2  # Adjust for control bar

        # Display the image
        self.preview_canvas.create_image(
            x, y, image=self.video_img, anchor=tk.CENTER,
            tags="video_frame"
        )

        # Make sure controls are on top
        self.preview_canvas.tag_raise("video_controls")

    def _remove_video_controls(self):
        """Remove video playback controls from canvas"""
        self.preview_canvas.delete("video_controls")

    def open_video_file(self, video_path):
        """Open the video file with system default player"""
        try:
            import subprocess
            # Use the default system application to open the video
            if sys.platform == "win32":
                os.startfile(video_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", video_path])
            else:
                subprocess.run(["xdg-open", video_path])
        except Exception as e:
            print(f"Error opening video: {e}")
            messagebox.showerror("Error", f"Failed to open video: {e}")

    def start_webcam_detection(self):
        """Start or stop real-time detection using the webcam."""
        if self.processing_active:
            # If already active, stop the webcam detection
            self.stop_webcam_detection()
            return
        try:
            # First, stop any running preview videos
            self.stop_preview = True
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
                try:
                    self.video_thread.join(timeout=0.5)
                except Exception as e:
                    print(f"Warning: Could not join existing video thread: {e}")

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
                # Display statistics after stopping
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
        """Thread function for processing webcam frames in real-time with optimizations."""
        try:
            # Import the video utilities
            from src.video_utils import process_webcam_with_model

            # Create events for stopping and pausing
            self.stop_event = threading.Event()
            self.pause_event = threading.Event()

            # Define display callback function for processing
            def display_callback(frame, frame_count, total_frames, model_type, progress=None):
                if frame is not None:
                    self._display_image_on_canvas(frame)
                if frame_count % 10 == 0:
                    self.root.after(0, lambda count=frame_count: 
                                   self.status_var.set(f"Webcam processing frame {count}"))

            # Update UI to indicate processing started
            self.status_var.set(f"Processing webcam with {model_manager.model_type}...")

            # Process webcam with the shared utility
            stats = process_webcam_with_model(
                model_manager=model_manager,
                callback=display_callback,
                stats_callback=None,  # We'll handle stats after completion
                stop_event=self.stop_event,
                pause_event=self.pause_event,
                camera_id=0,
                process_width=640,
                process_height=480,
                frame_skip=2
            )

            # Store statistics for display after stopping
            self.frame_count = stats['frames_processed']
            self.total_frames = stats['total_frames']
            self.processing_time = stats['processing_time']
            self.detection_counts = stats['detections']

            print(f"Webcam processing complete. Processed {self.frame_count} frames.")

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

    def pause_processing(self):
        """Pause or resume video processing"""

        # Signal the pauseevent to pause/resume the video processing thread

        if hasattr(self, 'pause_event'):
            if self.pause_processing:
                self.pause_event.clear()  # Clear the event to resume
            else:
                self.pause_event.set()  # Set the event to pause

        # Update button text
        self.pause_button.config(text="Resume" if self.pause_processing else "Pause")

        # Toggle the pause flag
        self.pause_processing = not self.pause_processing

    def process_video_with_model(self, video_path, model_manager):
        """Process a video with the specified model manager and display results in the preview panel"""
        # Update status
        self.status_var.set(f"Processing video with {model_manager.model_type}...")

        # Make sure we're in processing mode
        self.processing_active = True
        self._update_buttons(True)

        # Reset flags
        self.stop_preview = False
        self.pause_processing = False

        # Start the processing in a thread to keep UI responsive
        self.video_thread = threading.Thread(
            target=self._process_video_thread,
            args=(video_path, model_manager),
            daemon=True
        )
        self.video_thread.start()

    def _process_video_thread(self, video_path, model_manager):
        """Thread function for processing video with model"""
        try:
            # Import the video utilities
            from src.video_utils import process_video_with_model

            # Create events for stopping and pausing
            self.stop_event = threading.Event()
            self.pause_event = threading.Event()

            # Define display callback function for processing
            def display_callback(frame, frame_count, total_frames, model_type, progress=None):
                if frame is not None:
                    self._display_image_on_canvas(frame)
                if progress is not None and frame_count % 10 == 0:
                    self.root.after(0, lambda f=frame_count, p=progress: self.status_var.set(
                        f"Processing frame {f}/{total_frames} ({p:.1f}%)"))

            # Set flags
            self.stop_preview = False
            if hasattr(self, 'stop_event'):
                self.stop_event.clear()
            if hasattr(self, 'pause_event'):
                self.pause_event.clear()

            # Update UI to indicate processing started
            self.status_var.set(f"Processing video with {model_manager.model_type}...")

            # Process video with the shared utility
            stats = process_video_with_model(

                video_path=video_path,
                model_manager=model_manager,
                callback=display_callback,
                stats_callback=self.show_completion_stats, # Corrected duplicated argument
                stop_event=self.stop_event,
                pause_event=self.pause_event,
                add_fps=True
            )

            # Update UI to indicate completion
            self.root.after(0, lambda: self.status_var.set(
                f"Completed processing {stats['frames_processed']} frames in {stats['processing_time']:.1f}s ({stats['actual_fps']:.1f} FPS)"))

            # Update UI buttons back to normal state
            self.root.after(0, lambda: self._update_buttons(False))

        except Exception as e:
            print(f"Fatal error in video processing thread: {e}")
            traceback.print_exc()
            self.root.after(0, lambda err=str(e): self._display_message_on_canvas(
                f"Video processing error:\n{err}"))
            self.root.after(0, lambda: self._update_buttons(False))

    def show_completion_stats(self, stats):
        """Show completion statistics after processing the video"""
        # Format statistics
        stats_message = (
            f"Model: {stats['model_type'].upper()} ({stats['model_device']})\n"
            f"Frames: {stats['frames_processed']} / {stats['total_frames']}\n"
            f"Time: {stats['processing_time']:.2f}s\n"
            f"FPS: {stats['actual_fps']:.1f}\n\n"
        )

        # Sort detections by count (descending)
        sorted_detections = sorted(stats['detections'].items(), key=lambda x: x[1], reverse=True)
        if sorted_detections:
            stats_message += "Detections:\n"
            # Show top 10 detections
            for cls, count in sorted_detections[:10]:
                stats_message += f"• {cls}: {count}\n"
            # Indicate if there were more
            if len(sorted_detections) > 10:
                stats_message += f"• ... and {len(sorted_detections) - 10} more classes"
        else:
            stats_message += "No detections were found in the video."

        # Display stats in the preview area
        self._display_message_on_canvas(stats_message, "Processing Results")

    def show_metrics_dashboard(self):
        """Show comprehensive metrics dashboard in a new window"""
        try:
            from src.metrics_visualizer import MetricsVisualizer

            # Find the latest results file
            self.status_var.set("Loading metrics data...")
            metrics_vis = MetricsVisualizer()  # This will automatically load the latest results

            if not metrics_vis.results:
                messagebox.showerror("Error", "No evaluation results found. Run an evaluation first.") # Corrected line
                self.status_var.set("Ready")
                return

            # Display the dashboard - save to file but don't show interactively since we're likely
            # in a non-main thread (which can't safely use plt.show())
            self.status_var.set("Generating metrics dashboard...")
            output_path = metrics_vis.create_metrics_dashboard(show_plot=False)

            if output_path:
                # Instead of showing the plot interactively, open the saved file with the system viewer
                self.status_var.set(f"Metrics dashboard saved to {output_path}")
                if messagebox.askyesno("Dashboard Generated", f"Dashboard saved to:\n{output_path}\n\nOpen image now?"):
                    self.open_file(output_path)
            else:
                self.status_var.set("Error generating metrics dashboard")

        except ImportError:
            messagebox.showerror("Error", "Required packages for visualization are not installed.\n\n"
                               "Please install matplotlib, seaborn, and pandas.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to generate metrics dashboard:\n{str(e)}")
            self.status_var.set("Error generating metrics dashboard")

    def show_precision_recall_curve(self):
        """Show precision-recall curves in a new window"""
        try:
            from src.metrics_visualizer import MetricsVisualizer

            self.status_var.set("Loading metrics data...")
            metrics_vis = MetricsVisualizer()

            if not metrics_vis.results:
                messagebox.showerror("Error", "No evaluation results found. Run an evaluation first.")
                self.status_var.set("Ready")
                return

            self.status_var.set("Generating precision-recall curves...")
            output_path = metrics_vis.plot_precision_recall_curves(show_plot=True)

            if output_path:
                self.status_var.set(f"Precision-recall curves saved to {output_path}")
            else:
                self.status_var.set("Precision-recall curves displayed")

        except ImportError:
            messagebox.showerror("Error", "Required packages for visualization are not installed.\n\n"
                               "Please install matplotlib, seaborn, and pandas.")

    def show_reliability_analysis(self):
        """Show reliability analysis visualization in a new window"""
        try:
            from src.metrics_visualizer import MetricsVisualizer

            self.status_var.set("Loading metrics data...")
            metrics_vis = MetricsVisualizer()

            if not metrics_vis.results:
                messagebox.showerror("Error", "No evaluation results found. Run an evaluation first.")
                self.status_var.set("Ready")
                return

            self.status_var.set("Generating reliability analysis...")
            output_path = metrics_vis.plot_reliability_analysis(show_plot=True)

            if output_path:
                self.status_var.set(f"Reliability analysis saved to {output_path}")
            else:
                self.status_var.set("Reliability analysis displayed")

        except ImportError:
            messagebox.showerror("Error", "Required packages for visualization are not installed.\n\n"
                               "Please install matplotlib, seaborn, and pandas.")

    def show_performance_trends(self):
        """Show performance trends over time in a new window"""
        try:
            from src.metrics_visualizer import MetricsVisualizer

            self.status_var.set("Loading metrics data...")
            # For this one, we need multiple results files to show trends
            metrics_vis = MetricsVisualizer()
            aggregated_data = metrics_vis.load_multiple_results()

            if not aggregated_data:
                messagebox.showerror("Error", "Not enough evaluation results found to show trends.\n"
                                   "Run multiple evaluations to track performance over time.")
                self.status_var.set("Ready")
                return

            self.status_var.set("Generating performance trends...")
            output_path = metrics_vis.plot_performance_trends(aggregated_data, show_plot=True)

            if output_path:
                self.status_var.set(f"Performance trends saved to {output_path}")
            else:
                self.status_var.set("Performance trends displayed")

        except ImportError:
            messagebox.showerror("Error", "Required packages for visualization are not installed.\n\n"
                               "Please install matplotlib, seaborn, pandas, and jinja2.")

    def generate_metrics_report(self):
        """Generate a comprehensive HTML report with all visualizations"""
        try:
            from src.metrics_visualizer import MetricsVisualizer
            import webbrowser
            from pathlib import Path

            self.status_var.set("Loading metrics data...")
            metrics_vis = MetricsVisualizer()

            if not metrics_vis.results:
                messagebox.showerror("Error", "No evaluation results found. Run an evaluation first.")
                self.status_var.set("Ready")
                return

            self.status_var.set("Generating comprehensive report...")
            report_path = metrics_vis.generate_comprehensive_report()

            if report_path and Path(report_path).exists():
                self.status_var.set(f"Report generated: {report_path}")
                if messagebox.askyesno("Report Generated", 
                                       f"Report saved to:\n{report_path}\n\nOpen now?"):
                    webbrowser.open(f"file://{report_path}")
            else:
                messagebox.showerror("Error", "Failed to generate report")
                self.status_var.set("Report generation failed")

        except ImportError:
            messagebox.showerror("Error", "Required packages for report generation are not installed.\n\n"
                               "Please install matplotlib, seaborn, pandas, and jinja2.")

    def open_image_file(self, image_path):
        """Open the image file with system default viewer"""
        try:
            import subprocess
            # Use the default system application to open the image
            if sys.platform == "win32":
                os.startfile(image_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", image_path])
            else:
                subprocess.run(["xdg-open", image_path])
        except Exception as e:
            print(f"Error opening image: {e}")
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def open_file(self, file_path):
        """Open any file with system default application"""
        try:
            import subprocess
            # Use the default system application to open the file
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            print(f"Error opening file: {e}")
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def show_evaluation_results(self, results):
        """Display evaluation results in the preview panel"""
        # Clear canvas for fresh display
        self.preview_canvas.delete("all")

        canvas_width = self.preview_canvas.winfo_width() or 600 # Add default size
        canvas_height = self.preview_canvas.winfo_height() or 400 # Add default size

        # Create a sophisticated results display with metrics

        # Add header
        self.preview_canvas.create_text(
            canvas_width / 2, 30,
            text="Evaluation Results",
            font=("Arial", 18, "bold"),
            fill="#FFFFFF"
        )

        # Create model comparison table
        y_position = 80
        row_height = 30

        # Define column widths and positions
        col_widths = [120, 90, 90, 90, 90]
        col_positions = []
        current_x = 50
        for width in col_widths:
            col_positions.append(current_x)
            current_x += width

        # Draw header
        headers = ["Model", "mAP", "AP@50", "AP@75", "FPS"]
        for i, header in enumerate(headers):
            self.preview_canvas.create_text(
                col_positions[i] + col_widths[i] / 2, y_position,
                text=header,
                font=("Arial", 12, "bold"),
                fill="#AACCFF"
            )

        y_position += row_height

        # Store best values to highlight them
        best_values = {
            "mAP": 0,
            "AP@50": 0,
            "AP@75": 0,
            "FPS": 0
        }

        # First pass to find best values
        for model_type, metrics in results.items():
            coco_metrics = metrics.get("coco_metrics", {})
            fps = metrics.get("fps", 0)

            if coco_metrics.get("AP_IoU=0.50:0.95", 0) > best_values["mAP"]:
                best_values["mAP"] = coco_metrics.get("AP_IoU=0.50:0.95", 0)
            if coco_metrics.get("AP_IoU=0.50", 0) > best_values["AP@50"]:
                best_values["AP@50"] = coco_metrics.get("AP_IoU=0.50", 0)
            if coco_metrics.get("AP_IoU=0.75", 0) > best_values["AP@75"]:
                best_values["AP@75"] = coco_metrics.get("AP_IoU=0.75", 0)
            if fps > best_values["FPS"]:
                best_values["FPS"] = fps

        # Draw model rows
        for model_type, metrics in results.items():
            coco_metrics = metrics.get("coco_metrics", {})
            fps = metrics.get("fps", 0)

            # Draw model name (first column)
            self.preview_canvas.create_text(
                col_positions[0] + col_widths[0] / 2, y_position,
                text=model_type,
                font=("Arial", 11, "bold"),
                fill="#FFFFFF"
            )

            # Draw metrics values with conditional formatting
            metric_values = [
                coco_metrics.get("AP_IoU=0.50:0.95", 0),
                coco_metrics.get("AP_IoU=0.50", 0),
                coco_metrics.get("AP_IoU=0.75", 0),
                fps
            ]

            for i, value in enumerate(metric_values):
                text_color = "#FFFFFF"
                if i == 3:  # FPS
                    formatted_value = f"{value:.1f}"
                    # Highlight if it's the best FPS
                    if value >= best_values["FPS"] * 0.99:
                        text_color = "#88FF88"
                else:  # AP metrics
                    formatted_value = f"{value:.3f}"
                    # Highlight if it's the best AP value
                    best_key = ["mAP", "AP@50", "AP@75"][i]
                    if value >= best_values[best_key] * 0.99:
                        text_color = "#88FF88"

                self.preview_canvas.create_text(
                    col_positions[i+1] + col_widths[i+1] / 2, y_position,
                    text=formatted_value,
                    font=("Arial", 11),
                    fill=text_color
                )

            y_position += row_height

        # Add a note about the metrics dashboard
        note_y = y_position + 40
        self.preview_canvas.create_text(
            canvas_width / 2, note_y,
            text="For more detailed metrics and visualizations,\nuse the 'Metrics' menu at the top.",
            font=("Arial", 11),
            fill="#AAAAAA",
            justify=tk.CENTER
        )

        # Add button to open metrics dashboard
        button_y = note_y + 50
        button_width = 200
        button_height = 30
        button_x = canvas_width / 2 - button_width / 2

        # Create button elements with a common tag
        button_tag = "dashboard_button"
        button_area = self.preview_canvas.create_rectangle(
            button_x, button_y,
            button_x + button_width, button_y + button_height,
            fill="#2C5AA0", outline="#4477BB",
            tags=button_tag
        )
        dashboard_text = self.preview_canvas.create_text(
            canvas_width / 2, button_y + button_height / 2,
            text="Open Metrics Dashboard",
            font=("Arial", 11, "bold"),
            fill="#FFFFFF",
            tags=button_tag
        )

        # Add click feedback and action using the tag
        def on_button_click(event):
            self.preview_canvas.itemconfig(button_area, fill="#1B3E70") # Use button_area reference
            self.preview_canvas.after(100, lambda: self.preview_canvas.itemconfig(button_area, fill="#2C5AA0")) # Use button_area reference
            self.show_metrics_dashboard()

        self.preview_canvas.tag_bind(button_tag, "<Button-1>", on_button_click) # Bind the tag

    def run_best_model_on_test(self):
        """Run the best model on a test video based on evaluation results"""
        try:
            # Import the pipeline module for best model determination
            from src.pipeline import determine_best_model
            from src.metrics_visualizer import MetricsVisualizer
            import random
            from pathlib import Path

            self.status_var.set("Finding best model from evaluation results...")

            # Load the latest evaluation results
            metrics_vis = MetricsVisualizer()
            if not metrics_vis.results:
                messagebox.showerror("Error", "No evaluation results found. Run an evaluation first.")
                self.status_var.set("Ready")
                return

            # Determine the best model using the pipeline's logic
            best_model, scoring_details = determine_best_model(metrics_vis.results)

            if best_model:
                # Get sample videos
                sample_videos_dir = Path("data_sets/video_data/samples")
                if not sample_videos_dir.exists():
                    messagebox.showerror(
                        "No Sample Videos",
                        "Sample videos directory not found. Please ensure the data_sets/video_data/samples directory exists."
                    )
                    return

                video_files = list(sample_videos_dir.glob("*.mp4"))
                if not video_files:
                    messagebox.showerror(
                        "No Sample Videos",
                        "No sample videos found in the data_sets/video_data/samples directory."
                    )
                    return

                # Select an appropriate test video based on the model
                # For real applications, you'd match videos to model capabilities
                if best_model == "mask-rcnn":
                    # Mask R-CNN is good for general segmentation tasks
                    keyword_preferences = ["detection", "car", "person", "people"]
                elif best_model == "yolo-seg":
                    # YOLO-Seg is good for segmentation tasks
                    keyword_preferences = ["person", "fruit", "detection"]
                else:
                    # Default for any other model
                    keyword_preferences = ["person", "detection"]

                # Try to find a video that matches keywords
                selected_video = None
                for keyword in keyword_preferences:
                    for video in video_files:
                        if keyword.lower() in video.name.lower():
                            selected_video = video
                            break
                    if selected_video:
                        break

                # If no preference match,
                    selected_video = random.choice(video_files)

                if not selected_video:
                    messagebox.showerror(
                        "No Suitable Video",
                        "Could not find a suitable test video for the best model."
                    )
                    return

                # Show which model was determined to be best
                messagebox.showinfo(
                    "Best Model Determined",
                    f"The best performing model is: {best_model}\n\n"
                    f"Running this model on test video:\n{selected_video.name}"
                )

                # Get model path from default models
                from models import DEFAULT_MODELS
                model_path = DEFAULT_MODELS.get(best_model)
                if not model_path or not Path(model_path).exists():
                    messagebox.showerror(
                        "Model Not Found",
                        f"Could not find the model file for {best_model}."
                    )
                    return

                # Initialize the model
                self.status_var.set(f"Initializing {best_model} model...")
                from models import ModelManager
                model_manager = ModelManager(best_model, model_path)

                # Process the video
                self.status_var.set(f"Processing {selected_video.name} with {best_model}...")
                self.processing_active = True
                self._update_buttons(True)

                # Create output directory if it doesn't exist
                output_dir = Path("inference/output_videos")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Create output filename
                video_base = selected_video.stem
                output_path = output_dir / f"{video_base}_{best_model}_demo.mp4"

                # Process the video and save output
                from inference.create_demo_video import create_demo_video
                output_video_path = create_demo_video(best_model, str(selected_video), str(output_path))

                if output_video_path and Path(output_video_path).exists():
                    # Show success message and offer to play the video
                    play_video = messagebox.askyesno(
                        "Processing Complete",
                        f"The best model ({best_model}) has been run on {selected_video.name}.\n\n"
                        f"Output saved to {output_video_path}.\n\n"
                        "Would you like to play the output video now?"
                    )
                    if play_video:
                        self.play_video_in_preview(output_video_path)
                else:
                    messagebox.showerror(
                        "Processing Failed",
                        "Failed to create the demo video with the best model."
                    )
            else:
                messagebox.showerror(
                    "Best Model Determination Failed",
                    "Could not determine the best model from evaluation results."
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Error",
                f"An error occurred while running the best model: {str(e)}"
            )
        finally:
            self.status_var.set("Ready")
            self._update_buttons(False)

    def delete_datasets_gui(self):
        """Show confirmation dialog and delete all datasets"""
        confirm = messagebox.askyesno(
            "Delete All Datasets", 
            "Are you sure you want to delete all downloaded/extracted datasets?\n\n"
            "This will remove all COCO images, annotations, and compressed files.",
            icon='warning'
        )
        
        if not confirm:
            return
        
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Deleting Datasets")
        progress_dialog.geometry("400x200")
        progress_dialog.transient(self.root)  # Make dialog modal
        progress_dialog.grab_set()  # Prevent interaction with main window
        progress_dialog.resizable(False, False)
        
        # Center the dialog on the main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (400 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (200 // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Configure the grid
        progress_dialog.columnconfigure(0, weight=1)
        progress_dialog.rowconfigure(2, weight=1)
        
        # Dialog header
        header_label = ttk.Label(
            progress_dialog, 
            text="Deleting Datasets", 
            font=("Arial", 14, "bold"),
            padding=(0, 10, 0, 20)
        )
        header_label.grid(row=0, column=0, sticky="ew")
        
        # Status label
        status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(
            progress_dialog, 
            textvariable=status_var,
            font=("Arial", 10),
            padding=(0, 0, 0, 5)
        )
        status_label.grid(row=1, column=0, sticky="ew")
        
        # Progress indicator (indeterminate)
        progress = ttk.Progressbar(
            progress_dialog,
            mode='indeterminate',
            length=350
        )
        progress.grid(row=2, column=0, padx=25, pady=10, sticky="ew")
        progress.start()
        
        # Define a function for running the deletion in a separate thread
        def delete_thread():
            try:
                # Delete datasets
                dm = DatasetManager()
                dm.delete_datasets()
                
                # Show completion on the main thread
                self.root.after(0, lambda: _show_completion())
                
            except Exception as e:
                # Show error on the main thread
                error_message = f"Error: {str(e)}"
                self.root.after(0, lambda: _show_error(error_message))
        
        def _show_completion():
            # Update status
            status_var.set("All datasets have been deleted.")
            self.status_var.set("All datasets deleted successfully.")
            
            # Stop and hide progress bar
            progress.stop()
            progress.grid_remove()
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=3, column=0, pady=(0, 15))
        
        def _show_error(error_message):
            # Update status
            status_var.set(f"Error: {error_message}")
            self.status_var.set(f"Error deleting datasets: {error_message}")
            
            # Stop and hide progress bar
            progress.stop()
            progress.grid_remove()
            
            # Enable dialog closing
            progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
            
            # Add a close button
            close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
            close_button.grid(row=3, column=0, pady=(0, 15))
        
        # Start the deletion thread
        threading.Thread(target=delete_thread, daemon=True).start()

    def delete_generated_data(self):
        """Delete generated visualization data but keep the datasets"""
        confirm = messagebox.askyesno(
            "Delete Generated Data", 
            "Are you sure you want to delete all generated data?\n\n"
            "This will remove visualizations, results, and evaluation outputs,\n"
            "but will preserve downloaded datasets.",
            icon='warning'
        )
        
        if not confirm:
            return
            
        try:
            # Define paths to delete
            results_dir = Path("inference/results")
            viz_dir = results_dir / "visualizations"
            
            # Track what got deleted
            deleted_items = []
            
            # Delete visualization directory if it exists
            if viz_dir.exists():
                shutil.rmtree(viz_dir)
                deleted_items.append("visualizations")
            
            # Delete all JSON and CSV result files
            for ext in ["*.json", "*.csv"]:
                for file in results_dir.glob(ext):
                    os.remove(file)
                    deleted_items.append(file.name)
            
            # Delete all PNG charts
            for file in results_dir.glob("*.png"):
                os.remove(file)
                deleted_items.append(file.name)
            
            # Report success
            if deleted_items:
                messagebox.showinfo(
                    "Deletion Complete",
                    f"Successfully deleted {len(deleted_items)} items."
                )
                self.status_var.set("Generated data deleted successfully")
            else:
                messagebox.showinfo(
                    "Nothing to Delete",
                    "No generated data was found to delete."
                )
                self.status_var.set("No generated data found to delete")
                
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An error occurred while deleting generated data:\n{str(e)}"
            )
            self.status_var.set(f"Error deleting generated data: {str(e)}")

    def _show_evaluation_progress(self, message, progress_percent):
        """Show evaluation progress in the preview panel"""
        canvas_width = self.preview_canvas.winfo_width() or 600
        canvas_height = self.preview_canvas.winfo_height() or 400
        
        # Clear existing content
        self.preview_canvas.delete("all")
        
        # Draw header
        self.preview_canvas.create_text(
            canvas_width // 2, 50,
            text="Model Evaluation Progress",
            font=("Arial", 16, "bold"),
            fill="white"
        )
        
        # Draw message
        self.preview_canvas.create_text(
            canvas_width // 2, 100,
            text=message,
            font=("Arial", 12),
            fill="white"
        )
        
        # Draw progress bar background
        bar_width = 400
        bar_height = 20
        bar_x = (canvas_width - bar_width) // 2
        bar_y = 150
        self.preview_canvas.create_rectangle(
            bar_x, bar_y,
            bar_x + bar_width, bar_y + bar_height,
            fill="#444444",
            outline="#666666"
        )
        
        # Draw progress bar fill
        fill_width = (progress_percent / 100) * bar_width
        self.preview_canvas.create_rectangle(
            bar_x, bar_y,
            bar_x + fill_width, bar_y + bar_height,
            fill="#77AADD",
            outline=""
        )
        
        # Draw percentage text
        self.preview_canvas.create_text(
            canvas_width // 2, bar_y + bar_height + 20,
            text=f"{progress_percent:.1f}%",
            font=("Arial", 10),
            fill="white"
        )
        
        # Force update
        self.preview_canvas.update()

    def _update_evaluation_progress(self, progress_percent, message):
        """Update the evaluation progress display"""
        self._show_evaluation_progress(message, progress_percent)
        self.status_var.set(f"Evaluating models: {progress_percent:.1f}%")

    def run_evaluation_gui(self):
        """Prompt for evaluation parameters and run evaluation in background"""
        # Ask for number of images
        num = simpledialog.askinteger("Run Evaluation", f"Number of images (1-{COCO_VAL_TOTAL_IMAGES}):", minvalue=1, maxvalue=COCO_VAL_TOTAL_IMAGES)
        if not num:
            return
        no_vis = messagebox.askyesno("Visualizations", "Skip saving visualizations? (Yes to skip)")

        # Show initial progress display
        self._show_evaluation_progress("Starting evaluation...", 0)
        self.status_var.set(f"Evaluating models on {num} images...")
        self.root.update()

        # Run evaluation in a thread
        def eval_task():
            try:
                # --- Progress Callback Definition ---
                def update_progress_display(model_type, current_image, total_images):
                    """Callback function to update the preview canvas during evaluation."""
                    progress_percent = (current_image / total_images) * 100 if total_images > 0 else 0
                    message = f"Evaluating: {model_type.upper()}\nImage {current_image} of {total_images}"
                    # Update progress in main thread
                    self.root.after(0, lambda p=progress_percent, m=message: 
                                  self._update_evaluation_progress(p, m))

                # --- End Progress Callback Definition ---

                evaluator = ModelEvaluator()

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

    def delete_downloaded_models(self):
        """Delete downloaded model files from the models/pts directory"""
        confirm = messagebox.askyesno(
            "Delete Downloaded Models", 
            "Are you sure you want to delete all downloaded model files?\n\n"
            "This will remove all model files from the models/pts directory.",
            icon='warning'
        )
        
        if not confirm:
            return
            
        try:
            # Create a progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Deleting Model Files")
            progress_dialog.geometry("400x200")
            progress_dialog.transient(self.root)  # Make dialog modal
            progress_dialog.grab_set()  # Prevent interaction with main window
            progress_dialog.resizable(False, False)
            
            # Center the dialog on the main window
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (400 // 2)
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (200 // 2)
            progress_dialog.geometry(f"+{x}+{y}")
            
            # Configure the grid
            progress_dialog.columnconfigure(0, weight=1)
            progress_dialog.rowconfigure(2, weight=1)
            
            # Dialog header
            header_label = ttk.Label(
                progress_dialog, 
                text="Deleting Model Files", 
                font=("Arial", 14, "bold"),
                padding=(0, 10, 0, 20)
            )
            header_label.grid(row=0, column=0, sticky="ew")
            
            # Status label
            status_var = tk.StringVar(value="Initializing...")
            status_label = ttk.Label(
                progress_dialog, 
                textvariable=status_var,
                font=("Arial", 10),
                padding=(0, 0, 0, 5)
            )
            status_label.grid(row=1, column=0, sticky="ew")
            
            # Progress indicator (indeterminate)
            progress = ttk.Progressbar(
                progress_dialog,
                mode='indeterminate',
                length=350
            )
            progress.grid(row=2, column=0, padx=25, pady=10, sticky="ew")
            progress.start()
            
            def delete_thread():
                try:
                    models_dir = Path("models/pts")
                    deleted_count = 0
                    
                    # Check if directory exists
                    if models_dir.exists() and models_dir.is_dir():
                        # Get all model files
                        model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth")) + list(models_dir.glob("*.onnx"))
                        
                        if model_files:
                            # Update status
                            self.root.after(0, lambda: status_var.set(f"Found {len(model_files)} model files to delete..."))
                            
                            # Delete each file
                            for file in model_files:
                                try:
                                    os.remove(file)
                                    deleted_count += 1
                                    self.root.after(0, lambda c=deleted_count, t=len(model_files): status_var.set(f"Deleted {c}/{t} files..."))
                                except Exception as e:
                                    print(f"Error deleting {file}: {e}")
                        else:
                            self.root.after(0, lambda: status_var.set("No model files found to delete."))
                    else:
                        self.root.after(0, lambda: status_var.set("Models directory not found."))
                    
                    # Show completion
                    self.root.after(0, lambda d=deleted_count: _show_completion(d))
                    
                except Exception as e:
                    # Show error
                    error_message = f"Error: {str(e)}"
                    self.root.after(0, lambda: _show_error(error_message))
            
            def _show_completion(deleted_count):
                # Update status
                message = f"Successfully deleted {deleted_count} model files."
                status_var.set(message)
                self.status_var.set(message)
                
                # Stop and hide progress bar
                progress.stop()
                progress.grid_remove()
                
                # Enable dialog closing
                progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
                
                # Add a close button
                close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
                close_button.grid(row=3, column=0, pady=(0, 15))
            
            def _show_error(error_message):
                # Update status
                status_var.set(f"Error: {error_message}")
                self.status_var.set(f"Error deleting model files: {error_message}")
                
                # Stop and hide progress bar
                progress.stop()
                progress.grid_remove()
                
                # Enable dialog closing
                progress_dialog.protocol("WM_DELETE_WINDOW", progress_dialog.destroy)
                
                # Add a close button
                close_button = ttk.Button(progress_dialog, text="Close", command=progress_dialog.destroy)
                close_button.grid(row=3, column=0, pady=(0, 15))
            
            # Start the deletion thread
            threading.Thread(target=delete_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An error occurred while setting up model deletion:\n{str(e)}"
            )
            self.status_var.set(f"Error deleting model files: {str(e)}")

def main():
    """
    Main function that processes command line arguments or launches GUI
    """
    parser = argparse.ArgumentParser(description="Computer Vision Project")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode")
    # parser.add_argument("--terminal-ui", action="store_true", help="Run with terminal-based UI (Currently disabled)") # Disabled TUI
    parser.add_argument("--model-type", type=str, choices=["mask-rcnn", "yolo-seg"], 
                          help="Type of model to use (required for CLI)")
    parser.add_argument("--model-path", type=str, help="Path to the model weights file (optional for CLI, uses default)")
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
    #         model_manager = ModelManager(model_type, model_path)
    #         display_video_with_model(video_path, model_manager)
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



