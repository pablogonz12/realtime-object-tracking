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
    'mask-rcnn': 'models/pts/maskrcnn_resnet50_fpn.pt',
    'yolo-seg': 'models/pts/yolov8n-seg.pt',
    'sam': 'models/pts/sam_vit_h.pt'
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
        self.pause_button = ttk.Button(self.processing_buttons_frame, text="Pause", command=self.pause_processing)
        self.stop_button = ttk.Button(self.processing_buttons_frame, text="Stop", command=self.stop_processing)
        
        self._update_buttons(processing=False)

    def stop_processing(self):
        """Stop video processing"""
        self.stop_preview = True
        self.pause_processing = False
        
        # Signal the stop event to stop the video processing thread
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        
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
    
    def _show_evaluation_progress(self, message, percentage=0):
        """Show evaluation progress with progress bar in preview panel"""
        # Clear canvas
        self.preview_canvas.delete("all")
        canvas_width = self.preview_canvas.winfo_width() or 600
        canvas_height = self.preview_canvas.winfo_height() or 500
        
        # Drawing background
        bg_color = "#383838"
        self.preview_canvas.create_rectangle(
            10, 10, canvas_width - 10, canvas_height - 10,
            fill=bg_color, outline="#555555", width=1
        )
        
        # Add header
        header_text = "Model Evaluation"
        header_color = "#77AADD"
        self.preview_canvas.create_text(
            canvas_width // 2, 40,
            text=header_text,
            fill=header_color, font=("Arial", 16, "bold"),
            anchor=tk.CENTER
        )
        
        # Add message
        self.eval_message = self.preview_canvas.create_text(
            canvas_width // 2, canvas_height // 2 - 50,
            text=message,
            fill="#FFFFFF", font=("Arial", 14),
            justify=tk.CENTER
        )
        
        # Create progress bar background
        bar_width = int(canvas_width * 0.7)  # 70% of canvas width
        bar_height = 25
        bar_x = (canvas_width - bar_width) // 2
        bar_y = canvas_height // 2 + 20
        
        self.preview_canvas.create_rectangle(
            bar_x, bar_y,
            bar_x + bar_width, bar_y + bar_height,
            fill="#2B2B2B", outline="#AAAAAA"
        )
        
        # Store references for updating
        self.progress_bar_coords = (bar_x, bar_y, bar_width, bar_height)
        self.eval_progress_bar = self.preview_canvas.create_rectangle(
            bar_x, bar_y,
            bar_x, bar_y + bar_height,  # Zero width initially
            fill="#4CAF50", outline=""
        )
        
        # Create text to show percentage
        self.eval_progress_text = self.preview_canvas.create_text(
            canvas_width // 2, bar_y + bar_height + 20,
            text="0%",
            fill="#FFFFFF", font=("Arial", 12),
            anchor=tk.CENTER
        )
        
        # Add note about the process
        note_y = bar_y + bar_height + 60
        self.preview_canvas.create_text(
            canvas_width // 2, note_y,
            text="Evaluating three models on COCO dataset:\n" + 
                 "YOLO-Seg, RT-DETR, and Faster R-CNN",
            fill="#AAAAAA", font=("Arial", 12),
            justify=tk.CENTER
        )
    
    def _update_evaluation_progress(self, percentage, message):
        """Update the evaluation progress bar"""
        # Update progress bar
        if hasattr(self, 'progress_bar_coords') and hasattr(self, 'eval_progress_bar'):
            # Update progress bar width
            bar_x, bar_y, bar_width, bar_height = self.progress_bar_coords
            progress_width = int(bar_width * percentage / 100)
            
            # Update rectangle
            self.preview_canvas.coords(
                self.eval_progress_bar,
                bar_x, bar_y,
                bar_x + progress_width, bar_y + bar_height
            )
            
            # Update text
            if hasattr(self, 'eval_progress_text'):
                self.preview_canvas.itemconfig(
                    self.eval_progress_text, 
                    text=f"{percentage:.1f}%"
                )
            
            # Update message
            if hasattr(self, 'eval_message'):
                self.preview_canvas.itemconfig(
                    self.eval_message,
                    text=message
                )
            
            # Update status bar
            self.status_var.set(f"Evaluation: {percentage:.1f}% complete")
            
            # Force GUI update
            self.root.update_idletasks()
        
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
            self.root.after(0, lambda err=str(e): self.show_preview_message(
                f"Error playing video:\n{err}"))

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
            20, canvas_height - 20,
            fill="#77AADD", outline="",
            tags="video_controls"
        )
        
        # Add video preview info text
        self.preview_canvas.create_text(
            canvas_width // 2, canvas_height - 35,
            text="Click progress bar to seek",
            fill="white", font=("Arial", 9),
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
            self.seek_to_frame = int(total_frames * seek_ratio)
            
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
        """Thread function for processing webcam frames in real-time with optimizations."""
        try:
            # Import the video utilities
            from inference.video_utils import process_webcam_with_model
            
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
        self.pause_processing = not self.pause_processing
        
        # Signal the pause event to pause/resume the video processing thread
        if hasattr(self, 'pause_event'):
            if self.pause_processing:
                self.pause_event.set()  # Set the event to pause
            else:
                self.pause_event.clear()  # Clear the event to resume
                
        # Update button text
        self.pause_button.config(text="Resume" if self.pause_processing else "Pause")
        
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
            from inference.video_utils import process_video_with_model
            
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
                stats_callback=self.show_completion_stats,
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

def main():
    """
    Main function that processes command line arguments or launches GUI
    """
    parser = argparse.ArgumentParser(description="Computer Vision Project")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode")
    # parser.add_argument("--terminal-ui", action="store_true", help="Run with terminal-based UI (Currently disabled)") # Disabled TUI
    parser.add_argument("--model-type", type=str, choices=["mask-rcnn", "yolo-seg", "sam"], 
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