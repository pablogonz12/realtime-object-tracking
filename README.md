# Computer Vision Object Recognition System

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-development-yellow.svg)

This project implements a robust computer vision system designed for detecting and segmenting objects within both video sequences and static images. It evaluates various state-of-the-art object detection and segmentation algorithms using the COCO dataset, focusing on performance metrics related to validity, reliability, and objectivity.

## Getting Started

To get started with this project, please refer to the following documentation:

1.  **[Setup and Installation](docs/setup_installation.md)**: For environment setup, software/system requirements, and installation steps.
2.  **[Usage Guide](docs/usage_guide.md)**: For instructions on how to use the GUI and command-line tools.

## Key Components

-   **Source Code**: The core logic of the application resides in the `src/` directory.
-   **Tests**: Comprehensive tests are located in the `tests/` directory, with execution scripts in `scripts/`.
-   **Detailed Documentation**: For an in-depth understanding of the project, including specific solutions, project structure, and more, please explore the `docs/` folder.
-   **Analysis & Visualization**: Evaluation results (JSON), performance dashboards (PNG), detailed metrics reports (PDF), and image visualizations from model evaluations are stored in the `inference/results/` directory (visualizations are typically in subdirectories like `inference/results/[model_name]_visualizations/`). Processed demo videos can be found in `inference/output_videos/`.
-   **Requirements**: Project dependencies are listed in `requirements.txt`.

## Documentation Index

For more specific information, please see the relevant documents in the `docs/` folder:

- **[Project Overview](docs/project_overview.md)**: High-level description, dataset justification, and key features.
- **[Task #2 Solution](docs/task2_solution.md)**: Details on how this project addresses the specific task, including deliverables and reproduction steps.
- **[Project Structure](docs/project_structure.md)**: Overview of the repository's directory and file organization.
- **[Reproducibility and Troubleshooting](docs/reproducibility_troubleshooting.md)**: Notes on ensuring reproducible results and common troubleshooting tips.
- **[Testing & Quality Assurance](docs/testing_qa.md)**: Information on the testing framework, running tests, and QA measures.
- **[License and Development Status](docs/license_development.md)**: Licensing information and current development status.