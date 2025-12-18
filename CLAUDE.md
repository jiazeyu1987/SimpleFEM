# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimpleFEM is a standalone ROI (Region of Interest) capture and peak detection daemon for the NHEM (New HEM Monitor) medical signal processing system. It operates independently from the main NHEM system, providing focused analysis capabilities for HEM (高回声事件 - High Echo Event) detection through computer vision and signal processing. The system features dual input modes (screen capture and video file processing) with medical-grade reliability and comprehensive data export capabilities.

## Development Commands

### Running SimpleFEM
```bash
# Main daemon - HEM detection
python simple_roi_daemon.py

# Vein detection and following system
python auto_vein_detector.py

# Run the built standalone executable (after building)
dist\SimpleFEM_ROI_Daemon.exe

# Validate implementation
python verify_implementation.py
```

### Building Standalone Executable
```bash
# Install PyInstaller (if not already installed)
pip install pyinstaller

# Build Windows executable using the provided command
pyinstaller --onefile --console --name SimpleFEM_ROI_Daemon --add-data "simple_fem_config.json;." simple_roi_daemon.py

# View build command
type 打包命令.txt
```

### Testing and Development
```bash
# Verify core module imports
python -c "from green_detector import detect_green_intersection; print('Green detector OK')"
python -c "from peak_detection import detect_peaks; print('Peak detection OK')"
python -c "from safe_peak_statistics import safe_statistics; print('Statistics OK')"

# Test peak detection algorithms
python simple_peak_test.py
python improved_peak_detection.py

# Test vein detection system
python auto_vein_detector.py
```

### Configuration Management
```bash
# Main HEM detection configuration
simple_fem_config.json

# Vein detection specific configuration
vein_detection_config.json

# Environment variable overrides (NHEM_* prefix)
export NHEM_THRESHOLD=95.0
export NHEM_FPS=10
export NHEM_PROCESSING_MODE=video
```


## Architecture Overview

### Core Processing Pipeline

**SimpleFEM** implements a modular, medical-grade signal processing pipeline with dual operation modes:

#### Main Processing Components
1. **Main Daemon** (`simple_roi_daemon.py`) - Primary orchestrator with video/screen capture, batch video processing, and threshold protection
2. **Green Line Detection** (`green_detector.py`) - OpenCV HSV filtering with Canny edge detection and Hough line transformation
3. **Peak Detection** (`peak_detection.py`) - Signal processing with adaptive thresholding and green/red classification
4. **Improved Peak Detection** (`improved_peak_detection.py`) - Enhanced algorithms with signal preprocessing and lifecycle tracking
5. **Statistics Management** (`safe_peak_statistics.py`) - Three-layer deduplication with atomic CSV export and session tracking
6. **Vein Detection** (`auto_vein_detector.py`) - Connected component analysis with automatic ROI tracking for vein following

#### System Architecture Patterns
- **Modular Design**: Clear separation of concerns with plugin-like algorithm support
- **Dual Input Modes**: Seamless switching between real-time screen capture and video file processing
- **Circular Buffer Pattern**: Fixed-size sliding window (100 frames) for memory-efficient processing
- **Configuration Hierarchy**: JSON files with environment variable overrides (NHEM_* prefix)
- **Medical-Grade Reliability**: Comprehensive error handling and graceful degradation

### Data Flow Architecture

#### Primary HEM Detection Pipeline
```
Input Source (Screen/Video) → ROI1 Processing → Green Line Detection (OpenCV)
                                                              ↓
                                     Intersection Point Calculation → ROI2 Extraction
                                                              ↓
                                     Grayscale Value Computation → Circular Buffer (100 frames)
                                                              ↓
                                     Adaptive Peak Detection → Green/Red Classification
                                                              ↓
                                     Three-Layer Deduplication → CSV Export + Logging
```

#### Vein Detection Pipeline
```
Screen/Video Capture → ROI2 Extraction → Connected Component Analysis (0-10 threshold)
                                                              ↓
                                     Mask Generation → Center Point Calculation
                                                              ↓
                                     ROI Repositioning → Continuous Tracking
```

**Processing Modes:**
- **Screen Mode**: Real-time screen capture using PIL.ImageGrab with automatic boundary adjustment
- **Video Mode**: Process local video files with optional looping and batch multi-video support
- **Vein Following**: Automatic ROI tracking based on connected component detection
- **Hybrid Support**: Runtime switching between input sources via configuration

### Key Technologies

- **Computer Vision**: OpenCV HSV filtering, Canny edge detection, Hough line transformation, connected component analysis
- **Image Processing**: PIL/Pillow screen capture, numpy array operations, ROI coordinate transformation
- **Signal Processing**: Adaptive thresholding, morphological operations, peak lifecycle tracking, frame difference analysis
- **Data Management**: Thread-safe collections, atomic file operations, temporary file handling, session tracking
- **Build System**: PyInstaller standalone executables with embedded configuration files

## Configuration System

SimpleFEM uses a hierarchical configuration approach with runtime override capabilities:

### 1. Primary Configuration (`simple_fem_config.json`)
```json
{
  "processing_mode": "video",        // "screen" or "video" or "vein_following"
  "data_processing": {
    "save_roi1": true,              // Save large ROI captures
    "save_roi2": true,              // Save small ROI extracts
    "save_wave": true,              // Save waveform plots
    "only_delect": true             // Only save frames with detected peaks
  },
  "video_processing": {
    "video_path": "video.mp4",      // Video file path or array for batch processing
    "loop_enabled": false,          // Loop video playback
    "processing_frame_rate": 10.0   // Processing frame rate override
  },
  "roi_capture": {
    "frame_rate": 10,               // Capture interval (1-10 FPS)
    "default_config": {
      "x1": 1280, "y1": 80,         // Large ROI1 coordinates
      "x2": 1920, "y2": 980
    },
    "roi2_config": {
      "extension_params": {
        "left": 20, "right": 30,     // ROI2 extraction around intersection
        "top": 60, "bottom": 20
      }
    }
  },
  "peak_detection": {
    "threshold": 95.0,                      // Fixed grayscale threshold
    "adaptive_threshold_enabled": true,      // Enable adaptive thresholding
    "threshold_over_mean_ratio": 0.15,       // Adaptive threshold ratio (15% above mean)
    "margin_frames": 5,                     // Peak boundary extension
    "silence_frames": 15,                   // Minimum silence between peaks
    "difference_threshold": 2.1,            // Green/red classification threshold
    "min_region_length": 5                  // Minimum peak width in frames
  },
  "deduplication": {
    "consecutive_frame_window": 10,         // Consecutive frame deduplication
    "color_priority": ["green", "red"],     // Peak color priority for conflicts
    "recent_peak_window": 5                 // Recent peak comparison window
  }
}
```

### 2. Vein Detection Configuration (`vein_detection_config.json`)
```json
{
  "roi_capture": {
    "roi2_config": {
      "x1": 100, "y1": 100,           // Initial ROI2 coordinates
      "x2": 150, "y2": 150
    }
  },
  "vein_detection": {
    "pixel_threshold_min": 0,          // Minimum pixel value for detection
    "pixel_threshold_max": 10,         // Maximum pixel value for detection
    "roi_size_tolerance": 50,          // Maximum ROI change before reset
    "tracking_failure_fallback": "previous_position"  // Recovery strategy
  },
  "output_settings": {
    "save_masks": true,                // Save binary masks
    "save_roi2": true,                 // Save ROI2 images
    "output_directory": "roi2"         // Output folder for masks and ROI2
  }
}
```

### 3. Environment Variable Overrides
- **Prefix**: `NHEM_*` (e.g., `NHEM_FPS=10`, `NHEM_THRESHOLD=95.0`)
- **Runtime Configuration**: Override JSON values without restart
- **Development**: Quick parameter tuning for testing
- **Production**: Environment-specific deployments

## Core Processing Pipeline

### 1. ROI Capture and Input Processing
- **Dual Input Sources**: Real-time screen capture (PIL.ImageGrab) or video file processing (OpenCV)
- **ROI1**: Large capture region (default: 1280x80 to 1920x980) for green line detection
- **ROI2**: Small region (default: 50x50) extracted around green line intersection or vein tracking
- **Automatic Boundary Adjustment**: ROI coordinates clamped to screen/video dimensions
- **Frame Rate Control**: Configurable intervals (1-30 FPS) with precise timing

### 2. Green Line Detection (OpenCV Pipeline)
```python
# HSV color space filtering for robust green detection
hsv = cv2.cvtColor(roi1_array, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 80, 80])     # HSV lower bound
upper_green = np.array([85, 255, 255])   # HSV upper bound
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Edge detection and probabilistic Hough line transformation
edges = cv2.Canny(mask_green, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                        threshold=50, minLineLength=80, maxLineGap=20)

# Select non-parallel lines and compute geometric intersection
intersection = compute_line_intersection(selected_lines)
```

### 3. Adaptive Peak Detection System
- **Threshold Protection**: Background updates only when below threshold to prevent peak contamination
- **Adaptive Algorithm**: Dynamic threshold calculation based on historical background mean
- **Peak Classification**: Green (post-peak > pre-peak + threshold) vs Red (all other peaks)
- **Multi-Algorithm Support**: Basic threshold detection + improved morphological operations
- **Lifecycle Tracking**: Peak evolution monitoring with state transitions (white → green/red)

### 4. Vein Detection and Following
```python
# Connected component analysis in 0-10 pixel range
binary_mask = cv2.inRange(roi2_gray, 0, 10)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

# Identify component containing ROI center point
center_component = find_component_containing_point(labels, roi2_center)
mask_center = centroids[center_component]

# Automatic ROI repositioning for continuous tracking
new_roi2_coords = reposition_roi_around_center(mask_center, roi2_dimensions)
```

### 5. Three-Layer Deduplication System
1. **Recent Peak Comparison**: 5-frame window prevents immediate duplicates
2. **Consecutive Frame Deduplication**: Same-color peak filtering with configurable windows
3. **Invalid Data Filtering**: Zero average values and malformed peak rejection
- **Color Priority**: Green peaks take precedence over red in conflicts
- **Atomic CSV Export**: Thread-safe writing with temporary files

### 6. Data Export and Session Management
- **CSV Format**: Comprehensive metadata (timestamp, frame indices, coordinates, ROI info)
- **Session Tracking**: Unique IDs for data provenance and analysis
- **File Organization**: Structured output directories (`/exports/`, `/tmp/`, `/logs/`)
- **Waveform Visualization**: Matplotlib plots with peak annotations and classification

## Performance and Reliability

### Processing Performance
- **Frame Rate**: 1-30 FPS configurable (default: 10 FPS)
- **Processing Latency**: <100ms per frame for typical ROI sizes (<200x200 pixels)
- **Memory Management**: Fixed-size circular buffers prevent memory leaks
- **CPU Usage**: Optimized for continuous medical monitoring applications
- **Batch Processing**: Multi-video support with automatic video switching

### Data Persistence and Output
- **Atomic Operations**: Temporary files ensure data integrity during export
- **Structured Output**: Organized directory structure (`/exports/`, `/roi1/`, `/roi2/`, `/wave/`, `/logs/`)
- **Session Management**: Unique timestamped session IDs for data tracking
- **Log Rotation**: Daily log file rotation with configurable retention
- **File Formats**: CSV for data, PNG for visualizations, standard image formats

### Medical-Grade Reliability Features
- **Graceful Degradation**: Individual component failures don't stop processing
- **Comprehensive Error Recovery**: Automatic fallback mechanisms and recovery strategies
- **Resource Management**: Automatic cleanup of memory, file handles, and video resources
- **Data Integrity**: Three-layer validation prevents false positives and data corruption
- **Thread Safety**: Concurrent operations with proper synchronization

## Development Architecture

### Technology Stack
- **Core Dependencies** (bundled with PyInstaller):
  - `numpy` - Numerical computations and array operations
  - `opencv-python` - Computer vision and image processing
  - `Pillow` - Screen capture and image manipulation
  - `matplotlib` - Waveform visualization and plotting

### Modular Code Structure
```
SimpleFEM/
├── Core Processing Modules/
│   ├── simple_roi_daemon.py      # Main orchestrator with dual input modes
│   ├── green_detector.py         # OpenCV HSV filtering and intersection detection
│   ├── peak_detection.py         # Adaptive thresholding and classification
│   ├── improved_peak_detection.py # Enhanced algorithms with lifecycle tracking
│   └── auto_vein_detector.py     # Connected component analysis and ROI tracking
├── Data Management/
│   ├── safe_peak_statistics.py   # Three-layer deduplication and atomic export
│   └── VideoStatisticsManager    # Batch video processing coordination
├── Configuration/
│   ├── simple_fem_config.json    # Primary HEM detection configuration
│   └── vein_detection_config.json # Vein-specific tracking parameters
├── Testing and Validation/
│   ├── simple_peak_test.py       # Peak detection algorithm testing
│   └── verify_implementation.py  # Module integrity validation
├── Build System/
│   ├── 打包命令.txt               # PyInstaller build command
│   └── dist/                     # Standalone executable output
└── Documentation/
    ├── CLAUDE.md                 # Claude Code development guidance
    ├── docs/HEM检测/             # Algorithm documentation (Chinese)
    └── docs/静脉检测跟随/         # Vein detection specifications
```

### Development Patterns and Best Practices
- **Plugin Architecture**: Modular detection algorithms with consistent interfaces
- **Configuration Hierarchy**: JSON files with environment variable overrides
- **Error Isolation**: Component failures contained to prevent system crashes
- **Resource Lifecycle**: Context managers and explicit cleanup patterns
- **Logging Strategy**: Comprehensive logging with severity levels and timestamps
- **Testing Approach**: Module validation, component testing, and integration verification

## Medical Application Context

SimpleFEM is designed for **medical-grade signal monitoring**:

- **HEM Detection**: High Echo Event detection in medical imaging
- **Real-time Analysis**: Continuous monitoring with configurable sensitivity
- **Data Integrity**: Auditable logs and export capabilities
- **Clinical Use**: Suitable for research and diagnostic support
- **Quality Assurance**: Deduplication and validation for reliable data

## Integration with NHEM Ecosystem

While SimpleFEM operates independently, it integrates with the broader NHEM system:

- **Data Compatibility**: Outputs compatible with NHEM backend formats
- **Configuration Standards**: Uses NHEM-style JSON configuration
- **Performance Alignment**: Designed to complement main NHEM processing
- **Medical Standards**: Follows same medical-grade reliability requirements

## Important Development Notes

- **Dual Input Modes**: Supports both real-time screen capture and video file processing
- **Adaptive Thresholding**: Background-based threshold calculation with configurable ratios
- **Peak Classification**: Green (post-peak > pre-peak + threshold) vs Red (all other peaks)
- **Standalone Operation**: No network dependencies or external services required
- **Windows Platform**: Designed specifically for Windows screen capture capabilities
- **Resource Efficiency**: Fixed circular buffers prevent memory leaks during continuous operation
- **Medical Reliability**: Built for medical applications with comprehensive error handling
- **Configuration Management**: Runtime configuration changes via JSON without restart
- **Data Security**: Local processing with no external data transmission
- **Build System**: PyInstaller creates standalone executables with embedded configuration