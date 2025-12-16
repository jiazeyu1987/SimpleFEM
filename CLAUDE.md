# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimpleFEM is a standalone ROI (Region of Interest) capture and peak detection daemon for the NHEM (New HEM Monitor) medical signal processing system. It operates independently from the main NHEM system, providing focused analysis capabilities for HEM (高回声事件 - High Echo Event) detection through computer vision and signal processing.

## Development Commands

### Running SimpleFEM
```bash
# Run as Python script
python simple_roi_daemon.py

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

# Alternative: use the build command file
type 打包命令.txt  # View the build command
```

### Testing and Development
```bash
# Verify module imports and basic functionality
python -c "from green_detector import detect_green_intersection; print('Green detector OK')"
python -c "from peak_detection import detect_peaks; print('Peak detection OK')"
python -c "from safe_peak_statistics import safe_statistics; print('Statistics OK')"

# Test peak detection with sample data
python simple_peak_test.py

# Test improved peak detection (if available)
python improved_peak_detection.py
```


## Architecture Overview

### Core Components

**SimpleFEM** is designed as a **standalone daemon** that processes screen captures or video files to detect medical signal events:

1. **Main Daemon** (`simple_roi_daemon.py`) - Main application orchestrating the entire processing pipeline with video/screen capture support
2. **Green Line Detection** (`green_detector.py`) - Computer vision module using OpenCV HSV filtering to detect green line intersections
3. **Peak Detection** (`peak_detection.py`) - Signal processing with adaptive thresholding and green/red peak classification
4. **Improved Peak Detection** (`improved_peak_detection.py`) - Alternative enhanced detection algorithms (optional)
5. **Statistics Management** (`safe_peak_statistics.py`) - Data export, deduplication, and CSV generation with session tracking
6. **Configuration** (`simple_fem_config.json`) - Hierarchical JSON configuration with video/screen processing modes

### Data Flow Architecture

```
Input Source (Screen/Video) → ROI1 Processing → Green Line Detection (OpenCV)
                                                              ↓
                                     Intersection Point Calculation → ROI2 Extraction
                                                              ↓
                                     Grayscale Value Computation → Circular Buffer (100 frames)
                                                              ↓
                                     Adaptive Peak Detection → Green/Red Classification
                                                              ↓
                                     Statistics & Deduplication → CSV Export + Logging
```

**Processing Modes:**
- **Screen Mode**: Real-time screen capture using PIL.ImageGrab
- **Video Mode**: Process local video files with optional looping
- **Hybrid Support**: Seamlessly switch between input sources via configuration

### Key Technologies

- **Computer Vision**: OpenCV for green line detection using HSV color space filtering
- **Image Processing**: PIL/Pillow for screen capture and ROI extraction
- **Signal Processing**: Custom peak detection algorithms with configurable parameters
- **Data Management**: Thread-safe circular buffers with CSV export functionality
- **Build System**: PyInstaller for creating standalone Windows executables

## Configuration System

SimpleFEM uses a hierarchical configuration approach:

### 1. JSON Configuration (`simple_fem_config.json`)
```json
{
  "data_processing": {
    "save_roi1": true,           // Save large ROI captures
    "save_roi2": true,           // Save small ROI extracts
    "save_wave": true,           // Save waveform plots
    "only_delect": true          // Only log frames with detected peaks
  },
  "roi_capture": {
    "frame_rate": 10,            // Capture interval (1-10 FPS)
    "default_config": {
      "x1": 1280, "y1": 80,     // Large ROI1 coordinates
      "x2": 1920, "y2": 980
    },
    "roi2_config": {
      "extension_params": {
        "left": 20, "right": 30, // ROI2 extraction around intersection
        "top": 60, "bottom": 20
      }
    }
  },
  "peak_detection": {
    "threshold": 90.0,           // Grayscale threshold for detection
    "margin_frames": 5,          // Boundary extension for peaks
    "difference_threshold": 1.1, // Frame difference threshold
    "min_region_length": 5       // Minimum peak region length
  },
  "deduplication": {
    "consecutive_frame_window": 10,      // Deduplication window
    "consecutive_deduplication_enabled": true
  }
}
```

### 2. Environment Variables
- Prefix: `NHEM_*` (e.g., `NHEM_FPS`, `NHEM_THRESHOLD`)
- Override JSON configuration values
- Runtime configuration without file modification

### 3. Code Defaults
- Fallback values hardcoded in modules
- Ensure daemon operates with reasonable defaults

## Core Processing Pipeline

### 1. ROI Capture System
- **ROI1**: Large capture region (default: 1280x80 to 1920x980)
- **ROI2**: Small region (50x50) extracted around green line intersection
- **Frame Rate**: Configurable capture intervals (1-10 FPS)
- **Coordinate Transformation**: Automatic mapping between screen and image coordinates

### 2. Green Line Detection Algorithm
```python
# HSV color space filtering for green lines
lower_green = [35, 80, 80]     # HSV lower bound
upper_green = [85, 255, 255]   # HSV upper bound

# Edge detection and Hough line transformation
edges = cv2.Canny(mask_green, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                        threshold=50, minLineLength=80, maxLineGap=20)

# Geometric intersection calculation
intersection = compute_intersection(line1, line2)
```

### 3. Peak Detection Algorithm
- **Method**: Absolute threshold detection with margin extension
- **Classification**: Green peaks (stable) vs Red peaks (unstable)
- **Parameters**: Configurable thresholds, margins, and minimum region lengths
- **Output**: Peak intervals with frame difference calculations

### 4. Statistics and Export
- **CSV Export**: Daily log files with timestamped peak data
- **Deduplication**: Consecutive frame window to avoid duplicate peaks
- **Waveform Plots**: Matplotlib-generated visualizations saved to disk
- **Session Management**: Unique session IDs for data tracking

## Performance Characteristics

### Processing Metrics
- **Capture Rate**: 1-10 FPS (configurable, default: 10 FPS)
- **Buffer Size**: 100 frames circular buffer
- **Memory Usage**: Fixed-size buffers to prevent memory leaks
- **Processing Time**: <100ms per frame for typical ROI sizes

### Data Persistence
- **Log Rotation**: Daily log file rotation with timestamps
- **CSV Format**: Structured data export for analysis
- **Waveform Images**: PNG format plots for visual verification
- **Session Tracking**: Unique identifiers for data provenance

## Development Guidelines

### Module Dependencies
Core dependencies (automatically bundled with PyInstaller):
- `numpy` - Numerical computations and array operations
- `opencv-python` - Computer vision and image processing
- `Pillow` - Screen capture and image manipulation
- `matplotlib` - Waveform visualization and plotting

### Local Module Structure
```
SimpleFEM/
├── simple_roi_daemon.py      # Main application entry point with video/screen support
├── green_detector.py         # OpenCV-based green line detection (HSV filtering)
├── peak_detection.py         # Core peak detection with adaptive thresholding
├── improved_peak_detection.py # Enhanced detection algorithms (optional)
├── safe_peak_statistics.py   # Data export, deduplication, and CSV generation
├── simple_peak_test.py       # Peak detection testing and validation
├── simple_fem_config.json    # Hierarchical configuration with video/screen modes
├── 打包命令.txt               # PyInstaller build command
└── dist/                     # Build output directory
    └── SimpleFEM_ROI_Daemon.exe  # Standalone executable
```

### Error Handling Patterns
- **Graceful Degradation**: Continue processing even if individual components fail
- **Resource Cleanup**: Automatic memory management and file handle closure
- **Logging**: Comprehensive error logging with timestamps
- **Validation**: Input validation for all external data sources

### Testing Approach
- **Validation Script**: `verify_implementation.py` for module integrity
- **Manual Testing**: Interactive testing with real screen captures
- **Component Testing**: Individual module testing with Python imports
- **Integration Testing**: End-to-end workflow validation

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