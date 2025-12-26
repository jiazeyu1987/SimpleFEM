# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimpleFEM is a standalone ROI (Region of Interest) capture and peak detection daemon for the NHEM (New HEM Monitor) medical signal processing system. It operates independently from the main NHEM system, providing focused analysis capabilities for HEM (高回声事件 - High Echo Event) detection through computer vision and signal processing. The system features dual input modes (screen capture and video file processing) with medical-grade reliability and comprehensive data export capabilities.

### Key System Concepts

- **ROI1**: Large capture region (default: 1280x80 to 1920x980) for green line detection and peak detection
- **ROI2**: Small region extracted around green line intersection or vein tracking (default: ~80x120 pixels)
- **ROI3**: Extended vertical region for additional color classification validation
- **Green/Red Classification**: Peaks classified as GREEN when (post_peak_avg - pre_peak_avg) >= difference_threshold, otherwise RED
- **Hybrid Detection**: ROI1 detects peaks, ROI2 performs color classification (configurable via `hybrid_detection.enabled`)

## Development Commands

### Running SimpleFEM
```bash
# Main daemon - HEM detection
python simple_roi_daemon.py

# Vein detection and following system
python auto_vein_detector.py

# Configuration GUI for visual ROI setup and parameter tuning
python config_gui.py

# Run the built standalone executable (after building)
dist\SimpleFEM_ROI_Daemon.exe

# Test improved peak detection algorithms
python improved_peak_detection.py

# Test anti-jitter implementations
python test_anti_jitter.py
python manual_test_anti_jitter.py
```

### Building Standalone Executable
```bash
# Install PyInstaller (if not already installed)
pip install pyinstaller

# Build Windows executable using the provided command
pyinstaller --onefile --console --name SimpleFEM_ROI_Daemon --add-data "simple_fem_config.json;." simple_roi_daemon.py

# View build command reference
type 打包命令.txt

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Testing and Development
```bash
# Verify core module imports
python -c "from green_detector import detect_green_intersection; print('Green detector OK')"
python -c "from peak_detection import detect_peaks; print('Peak detection OK')"
python -c "from safe_peak_statistics import SafePeakStatistics; print('Statistics OK')"

# Test peak detection algorithms
python improved_peak_detection.py

# Test ROI1 implementations and thresholds
python test_roi1_implementation.py
python test_roi1_threshold.py
python test_roi1_simple.py

# Test anti-jitter and stability fixes
python test_anti_jitter.py
python manual_test_anti_jitter.py
python verify_anti_jitter_simple.py

# Test multi-video processing
python test_multi_video_fix.py

# Debug and diagnose issues
python diagnose_basic.py
python debug_wave1.py
python analyze_roi2_jitter.py

# Test frame difference and signal validation
python simple_frame_diff_test.py
python proper_frame_diff_test.py
python extreme_frame_diff_test.py
python correct_signal_test.py
python create_test_sequence.py

# Test ROI2 sampling window and anti-jitter
python test_roi2_sampling_window.py
python check_jitter_simple.py
python final_filter_test.py
```

### Python Dependencies
```bash
# Core dependencies (already bundled in executable)
pip install numpy opencv-python Pillow matplotlib

# Optional: For development and testing
pip install pyinstaller  # For building standalone executable
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
1. **Main Daemon** (`simple_roi_daemon.py`) - Primary orchestrator with video/screen capture, batch video processing, threshold protection, and analysis cache (JSONL)
2. **Configuration GUI** (`config_gui.py`) - Visual interface for ROI setup, parameter tuning, and real-time preview with zoom/pan controls
3. **Green Line Detection** (`green_detector.py`) - OpenCV HSV filtering with Canny edge detection, Hough line transformation, and EMA/velocity-based intersection filtering
4. **Peak Detection** (`peak_detection.py`) - Signal processing with adaptive thresholding, green/red classification, and ROI3 override support
5. **Improved Peak Detection** (`improved_peak_detection.py`) - Enhanced algorithms with signal preprocessing and lifecycle tracking
6. **Statistics Management** (`safe_peak_statistics.py`) - Three-layer deduplication with atomic CSV export, batch mode session management, and ROI1 peak ID tracking
7. **Vein Detection** (`auto_vein_detector.py`) - Connected component analysis with automatic ROI tracking for vein following

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
                                     Intersection Point Calculation → ROI2/ROI3 Extraction
                                                              ↓
                                     Grayscale Value Computation → Circular Buffer (100 frames)
                                                              ↓
                                     Adaptive Peak Detection → Green/Red Classification (+ROI3 Override)
                                                              ↓
                                     Three-Layer Deduplication → Analysis Cache (JSONL) + CSV Export + Logging
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

- **Computer Vision**: OpenCV HSV filtering, Canny edge detection, Hough line transformation, connected component analysis, EMA filtering
- **Image Processing**: PIL/Pillow screen capture, numpy array operations, ROI coordinate transformation, histogram-based averaging
- **Signal Processing**: Adaptive thresholding with threshold protection, morphological operations, peak lifecycle tracking, frame difference analysis
- **Data Management**: Thread-safe collections, atomic file operations, temporary file handling, session tracking, JSONL analysis cache
- **Configuration Management**: JSON-based configuration with NHEM_* environment variable overrides, visual GUI editor
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
    "save_roi3": true,              // Save extended ROI3 for color validation
    "save_wave": true,              // Save waveform plots
    "save_roi1_wave": true,         // Save ROI1 waveform data
    "only_delect": false            // Only save frames with detected peaks
  },
  "analysis_cache": {
    "enabled": true,                // Save per-frame analysis cache (JSONL) for debugging
    "flush_every": 50               // Flush cache every N frames
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

### 3.1 Anti-Jitter and ROI Stabilization
SimpleFEM includes multiple anti-jitter mechanisms to ensure stable ROI tracking:

#### Intersection Filtering (`green_detector.py`):
- **EMA Smoothing**: Exponential moving average for intersection point filtering
- **Velocity Filtering**: Rejects intersection movements exceeding configurable velocity thresholds
- **Threshold-Based Filtering**: Fixed and adaptive threshold methods for ROI stabilization

#### Threshold-Based Anti-Jitter (`threshold_based_anti_jitter.py`):
- **Pixel Intensity Analysis**: Analyzes pixel distribution within ROI2
- **Adaptive Thresholding**: Dynamic threshold calculation based on local statistics
- **Movement Validation**: Validates ROI movements based on pixel intensity patterns

#### Multi-Algorithm Support:
- **Hybrid Detection**: ROI1 detects peaks, ROI2 performs color classification
- **Consistency Validation**: Cross-validation between different detection methods
- **Fallback Mechanisms**: Graceful degradation when primary detection fails

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
2. **Consecutive Frame Deduplication**: Same-color peak filtering with configurable windows (default: 40 frames)
3. **Cross-Color Deduplication**: Different-color peaks at same frame keep higher priority (green > red)
4. **Invalid Data Filtering**: Zero average values and malformed peak rejection
- **Color Priority**: Green peaks (priority 2) take precedence over red (priority 1) in conflicts
- **ROI1 Peak ID Tracking**: Unique IDs prevent duplicate processing in hybrid detection mode
- **Atomic CSV Export**: Thread-safe writing with temporary files

### 6. Data Export and Session Management
- **Analysis Cache (JSONL)**: Per-frame analysis data in `export/roi_analysis_cache_*.jsonl` for debugging
- **CSV Format**: Comprehensive metadata (timestamp, frame indices, coordinates, ROI info, peak IDs)
- **Session Tracking**: Unique IDs with video name for batch processing mode
- **File Organization**: Structured output directories (`/exports/`, `/roi1/`, `/roi2/`, `/roi3/`, `/wave/`, `/logs/`)
- **Waveform Visualization**: Matplotlib plots with peak annotations and classification
- **Batch Mode Support**: Each video generates separate CSV with sanitized filename and timestamp

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
  - `tkinter` - GUI for configuration management (built into Python)

### Modular Code Structure
```
SimpleFEM/
├── Core Processing Modules/
│   ├── simple_roi_daemon.py      # Main orchestrator with dual input modes, analysis cache
│   ├── config_gui.py             # Visual configuration editor with ROI preview
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
│   ├── Algorithm Testing/
│   │   ├── improved_peak_detection.py  # Enhanced peak detection algorithms
│   │   ├── test_anti_jitter.py         # Anti-jitter mechanism testing
│   │   ├── test_roi1_*.py             # ROI1 implementation variants
│   │   └── test_roi3_*.py             # ROI3 override testing
│   ├── Debug and Diagnosis/
│   │   ├── diagnose_basic.py           # Basic system diagnostics
│   │   ├── debug_wave1.py              # Waveform debugging
│   │   └── analyze_roi2_jitter.py      # ROI2 jitter analysis
│   │   ├── trace_data_flow.py          # Data flow tracing
│   │   └── final_diagnosis.py          # Comprehensive diagnosis
│   └── Validation/
│       ├── verify_*.py                 # Implementation verification scripts
│       └── manual_test_anti_jitter.py  # Manual anti-jitter testing
├── Build System/
│   ├── 打包命令.txt               # PyInstaller build command
│   ├── requirements.txt           # Python dependencies
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

### Core System Characteristics
- **Dual Input Modes**: Supports both real-time screen capture and video file processing
- **Adaptive Thresholding**: Background-based threshold calculation with configurable ratios
- **Peak Classification**: Green (post-peak > pre-peak + threshold) vs Red (all other peaks)
- **Standalone Operation**: No network dependencies or external services required
- **Windows Platform**: Designed specifically for Windows screen capture capabilities using PIL.ImageGrab
- **Resource Efficiency**: Fixed circular buffers prevent memory leaks during continuous operation
- **Medical Reliability**: Built for medical applications with comprehensive error handling
- **Configuration Management**: Runtime configuration changes via JSON without restart
- **Data Security**: Local processing with no external data transmission
- **Build System**: PyInstaller creates standalone executables with embedded configuration

### Recent System Improvements
- **Anti-Jitter Mechanisms**: Multiple filtering algorithms (EMA, velocity, threshold-based) for stable ROI tracking
- **Enhanced Peak Detection**: Improved algorithms with lifecycle tracking and morphological operations
- **Multi-Video Support**: Batch processing with automatic video switching and session management
- **Threshold Protection**: Prevents peak contamination of background calculations with configurable recovery delays
- **Hybrid Detection**: Separate ROI1 (peak detection) and ROI2 (color classification) pipelines with fusion strategies
- **ROI3 Override Support**: Extended vertical region for additional color classification validation
- **Analysis Cache**: Per-frame JSONL logging for debugging and analysis reproducibility
- **Configuration GUI**: Visual interface for ROI setup with real-time preview, zoom, and pan controls
- **Velocity Filtering**: Rejects rapid intersection movements that indicate detection instability
- **Consistency Validation**: Cross-validation between different detection methods

### Development Environment Considerations
- **Python Dependencies**: Core modules (numpy, opencv-python, Pillow, matplotlib) are bundled in executable
- **Testing Infrastructure**: Extensive test suite with debugging and diagnostic tools
- **Windows-Specific**: Screen capture functionality requires Windows platform
- **Resource Management**: Automatic cleanup of memory, file handles, and video resources
- **Thread Safety**: Concurrent operations with proper synchronization mechanisms

## Common Development Issues and Solutions

### Debugging Peak Detection
When peak detection is not working as expected:
1. **Check Threshold Values**: The `threshold` in config may be too high/low for your video content
2. **Enable ROI1 Waveform Saving**: Set `save_roi1_wave: true` to visualize ROI1 waveform data
3. **Enable Analysis Cache**: Set `analysis_cache.enabled: true` to generate JSONL files for detailed frame-by-frame analysis
4. **Verify Green Line Detection**: Check if ROI1 coordinates align with actual green line position
5. **Review Adaptive Threshold**: If enabled, check `adaptive_window_seconds` and `threshold_over_mean_ratio`
6. **Check Threshold Protection**: Ensure `threshold_protection.enabled` is not preventing background updates
7. **Use Diagnostic Scripts**: Run `diagnose_basic.py`, `trace_data_flow.py`, or `final_diagnosis.py` for debugging

### ROI2 Jitter Issues
If ROI2 region is unstable or jumping:
1. **Enable Anti-Jitter**: Set `roi2_anti_jitter.enabled: true` in config
2. **Choose Algorithm**: Set `roi2_anti_jitter.algorithm` to 'threshold' (no small movements) or 'ema' (smooth filtering)
3. **Adjust Movement Threshold**: For 'threshold' algorithm, increase `movement_threshold` (default: 20.0 pixels)
4. **Tune EMA Settings**: For 'ema' algorithm, adjust `alpha` (default: 0.25, lower = more smoothing)
5. **Verify Green Line Detection**: Instability often originates from upstream green line detection
6. **Check Velocity Filtering**: Review `green_detector.py` for velocity-based filtering parameters

### CSV Export Issues
When CSV exports are missing or incomplete:
1. **Check Deduplication**: Peaks may be filtered by `consecutive_frame_window` (default: 40) or `cross_color_deduplication_enabled`
2. **Verify Only-Detect Mode**: If `only_delect: true`, only frames with peaks generate output files
3. **Review Color Priority**: Green peaks (priority 2) override red peaks (priority 1) in deduplication
4. **Check Export Directory**: CSV files are written to `export/` folder with `peak_statistics_*.csv` naming
5. **Analysis Cache**: Check `roi_analysis_cache_*.jsonl` files for complete frame-by-frame data
6. **Batch Mode**: Each video generates a separate CSV with video name prefix in batch processing

### Video Processing Problems
For video playback or processing issues:
1. **Video Path Configuration**: `video_path` can be a single file or directory (processes all videos)
2. **OpenCV Codec**: Ensure your OpenCV build supports the video codec (H.264 recommended)
3. **Frame Rate Override**: Use `processing_frame_rate` to limit processing speed
4. **Loop Mode**: `loop_enabled: true` will restart video when complete

### Multi-Video Batch Processing
To process multiple videos sequentially:
1. **Set Video Path as Directory**: `video_path: "video"` will process all files in that folder
2. **Session Management**: Each video generates a new session with unique ID
3. **CSV Naming**: Format includes video name: `peak_statistics_{video_name}_{timestamp}.csv`
4. **Auto-Cleanup**: `startup_cleanup.enabled: true` clears old data between runs

## Configuration Quick Reference

### Using the Configuration GUI
The visual configuration editor (`config_gui.py`) provides:
- **ROI Visual Setup**: Load ROI1/ROI2/ROI3 images and visualize current configuration
- **Real-time Preview**: See ROI regions overlaid on actual captured images
- **Zoom and Pan**: Mouse wheel to zoom (Y-axis for waveform), click-drag to pan
- **Image Navigation**: Browse through sequence of ROI1 captures
- **Parameter Tuning**: Adjust all configuration parameters with immediate visual feedback
- **Save/Reload**: Save changes to JSON and reload configuration

**Best Practices**:
- Run `config_gui.py` after initial screen capture to visually verify ROI alignment
- Use zoom/pan to inspect fine details of green line intersection areas
- Navigate through multiple ROI1 images to ensure configuration works across different frames

### Key Parameters for Tuning
| Parameter | Location | Purpose | Typical Range |
|-----------|----------|---------|---------------|
| `threshold` | `peak_detection.threshold` | Base detection threshold | 30-100 (depends on video) |
| `difference_threshold` | `peak_detection.difference_threshold` | Green/red classification | 1.5-3.0 |
| `frame_rate` | `roi_capture.frame_rate` | Capture/processing speed | 5-30 FPS |
| `consecutive_frame_window` | `deduplication.consecutive_frame_window` | Peak deduplication | 10-50 frames |
| `movement_threshold` | `roi2_anti_jitter.movement_threshold` | ROI2 stability | 10-30 pixels |
| `roi2_anti_jitter.algorithm` | `roi2_anti_jitter.algorithm` | Anti-jitter method | 'threshold' or 'ema' |
| `analysis_cache.enabled` | `analysis_cache.enabled` | Debug logging | true/false |
| `hybrid_detection.enabled` | `hybrid_detection.enabled` | Hybrid mode | true/false |

### Mode Switching
Change `processing_mode` in `simple_fem_config.json`:
- `"screen"`: Real-time screen capture (requires PIL.ImageGrab)
- `"video"`: Process video file(s) from `video_path`
- `"vein_following"`: Automatic vein tracking mode

### Important Constraints
- **ROI1 must contain visible green line** for intersection detection to work
- **ROI2 coordinates are relative to ROI1**, not absolute screen coordinates
- **ROI3 coordinates are relative to ROI1**, extended vertically for color validation
- **Circular buffer size is fixed at 100 frames** - affects memory and detection window
- **Environment variables use NHEM_ prefix** and override JSON config values
- **Analysis Cache files can grow large** - disable in production if not needed for debugging
- **Video path as directory processes all videos** - for batch processing, ensure directory contains only target videos
- **PyInstaller bundles configuration** - modify `simple_fem_config.json` before building executable

## Data Formats and File Organization

### Output Directory Structure
```
SimpleFEM/
├── export/                        # All exported data and analysis results
│   ├── peak_statistics_{session_id}.csv        # Peak detection results
│   ├── roi_analysis_cache_{session_id}_{run_id}.jsonl  # Per-frame analysis cache
│   └── hybrid_peak_statistics_{session_id}.csv # Hybrid detection results (if enabled)
├── roi1/                          # ROI1 large region captures
│   └── roi1_{frame_index:06d}.png
├── roi2/                          # ROI2 small region extracts
│   └── roi2_{frame_index:06d}.png
├── roi3/                          # ROI3 extended vertical regions (if enabled)
│   └── roi3_{frame_index:06d}.png
├── wave/                          # Waveform visualization plots
│   ├── wave_{frame_index:06d}.png
│   └── roi1_wave_{frame_index:06d}.png  # ROI1 waveform data (if enabled)
├── logs/                          # Daily log files with rotation
│   └── roi_peak_daemon_{YYYY-MM-DD}.log
└── tmp/                           # Temporary files (auto-cleaned)
```

### CSV Export Format
**Peak Statistics CSV** (`peak_statistics_*.csv`):
```csv
timestamp,frame_index,peak_start,peak_end,peak_color,peak_value,pre_avg,post_avg,frame_difference,roi1_coords,roi2_coords,roi3_coords,session_id,roi1_peak_id
2025-12-26T10:30:45,349,345,350,green,142.3,45.2,67.8,22.6,"1280,80,1920,980","1340,120,1420,180","1330,100,1410,200","session_001",roi1_349_345
```

**Analysis Cache JSONL** (`roi_analysis_cache_*.jsonl`):
```jsonl
{"type":"meta","cache_version":1,"created_at":"2025-12-26T10:30:00","session_id":"session_001",...}
{"type":"frame","frame_index":349,"timestamp":"2025-12-26T10:30:45","roi1_avg":52.3,"roi2_avg":142.3,"intersection":{"x":1380,"y":150},...}
{"type":"session_end","ended_at":"2025-12-26T10:35:00","reason":"video_complete"}
```

### File Naming Conventions
- **CSV Files**: `peak_statistics_{video_name}_{timestamp}.csv` (batch mode) or `peak_statistics_{timestamp}.csv` (single mode)
- **ROI Images**: `{roi_type}_{frame_index:06d}.png` (6-digit zero-padded frame index)
- **Waveform Plots**: `wave_{frame_index:06d}.png` or `roi1_wave_{frame_index:06d}.png`
- **Analysis Cache**: `roi_analysis_cache_{session_id}_{run_id}.jsonl` (run_id is 12-char hex)
- **Log Files**: `roi_peak_daemon_{YYYY-MM-DD}.log` (daily rotation)
