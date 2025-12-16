"""
Automatic Vein Detection Following System

Extends SimpleFEM framework with real-time vein tracking capabilities.
Captures ROI2 regions, detects connected components at center point,
creates masks, and automatically moves ROI to follow detected veins.

Usage:
    python auto_vein_detector.py [--config vein_detection_config.json]
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import argparse
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageGrab
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import gc


@dataclass
class ROIState:
    """ROI coordinate state management for tracking."""
    current_coords: Tuple[int, int, int, int]
    previous_coords: Tuple[int, int, int, int]
    fallback_coords: Tuple[int, int, int, int]
    tracking_active: bool
    consecutive_failures: int

    def update_with_centroid(self, centroid: Tuple[int, int], roi_width: int, roi_height: int) -> None:
        """Update ROI coordinates to center on centroid while maintaining dimensions."""
        cx, cy = centroid
        self.current_coords = (
            cx - roi_width // 2,
            cy - roi_height // 2,
            cx + roi_width // 2,
            cy + roi_height // 2
        )

    def apply_screen_bounds(self, screen_width: int, screen_height: int) -> None:
        """Ensure ROI stays within screen boundaries."""
        x1, y1, x2, y2 = self.current_coords
        self.current_coords = (
            max(0, min(x1, screen_width - 1)),
            max(0, min(y1, screen_height - 1)),
            min(screen_width, max(x1 + 1, x2)),
            min(screen_height, max(y1 + 1, y2))
        )

    def reset_to_fallback(self) -> None:
        """Reset ROI to fallback coordinates."""
        self.current_coords = self.fallback_coords
        self.tracking_active = False
        self.consecutive_failures = 0

    def increment_failure_count(self) -> bool:
        """Increment failure count and return if max failures exceeded."""
        self.consecutive_failures += 1
        return self.consecutive_failures >= 5

    def get_center(self) -> Tuple[int, int]:
        """Get center point of current ROI."""
        x1, y1, x2, y2 = self.current_coords
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_dimensions(self) -> Tuple[int, int]:
        """Get current ROI dimensions."""
        x1, y1, x2, y2 = self.current_coords
        return (x2 - x1, y2 - y1)


@dataclass
class FrameResult:
    """Frame processing result tracking."""
    frame_index: int
    processing_time: float
    detection_successful: bool
    roi_coordinates: Optional[Tuple[int, int, int, int]]
    mask_saved: bool
    roi2_saved: bool
    error_message: Optional[str]
    roi2_image: Optional[Image.Image] = None
    mask: Optional[np.ndarray] = None
    centroid: Optional[Tuple[int, int]] = None

    def has_valid_detection(self) -> bool:
        """Check if frame resulted in successful vein detection."""
        return self.detection_successful and self.mask is not None and self.centroid is not None


@dataclass
class VeinDetectionConfig:
    """Configuration data model for vein detection with validation."""
    # Processing Configuration
    processing_mode: str = "screen"
    frame_rate: int = 10

    # ROI Configuration
    roi_x1: int = 1480
    roi_y1: int = 480
    roi_x2: int = 1580
    roi_y2: int = 580

    # Detection Parameters
    threshold_min: int = 0
    threshold_max: int = 10
    connectivity: int = 4
    min_component_size: int = 50

    # Tracking Parameters
    max_tracking_failures: int = 5
    fallback_to_previous: bool = True
    reset_on_failure: bool = False

    # File Processing
    save_roi2: bool = True
    save_masks: bool = True
    only_save_with_detection: bool = False

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate frame rate
        if self.frame_rate < 5 or self.frame_rate > 30:
            errors.append(f"Frame rate {self.frame_rate} must be between 5-30 FPS")

        # Validate threshold range
        if self.threshold_min < 0 or self.threshold_max > 255:
            errors.append(f"Threshold range {self.threshold_min}-{self.threshold_max} invalid")
        if self.threshold_min >= self.threshold_max:
            errors.append(f"threshold_min ({self.threshold_min}) must be less than threshold_max ({self.threshold_max})")

        # Validate ROI coordinates
        if self.roi_x2 <= self.roi_x1 or self.roi_y2 <= self.roi_y1:
            errors.append("Invalid ROI coordinates: x2 must be > x1 and y2 must be > y1")

        # Validate ROI size (maximum 200x200 pixels)
        roi_width = self.roi_x2 - self.roi_x1
        roi_height = self.roi_y2 - self.roi_y1
        if roi_width > 200 or roi_height > 200:
            errors.append(f"ROI size {roi_width}x{roi_height} exceeds maximum 200x200 pixels")

        # Validate connectivity
        if self.connectivity not in [4, 8]:
            errors.append(f"Connectivity must be 4 or 8, got {self.connectivity}")

        return errors

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VeinDetectionConfig':
        """Create config instance from dictionary."""
        roi_config = config_dict.get("roi_capture", {}).get("roi2_config", {})
        vein_config = config_dict.get("vein_detection", {})
        data_config = config_dict.get("data_processing", {})

        return cls(
            processing_mode=config_dict.get("processing_mode", "screen"),
            frame_rate=config_dict.get("roi_capture", {}).get("frame_rate", 10),
            roi_x1=roi_config.get("x1", 1480),
            roi_y1=roi_config.get("y1", 480),
            roi_x2=roi_config.get("x2", 1580),
            roi_y2=roi_config.get("y2", 580),
            threshold_min=vein_config.get("threshold_min", 0),
            threshold_max=vein_config.get("threshold_max", 10),
            connectivity=vein_config.get("connectivity", 4),
            min_component_size=vein_config.get("min_component_size", 50),
            max_tracking_failures=vein_config.get("tracking", {}).get("max_tracking_failures", 5),
            fallback_to_previous=vein_config.get("tracking", {}).get("fallback_to_previous", True),
            reset_on_failure=vein_config.get("tracking", {}).get("reset_on_failure", False),
            save_roi2=data_config.get("save_roi2", True),
            save_masks=data_config.get("save_masks", True),
            only_save_with_detection=data_config.get("only_save_with_detection", False)
        )


def _get_base_dir() -> str:
    """
    Resolve base directory both for source (.py) and frozen (.exe) modes.

    When packaged with PyInstaller, sys.frozen is True and sys.executable
    points to the .exe location. In source mode, use this file's directory.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _get_base_dir()


def _setup_import_paths() -> None:
    """
    Ensure we can import local modules if needed.
    """
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)


_setup_import_paths()


class VeinDetectionEngine:
    """Main class for automatic vein detection and ROI tracking."""

    def __init__(self, config_path: str = None):
        """Initialize the vein detection engine."""
        self.config_path = config_path or os.path.join(BASE_DIR, "vein_detection_config.json")
        self.config = {}
        self.config_obj = None
        self.logger = None
        self.frame_index = 0
        self.running = False

        # Initialize state management
        self.roi_state = None
        self.processing_times = deque(maxlen=100)  # Track last 100 frame processing times

    def load_configuration(self) -> bool:
        """Load and validate configuration with error handling."""
        try:
            # Check if configuration file exists
            if not os.path.exists(self.config_path):
                print(f"Configuration file not found: {self.config_path}")
                print("Creating default configuration file...")
                self.create_default_config()
                return False

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            # Create configuration object and validate
            self.config_obj = VeinDetectionConfig.from_dict(self.config)
            errors = self.config_obj.validate()

            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")

                # Provide specific suggestions
                self.handle_configuration_errors(errors)
                return False

            print("Configuration loaded and validated successfully")
            return True

        except json.JSONDecodeError as e:
            print(f"Invalid JSON in configuration file: {e}")
            print("Suggestion: Check vein_detection_config.json for JSON syntax errors")
            return False
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def handle_configuration_errors(self, errors: List[str]) -> None:
        """Provide specific error messages with suggested corrections."""
        print("\nConfiguration Error Suggestions:")

        for error in errors:
            if "Frame rate" in error:
                print(f"  - {error}")
                print("    Suggestion: Set frame_rate between 5 and 30 in roi_capture section")
            elif "threshold range" in error or "threshold_min" in error:
                print(f"  - {error}")
                print("    Suggestion: Set threshold_min to 0 and threshold_max to 10 in vein_detection section")
            elif "Invalid ROI coordinates" in error:
                print(f"  - {error}")
                print("    Suggestion: Ensure x2 > x1 and y2 > y1 in roi2_config section")
            elif "ROI size" in error:
                print(f"  - {error}")
                print("    Suggestion: Ensure ROI dimensions are within 200x200 pixels")
            elif "Connectivity" in error:
                print(f"  - {error}")
                print("    Suggestion: Set connectivity to 4 or 8 in vein_detection section")
            else:
                print(f"  - {error}")
                print("    Suggestion: Check vein_detection_config.json format and values")

    def create_default_config(self) -> None:
        """Create a default configuration file."""
        default_config = {
            "processing_mode": "screen",
            "roi_capture": {
                "frame_rate": 10,
                "roi2_config": {
                    "x1": 1480,
                    "y1": 480,
                    "x2": 1580,
                    "y2": 580
                }
            },
            "vein_detection": {
                "threshold_min": 0,
                "threshold_max": 10,
                "connectivity": 4,
                "min_component_size": 50,
                "tracking": {
                    "max_tracking_failures": 5,
                    "fallback_to_previous": True,
                    "reset_on_failure": False
                }
            },
            "data_processing": {
                "save_roi2": True,
                "save_masks": True,
                "only_save_with_detection": False
            }
        }

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Default configuration created: {self.config_path}")
        except Exception as e:
            print(f"Failed to create default configuration: {e}")

    def setup_logging(self) -> None:
        """Setup logging following SimpleFEM patterns with TimedRotatingFileHandler."""
        self.logger = logging.getLogger("vein_detector")
        if self.logger.handlers:
            return  # Already configured

        self.logger.setLevel(logging.INFO)

        # Keep logs local to SimpleFEM project directory (same as simple_roi_daemon.py)
        log_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "vein_detector.log")

        # Use identical handler setup as simple_roi_daemon.py
        handler = logging.handlers.TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

        # Initialize ROI state after configuration is loaded
        self.initialize_roi_state()

        # Log startup
        self.logger.info("=== Vein Detection Following System Started ===")
        self.logger.info(f"Configuration file: {self.config_path}")
        self.logger.info(f"Frame rate: {self.config_obj.frame_rate} FPS")
        self.logger.info(f"ROI coordinates: ({self.config_obj.roi_x1}, {self.config_obj.roi_y1}) -> ({self.config_obj.roi_x2}, {self.config_obj.roi_y2})")
        self.logger.info(f"Detection threshold: {self.config_obj.threshold_min}-{self.config_obj.threshold_max}")

    def initialize_roi_state(self) -> None:
        """Initialize ROI state based on configuration."""
        if not self.config_obj:
            return

        initial_coords = (self.config_obj.roi_x1, self.config_obj.roi_y1, self.config_obj.roi_x2, self.config_obj.roi_y2)
        self.roi_state = ROIState(
            current_coords=initial_coords,
            previous_coords=initial_coords,
            fallback_coords=initial_coords,
            tracking_active=False,
            consecutive_failures=0
        )

        self.logger.info(f"ROI state initialized: {initial_coords}")

    def monitor_memory_usage(self) -> bool:
        """Monitor memory usage and trigger cleanup at 450MB threshold."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > 450:
                self.logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds 450MB threshold")
                self.trigger_memory_cleanup()
                return True
            return False
        except ImportError:
            # psutil not available, skip monitoring
            return False
        except Exception as e:
            self.logger.error(f"Memory monitoring failed: {e}")
            return False

    def trigger_memory_cleanup(self) -> None:
        """Perform memory cleanup and reduce footprint."""
        self.logger.info("Performing memory cleanup...")

        try:
            # Force garbage collection
            gc.collect()

            # Clear processing times buffer if it exists
            if hasattr(self, 'processing_times') and self.processing_times:
                self.processing_times.clear()

            # Clear any cached data (future tasks may add frame cache)
            if hasattr(self, 'frame_cache'):
                self.frame_cache.clear()

            self.logger.info("Memory cleanup completed")
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

    def add_processing_time(self, processing_time: float) -> None:
        """Add processing time to tracking buffer and monitor performance."""
        if hasattr(self, 'processing_times'):
            self.processing_times.append(processing_time)

            # Log average processing time every 50 frames
            if self.frame_index % 50 == 0 and len(self.processing_times) > 0:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                self.logger.info(f"Average processing time: {avg_time*1000:.1f}ms (~{current_fps:.1f} FPS)")

    def capture_roi2_region(self, coords: Optional[Tuple[int, int, int, int]] = None) -> Optional[Image.Image]:
        """Capture ROI2 region using PIL.ImageGrab following SimpleFEM patterns."""
        try:
            # Use provided coordinates or current ROI state coordinates
            if coords is None and self.roi_state:
                coords = self.roi_state.current_coords
            elif coords is None:
                coords = (self.config_obj.roi_x1, self.config_obj.roi_y1, self.config_obj.roi_x2, self.config_obj.roi_y2)

            if not coords:
                self.logger.error("No ROI coordinates available for capture")
                return None

            x1, y1, x2, y2 = coords

            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                self.logger.error(f"Invalid ROI coordinates: ({x1}, {y1}, {x2}, {y2})")
                return None

            # Capture screen using PIL.ImageGrab (same as simple_roi_daemon.py)
            screen = ImageGrab.grab()
            roi_image = screen.crop((x1, y1, x2, y2))

            self.logger.debug(f"Captured ROI2 region: {coords} -> size: {roi_image.size}")
            return roi_image

        except Exception as e:
            self.log_error("ROI2 capture failed", e)
            return None

    def get_screen_size(self) -> Tuple[int, int]:
        """Get current screen size."""
        try:
            screen = ImageGrab.grab()
            screen_size = screen.size
            self.logger.debug(f"Screen size detected: {screen_size}")
            return screen_size
        except Exception as e:
            self.log_error("Failed to get screen size", e)
            return (1920, 1080)  # Default fallback

    def enforce_roi_size_limits(self, coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Enforce maximum ROI size limit of 200x200 pixels."""
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1

        # Check if ROI exceeds maximum size
        if width > 200 or height > 200:
            self.logger.warning(f"ROI size {width}x{height} exceeds 200x200 limit, resizing...")

            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Ensure maximum dimensions
            new_width = min(width, 200)
            new_height = min(height, 200)

            # Recalculate coordinates centered on original center
            x1 = cx - new_width // 2
            y1 = cy - new_height // 2
            x2 = cx + new_width // 2
            y2 = cy + new_height // 2

            coords = (x1, y1, x2, y2)
            self.logger.info(f"ROI resized to {coords} ({new_width}x{new_height})")

        return coords

    def apply_all_validations(self, coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply all coordinate validations in sequence."""
        # 1. Enforce size limits
        coords = self.enforce_roi_size_limits(coords)

        # 2. Apply screen bounds
        coords = self.validate_roi_coordinates(coords)

        return coords

    def validate_roi_coordinates(self, coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Validate and adjust ROI coordinates to stay within screen bounds."""
        screen_width, screen_height = self.get_screen_size()
        x1, y1, x2, y2 = coords

        # Adjust coordinates to stay within screen bounds (same logic as adjust_roi1_to_screen)
        adjusted_coords = (
            max(0, min(x1, screen_width - 1)),
            max(0, min(y1, screen_height - 1)),
            min(screen_width, max(x1 + 1, x2)),
            min(screen_height, max(y1 + 1, y2))
        )

        if adjusted_coords != coords:
            self.logger.warning(f"ROI coordinates adjusted: {coords} -> {adjusted_coords}")

        return adjusted_coords

    def update_roi_state_for_capture(self) -> Tuple[int, int, int, int]:
        """Update ROI state for screen capture and return validated coordinates."""
        if not self.roi_state:
            self.logger.error("ROI state not initialized")
            return (0, 0, 100, 100)

        # Apply screen bounds validation to current coordinates
        validated_coords = self.validate_roi_coordinates(self.roi_state.current_coords)

        # Update previous coordinates before capture
        self.roi_state.previous_coords = self.roi_state.current_coords
        self.roi_state.current_coords = validated_coords

        return validated_coords

    def log_frame_result(self, frame_index: int, detection_success: bool, centroid: Optional[Tuple[int, int]] = None, processing_time: float = 0.0) -> None:
        """Log frame processing result."""
        if detection_success and centroid:
            self.logger.info(f"Frame {frame_index:6d}: ✓ Detection at ({centroid[0]}, {centroid[1]}) ({processing_time*1000:.1f}ms)")
        else:
            self.logger.info(f"Frame {frame_index:6d}: ✗ No detection ({processing_time*1000:.1f}ms)")

    def apply_threshold_filter(self, image: np.ndarray, min_val: int = 0, max_val: int = 10) -> np.ndarray:
        """Apply threshold filtering for vein detection (0-10 pixel range)."""
        try:
            # Validate threshold range
            if min_val < 0 or max_val > 255:
                self.logger.error(f"Invalid threshold range: {min_val}-{max_val}")
                min_val = max(0, min_val)
                max_val = min(255, max_val)

            # Convert image to grayscale if needed
            if len(image.shape) == 3:
                # RGB image, convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 4:
                # RGBA image, convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                # Already grayscale
                gray = image.copy()

            # Apply threshold filter (same pattern as green_detector.py)
            mask = cv2.inRange(gray, min_val, max_val)

            self.logger.debug(f"Threshold filter applied: {min_val}-{max_val}, positive pixels: {np.sum(mask > 0)}")
            return mask

        except Exception as e:
            self.log_error("Threshold filtering failed", e)
            # Return empty mask on error
            return np.zeros_like(image)

    def analyze_connected_components(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """Perform connected component analysis using OpenCV with performance optimization."""
        try:
            if self.config_obj is None:
                self.logger.error("Configuration not loaded for component analysis")
                return None, 0

            # Apply threshold filter first
            filtered_mask = self.apply_threshold_filter(
                image,
                self.config_obj.threshold_min,
                self.config_obj.threshold_max
            )

            # Check if any pixels remain after filtering
            if np.sum(filtered_mask) == 0:
                self.logger.debug("No pixels found in threshold range")
                return None, 0

            # Perform connected component analysis with optimized parameters
            connectivity = self.config_obj.connectivity  # 4 or 8
            min_component_size = self.config_obj.min_component_size

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                filtered_mask,
                connectivity=connectivity,
                ltype=cv2.CV_32S
            )

            self.logger.debug(f"Connected components found: {num_labels-1} (excluding background)")

            # Filter by minimum component size
            if min_component_size > 0:
                valid_labels = []
                for i in range(1, num_labels):  # Skip label 0 (background)
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_component_size:
                        valid_labels.append(i)
                self.logger.debug(f"Components after size filter: {len(valid_labels)}")

                # Create filtered mask
                filtered_result = np.zeros_like(filtered_mask)
                for label in valid_labels:
                    filtered_result[labels == label] = 255

                return filtered_result, num_labels
            else:
                # No size filtering, return all components
                return (filtered_mask > 0).astype(np.uint8) * 255, num_labels

        except Exception as e:
            self.log_error("Connected component analysis failed", e)
            return None, 0

    def extract_center_component(self, labels: np.ndarray, center: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract the connected component containing the specified center point."""
        try:
            if center is None:
                self.logger.error("No center point provided for component extraction")
                return None

            cx, cy = center
            height, width = labels.shape

            # Validate center coordinates
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                self.logger.warning(f"Center point {center} outside image bounds {width}x{height}")
                return None

            # Get the label at the center point
            center_label = labels[cy, cx]

            if center_label == 0:
                # No component at center point
                self.logger.debug(f"No connected component at center point ({cx}, {cy})")
                return None

            # Create binary mask for the center component
            center_mask = (labels == center_label).astype(np.uint8) * 255

            self.logger.debug(f"Center component extracted: label {center_label}, pixels: {np.sum(center_mask > 0)}")
            return center_mask

        except Exception as e:
            self.log_error("Center component extraction failed", e)
            return None

    def create_binary_mask(self, component_mask: np.ndarray) -> np.ndarray:
        """Create and validate binary mask from component mask."""
        try:
            if component_mask is None:
                self.logger.error("No component mask provided for binary mask creation")
                return np.zeros((100, 100), dtype=np.uint8)

            # Ensure binary format (0 or 255)
            binary_mask = ((component_mask > 0) * 255).astype(np.uint8)

            self.logger.debug(f"Binary mask created: {binary_mask.shape}, positive pixels: {np.sum(binary_mask > 0)}")
            return binary_mask

        except Exception as e:
            self.log_error("Binary mask creation failed", e)
            return np.zeros((100, 100), dtype=np.uint8)

    def log_error(self, message: str, error: Exception = None) -> None:
        """Log error with optional exception details."""
        if error:
            self.logger.error(f"{message}: {error}")
        else:
            self.logger.error(message)

    def calculate_mask_centroid(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Calculate mask centroid using OpenCV moments with error handling for empty masks."""
        try:
            if mask is None or mask.size == 0:
                self.logger.error("Empty or None mask provided for centroid calculation")
                return None

            # Ensure binary format
            if mask.dtype != np.uint8:
                mask = ((mask > 0) * 255).astype(np.uint8)

            # Calculate moments
            moments = cv2.moments(mask, binaryImage=True)

            # Check if mask has any content
            if moments['m00'] == 0:
                self.logger.debug("Empty mask: no centroid can be calculated")
                return None

            # Calculate centroid
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            self.logger.debug(f"Centroid calculated: ({cx}, {cy}) for mask with {np.sum(mask > 0)} pixels")
            return (cx, cy)

        except Exception as e:
            self.log_error("Mask centroid calculation failed", e)
            return None

    def update_roi_with_centroid(self, centroid: Tuple[int, int]) -> bool:
        """Update ROI coordinates to center on mask centroid while maintaining dimensions."""
        try:
            if not self.roi_state or not centroid:
                return False

            # Get current ROI dimensions
            width, height = self.roi_state.get_dimensions()

            # Update ROI coordinates to center on centroid
            self.roi_state.update_with_centroid(centroid, width, height)

            # Apply screen bounds validation
            screen_width, screen_height = self.get_screen_size()
            self.roi_state.apply_screen_bounds(screen_width, screen_height)

            # Mark as active tracking
            self.roi_state.tracking_active = True
            self.roi_state.consecutive_failures = 0

            new_coords = self.roi_state.current_coords
            self.logger.info(f"ROI updated to center on centroid: {new_coords}")

            return True

        except Exception as e:
            self.log_error("ROI update with centroid failed", e)
            return False

    def get_roi_center(self) -> Optional[Tuple[int, int]]:
        """Get current ROI center point."""
        if not self.roi_state:
            return None
        return self.roi_state.get_center()

    def handle_detection_failure(self, frame_index: int) -> None:
        """Handle mask detection failure with automatic recovery."""
        try:
            if not self.roi_state:
                return

            self.roi_state.consecutive_failures += 1
            self.logger.warning(f"Frame {frame_index}: Mask detection failed (attempt {self.roi_state.consecutive_failures})")

            # Recovery strategy
            max_failures = self.config_obj.max_tracking_failures if self.config_obj else 5

            if self.roi_state.consecutive_failures <= max_failures:
                # Use previous ROI position
                self.roi_state.current_coords = self.roi_state.previous_coords
                self.roi_state.tracking_active = False
                self.logger.info(f"Recovery: Using previous ROI position {self.roi_state.current_coords}")
            else:
                # Reset to fallback coordinates
                self.roi_state.reset_to_fallback()
                self.logger.warning(f"Recovery: Reset to fallback coordinates {self.roi_state.fallback_coords}")

        except Exception as e:
            self.log_error("Detection failure handling failed", e)

    def shutdown_logging(self) -> None:
        """Log shutdown message."""
        if self.logger:
            self.logger.info("=== Vein Detection Following System Stopped ===")
            self.logger.info(f"Total frames processed: {self.frame_index}")

    def setup_output_directories(self) -> bool:
        """Setup output directories for ROI2 images and masks."""
        try:
            base_dir = os.path.join(BASE_DIR, "vein_detection_output")

            # Create main output directory
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                self.logger.info(f"Created output directory: {base_dir}")

            # Create ROI2 subdirectory if enabled
            if self.config_obj and self.config_obj.save_roi2:
                roi2_dir = os.path.join(base_dir, "roi2")
                if not os.path.exists(roi2_dir):
                    os.makedirs(roi2_dir)
                    self.logger.info(f"Created ROI2 directory: {roi2_dir}")

            # Create masks subdirectory if enabled
            if self.config_obj and self.config_obj.save_masks:
                masks_dir = os.path.join(base_dir, "masks")
                if not os.path.exists(masks_dir):
                    os.makedirs(masks_dir)
                    self.logger.info(f"Created masks directory: {masks_dir}")

            return True

        except Exception as e:
            if self.logger:
                self.log_error("Failed to setup output directories", e)
            else:
                print(f"Failed to setup output directories: {e}")
            return False

    def get_output_filename(self, file_type: str, frame_index: int, create_subdir: bool = False) -> str:
        """Generate output filename following SimpleFEM conventions.

        Args:
            file_type: Type of file ('roi2' or 'mask')
            frame_index: Current frame index
            create_subdir: Whether to create subdirectories

        Returns:
            Full path to output file
        """
        try:
            base_dir = os.path.join(BASE_DIR, "vein_detection_output")

            if create_subdir:
                file_dir = os.path.join(base_dir, file_type)
            else:
                file_dir = base_dir

            # Ensure directory exists
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # Generate filename with frame index padding (6 digits)
            filename = f"{file_type}_{frame_index:06d}.png"
            return os.path.join(file_dir, filename)

        except Exception as e:
            self.log_error(f"Failed to generate output filename for {file_type}", e)
            return ""

    def validate_output_path(self, file_path: str) -> bool:
        """Validate that output path is writable and safe."""
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(file_path)

            # Check if path is within allowed output directory
            allowed_dir = os.path.abspath(os.path.join(BASE_DIR, "vein_detection_output"))
            if not abs_path.startswith(allowed_dir):
                self.logger.error(f"Output path outside allowed directory: {file_path}")
                return False

            # Check if parent directory exists and is writable
            parent_dir = os.path.dirname(abs_path)
            if not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir)
                except Exception as e:
                    self.log_error(f"Cannot create parent directory: {parent_dir}", e)
                    return False

            # Test write permission
            test_file = os.path.join(parent_dir, ".test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                self.log_error(f"No write permission in directory: {parent_dir}", e)
                return False

            return True

        except Exception as e:
            self.log_error(f"Path validation failed for: {file_path}", e)
            return False

    def save_roi2_image(self, roi2_image: np.ndarray, frame_index: int) -> bool:
        """Save ROI2 image using exact SimpleFEM naming convention.

        Args:
            roi2_image: ROI2 image as numpy array
            frame_index: Current frame index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if ROI2 saving is enabled
            if not self.config_obj or not self.config_obj.save_roi2:
                return True  # Not an error, just disabled

            # Check if we should only save with detection
            if self.config_obj.only_save_with_detection and not self.roi_state.tracking_active:
                return True  # Skip saving but don't error

            # Generate output filename using exact SimpleFEM convention
            output_path = self.get_output_filename("roi2", frame_index, create_subdir=True)

            if not output_path:
                self.logger.error("Failed to generate ROI2 output filename")
                return False

            # Validate output path
            if not self.validate_output_path(output_path):
                return False

            # Convert numpy array to PIL Image if needed
            if hasattr(roi2_image, 'shape'):
                # This is a numpy array
                image_array = roi2_image
                if len(image_array.shape) == 3:
                    # OpenCV BGR to RGB conversion
                    if image_array.shape[2] == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_array)
                else:
                    pil_image = Image.fromarray(image_array)
            else:
                # This is already a PIL Image
                pil_image = roi2_image

            # Save image
            pil_image.save(output_path, "PNG")

            self.logger.debug(f"ROI2 image saved: {output_path}")
            return True

        except Exception as e:
            self.log_error("Failed to save ROI2 image", e)
            return False

    def save_mask_image(self, mask: np.ndarray, frame_index: int) -> bool:
        """Save binary mask using matching naming convention.

        Args:
            mask: Binary mask as numpy array
            frame_index: Current frame index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if mask saving is enabled
            if not self.config_obj or not self.config_obj.save_masks:
                return True  # Not an error, just disabled

            # Check if we should only save with detection
            if self.config_obj.only_save_with_detection and not self.roi_state.tracking_active:
                return True  # Skip saving but don't error

            # Generate output filename using matching convention
            output_path = self.get_output_filename("mask", frame_index, create_subdir=True)

            if not output_path:
                self.logger.error("Failed to generate mask output filename")
                return False

            # Validate output path
            if not self.validate_output_path(output_path):
                return False

            # Ensure mask is in proper format for saving
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255

            # Convert to PIL Image
            pil_image = Image.fromarray(mask, mode='L')  # L for grayscale

            # Save mask as PNG
            pil_image.save(output_path, "PNG")

            self.logger.debug(f"Mask image saved: {output_path}")
            return True

        except Exception as e:
            self.log_error("Failed to save mask image", e)
            return False

    def handle_file_io_error(self, operation: str, file_path: str, error: Exception) -> None:
        """Handle file I/O errors with detailed logging and recovery suggestions.

        Args:
            operation: Description of the file operation being attempted
            file_path: Path to the file being accessed
            error: Exception that occurred
        """
        try:
            error_type = type(error).__name__
            error_msg = str(error)

            self.logger.error(f"File I/O error during {operation}: {error_type} - {error_msg}")
            self.logger.error(f"File path: {file_path}")

            # Provide specific recovery suggestions based on error type
            if isinstance(error, PermissionError):
                self.logger.error("Recovery suggestion: Check file/directory permissions")
                self.logger.error("  - Ensure the application has write access to the output directory")
                self.logger.error("  - Close any other programs that might be using the file")
                self.logger.error("  - Try running the application as administrator if needed")

            elif isinstance(error, FileNotFoundError):
                self.logger.error("Recovery suggestion: Check directory structure")
                self.logger.error("  - Ensure the parent directories exist")
                self.logger.error("  - Verify the output directory path is correct")

            elif isinstance(error, OSError):
                if "No space left on device" in error_msg:
                    self.logger.error("Recovery suggestion: Disk space issue")
                    self.logger.error("  - Free up disk space on the target drive")
                    self.logger.error("  - Move old output files to another location")
                elif "File name too long" in error_msg:
                    self.logger.error("Recovery suggestion: File path issue")
                    self.logger.error("  - Move output directory closer to root")
                    self.logger.error("  - Shorten the file path")

            elif isinstance(error, (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError)):
                self.logger.error("Recovery suggestion: Image format issue")
                self.logger.error("  - Check if the image data is valid")
                self.logger.error("  - Verify the image dimensions are reasonable")

            else:
                self.logger.error("Recovery suggestion: Generic file I/O issue")
                self.logger.error("  - Check if the file path is valid")
                self.logger.error("  - Ensure sufficient system resources")
                self.logger.error("  - Restart the application if the issue persists")

        except Exception as e:
            # Fallback error logging in case the error handler itself fails
            self.logger.error(f"Error in file I/O error handler: {e}")

    def safe_file_operation(self, operation: callable, operation_name: str, *args, **kwargs) -> Tuple[bool, any]:
        """Safely execute a file operation with comprehensive error handling.

        Args:
            operation: Function to execute
            operation_name: Description of the operation for logging
            *args, **kwargs: Arguments to pass to the operation function

        Returns:
            Tuple of (success: bool, result: any)
        """
        try:
            result = operation(*args, **kwargs)
            return True, result

        except Exception as e:
            # Extract file path from args if available
            file_path = "unknown"
            if args and hasattr(args[0], '__str__'):
                file_path = str(args[0])

            self.handle_file_io_error(operation_name, file_path, e)
            return False, None

    def cleanup_orphaned_files(self) -> None:
        """Clean up potentially orphaned files from previous sessions."""
        try:
            base_dir = os.path.join(BASE_DIR, "vein_detection_output")
            if not os.path.exists(base_dir):
                return

            self.logger.info("Cleaning up orphaned files from previous sessions...")

            # Look for incomplete files (0 bytes)
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getsize(file_path) == 0:
                            os.remove(file_path)
                            self.logger.debug(f"Removed empty file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove empty file {file_path}: {e}")

        except Exception as e:
            self.log_error("Failed to cleanup orphaned files", e)

    def control_frame_rate(self) -> None:
        """Control frame rate with configurable timing and processing monitoring."""
        try:
            if not self.config_obj:
                return

            target_frame_time = 1.0 / self.config_obj.frame_rate  # Convert FPS to seconds
            current_time = time.time()

            # Track processing time
            if not hasattr(self, 'last_frame_time'):
                self.last_frame_time = current_time
                return

            # Calculate actual elapsed time
            elapsed_time = current_time - self.last_frame_time

            # Log processing time performance
            if elapsed_time > target_frame_time * 1.5:  # 50% longer than expected
                self.logger.warning(
                    f"Frame processing exceeded target time: {elapsed_time:.3f}s vs {target_frame_time:.3f}s"
                )
            elif elapsed_time > target_frame_time:  # Any exceedance
                self.logger.debug(
                    f"Frame processing slow: {elapsed_time:.3f}s (target: {target_frame_time:.3f}s)"
                )

            # Calculate sleep time to maintain frame rate
            sleep_time = target_frame_time - elapsed_time

            if sleep_time > 0:
                # Sleep for remaining time to maintain consistent frame rate
                time.sleep(sleep_time)
                self.last_frame_time = time.time()
            else:
                # No sleep needed, but update timestamp for next frame
                self.last_frame_time = current_time

                # Log if we're running behind schedule
                if sleep_time < -target_frame_time:
                    frames_behind = int(abs(sleep_time) / target_frame_time)
                    self.logger.warning(
                        f"Processing is {frames_behind} frames behind schedule"
                    )

        except Exception as e:
            self.log_error("Frame rate control failed", e)

    def get_frame_rate_statistics(self) -> Dict[str, float]:
        """Calculate current frame rate statistics from processing times.

        Returns:
            Dictionary with frame rate metrics
        """
        try:
            if not hasattr(self, 'processing_times') or len(self.processing_times) == 0:
                return {
                    "current_fps": 0.0,
                    "average_processing_time": 0.0,
                    "max_processing_time": 0.0,
                    "min_processing_time": 0.0
                }

            # Calculate statistics from processing times
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            min_time = min(self.processing_times)

            # Calculate FPS from processing times
            current_fps = 1.0 / avg_time if avg_time > 0 else 0.0

            return {
                "current_fps": current_fps,
                "average_processing_time": avg_time,
                "max_processing_time": max_time,
                "min_processing_time": min_time,
                "frame_count": len(self.processing_times)
            }

        except Exception as e:
            self.log_error("Failed to calculate frame rate statistics", e)
            return {"error": str(e)}

    def validate_frame_rate_performance(self) -> bool:
        """Check if current performance meets requirements.

        Returns:
            True if performance is acceptable, False otherwise
        """
        try:
            stats = self.get_frame_rate_statistics()

            if "error" in stats:
                return False

            if not self.config_obj:
                return False

            # Check if we meet minimum FPS requirement
            target_fps = self.config_obj.frame_rate
            current_fps = stats["current_fps"]

            # Allow 10% tolerance for frame rate
            min_acceptable_fps = target_fps * 0.9

            if current_fps < min_acceptable_fps:
                self.logger.warning(
                    f"Performance below target: {current_fps:.1f} FPS (target: {target_fps} FPS)"
                )
                return False

            # Check processing time consistency
            avg_time = stats["average_processing_time"]
            target_time = 1.0 / target_fps

            if avg_time > target_time * 1.2:  # 20% tolerance
                self.logger.warning(
                    f"Processing time inconsistent: {avg_time:.3f}s (target: {target_time:.3f}s)"
                )
                return False

            return True

        except Exception as e:
            self.log_error("Frame rate performance validation failed", e)
            return False

    def process_single_frame(self) -> FrameResult:
        """Orchestrate single frame processing with all modules.

        Returns:
            FrameResult object containing processing results and metadata
        """
        start_time = time.time()

        try:
            # Initialize frame result
            frame_result = FrameResult(
                frame_index=self.frame_index,
                processing_time=0.0,
                detection_successful=False,
                roi_coordinates=None,
                mask_saved=False,
                roi2_saved=False,
                error_message=None
            )

            # Step 1: Capture ROI2 region
            roi2_image = self.capture_roi2_region()
            if roi2_image is None:
                frame_result.error_message = "Failed to capture ROI2 region"
                frame_result.processing_time = time.time() - start_time
                self.processing_times.append(frame_result.processing_time)
                return frame_result

            # Store ROI coordinates in result
            if self.roi_state:
                frame_result.roi_coordinates = self.roi_state.current_coords

            # Step 2: Apply threshold filtering for vein detection
            # Convert PIL Image to numpy array if needed
            if isinstance(roi2_image, Image.Image):
                roi2_array = np.array(roi2_image)
                # Convert to grayscale if needed
                if len(roi2_array.shape) == 3:
                    roi2_array = cv2.cvtColor(roi2_array, cv2.COLOR_RGB2GRAY)
            else:
                roi2_array = roi2_image

            thresholded_image = self.apply_threshold_filter(roi2_array)
            if thresholded_image is None:
                frame_result.error_message = "Failed to apply threshold filter"
                frame_result.processing_time = time.time() - start_time
                self.processing_times.append(frame_result.processing_time)
                return frame_result

            # Step 3: Perform connected component analysis
            components, num_labels = self.analyze_connected_components(thresholded_image)
            if components is None or num_labels <= 1:  # Only background (label 0)
                frame_result.detection_successful = False
                frame_result.error_message = "No connected components detected"

                # Handle detection failure (recovery logic)
                self.handle_detection_failure(self.frame_index)
            else:
                # Step 4: Get ROI center point for component selection
                roi_center = self.get_roi_center()
                if roi_center is None:
                    # Use image center as fallback
                    height, width = roi2_array.shape[:2]
                    roi_center = (width // 2, height // 2)

                # Adjust center point to be relative to the thresholded image coordinates
                # Since components is already a mask of the detected areas, we can use it directly
                center_component_mask = components  # This is already the filtered mask of detected components

                if center_component_mask is not None:
                    # Step 5: Calculate mask centroid for tracking
                    centroid = self.calculate_mask_centroid(center_component_mask)

                    if centroid:
                        # Step 6: Update ROI coordinates to follow centroid
                        self.update_roi_with_centroid(centroid)
                        frame_result.detection_successful = True

                        # Save the center component mask
                        if self.save_mask_image(center_component_mask, self.frame_index):
                            frame_result.mask_saved = True
                        else:
                            self.logger.warning("Failed to save center component mask")
                    else:
                        # Centroid calculation failed
                        self.handle_detection_failure(self.frame_index)
                        frame_result.error_message = "Failed to calculate mask centroid"
                else:
                    # No center component found
                    self.handle_detection_failure(self.frame_index)
                    frame_result.error_message = "No center component found"

            # Step 7: Save ROI2 image if enabled
            if self.save_roi2_image(roi2_image, self.frame_index):
                frame_result.roi2_saved = True
            else:
                self.logger.warning("Failed to save ROI2 image")

            # Calculate total processing time
            frame_result.processing_time = time.time() - start_time
            self.processing_times.append(frame_result.processing_time)

            # Log frame processing results
            if frame_result.detection_successful:
                self.logger.info(
                    f"Frame {self.frame_index}: SUCCESS - "
                    f"Processing: {frame_result.processing_time*1000:.1f}ms, "
                    f"ROI: {frame_result.roi_coordinates}, "
                    f"Files saved: ROI2={frame_result.roi2_saved}, Mask={frame_result.mask_saved}"
                )
            else:
                self.logger.warning(
                    f"Frame {self.frame_index}: NO DETECTION - "
                    f"Processing: {frame_result.processing_time*1000:.1f}ms, "
                    f"Error: {frame_result.error_message}"
                )

            return frame_result

        except Exception as e:
            # Handle unexpected errors during frame processing
            self.log_error(f"Frame {self.frame_index} processing failed", e)

            # Return failed frame result
            frame_result = FrameResult(
                frame_index=self.frame_index,
                processing_time=time.time() - start_time,
                detection_successful=False,
                roi_coordinates=self.roi_state.current_coords if self.roi_state else None,
                mask_saved=False,
                roi2_saved=False,
                error_message=f"Unexpected error: {str(e)}"
            )

            self.processing_times.append(frame_result.processing_time)
            return frame_result

    def run_detection(self) -> None:
        """Main detection loop with configurable frame rate and user interruption handling."""
        try:
            self.running = True
            self.frame_index = 0

            # Initialize ROI state if not already done
            if not self.roi_state and self.config_obj:
                self.roi_state = ROIState(
                    current_coords=(self.config_obj.roi_x1, self.config_obj.roi_y1,
                                  self.config_obj.roi_x2, self.config_obj.roi_y2),
                    previous_coords=(self.config_obj.roi_x1, self.config_obj.roi_y1,
                                   self.config_obj.roi_x2, self.config_obj.roi_y2),
                    fallback_coords=(self.config_obj.roi_x1, self.config_obj.roi_y1,
                                   self.config_obj.roi_x2, self.config_obj.roi_y2),
                    tracking_active=False,
                    consecutive_failures=0
                )

            # Setup output directories
            if not self.setup_output_directories():
                self.logger.error("Failed to setup output directories, aborting")
                return

            # Cleanup orphaned files from previous sessions
            self.cleanup_orphaned_files()

            # Log startup information
            self.logger.info("=== Vein Detection Following System Started ===")
            if self.config_obj:
                self.logger.info(f"Target frame rate: {self.config_obj.frame_rate} FPS")
                self.logger.info(f"Initial ROI coordinates: {self.roi_state.current_coords}")
                self.logger.info(f"Detection threshold: {self.config_obj.threshold_min}-{self.config_obj.threshold_max}")
                self.logger.info(f"Output directory: {os.path.join(BASE_DIR, 'vein_detection_output')}")

            print("\nVein detection started. Press Ctrl+C to stop.")
            print(f"Processing at {self.config_obj.frame_rate} FPS...")
            print(f"ROI coordinates: {self.roi_state.current_coords}")

            # Main processing loop
            consecutive_failures = 0
            max_consecutive_failures = 50  # Emergency stop after 50 consecutive failures

            while self.running:
                try:
                    # Process single frame
                    frame_result = self.process_single_frame()

                    # Increment frame counter
                    self.frame_index += 1

                    # Reset consecutive failures on success
                    if frame_result.detection_successful:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    # Emergency stop after too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(f"Emergency stop: {consecutive_failures} consecutive frame failures")
                        break

                    # Memory monitoring (check every 10 frames)
                    if self.frame_index % 10 == 0:
                        if not self.monitor_memory_usage():
                            self.logger.warning("High memory usage detected")

                    # Performance monitoring (check every 30 frames)
                    if self.frame_index % 30 == 0:
                        if not self.validate_frame_rate_performance():
                            stats = self.get_frame_rate_statistics()
                            self.logger.warning(
                                f"Performance issue detected: {stats.get('current_fps', 0):.1f} FPS"
                            )

                    # Periodic status logging (every 60 frames)
                    if self.frame_index % 60 == 0:
                        stats = self.get_frame_rate_statistics()
                        self.logger.info(
                            f"Status update - Frame {self.frame_index}: "
                            f"Current FPS: {stats.get('current_fps', 0):.1f}, "
                            f"Avg processing: {stats.get('average_processing_time', 0)*1000:.1f}ms"
                        )

                    # Frame rate control
                    self.control_frame_rate()

                except KeyboardInterrupt:
                    print("\n\nUser requested stop. Shutting down gracefully...")
                    self.running = False
                    break

                except Exception as e:
                    # Handle unexpected errors in the main loop
                    consecutive_failures += 1
                    self.log_error(f"Main loop error (frame {self.frame_index})", e)

                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(f"Emergency stop: {consecutive_failures} consecutive loop failures")
                        break

                    # Continue to next frame after logging
                    continue

            # Log shutdown statistics
            self.log_shutdown_statistics()

        except Exception as e:
            self.log_error("Fatal error in main detection loop", e)
            raise

        finally:
            self.running = False

    def log_shutdown_statistics(self) -> None:
        """Log comprehensive shutdown statistics."""
        try:
            self.logger.info("=== Vein Detection Following System Stopped ===")

            # Basic statistics
            self.logger.info(f"Total frames processed: {self.frame_index}")

            if self.roi_state:
                self.logger.info(f"Final ROI coordinates: {self.roi_state.current_coords}")
                self.logger.info(f"Final tracking status: {'Active' if self.roi_state.tracking_active else 'Inactive'}")

            # Performance statistics
            stats = self.get_frame_rate_statistics()
            if "error" not in stats and stats.get("frame_count", 0) > 0:
                self.logger.info(f"Performance Statistics:")
                self.logger.info(f"  - Average FPS: {stats.get('current_fps', 0):.2f}")
                self.logger.info(f"  - Average processing time: {stats.get('average_processing_time', 0)*1000:.1f}ms")
                self.logger.info(f"  - Max processing time: {stats.get('max_processing_time', 0)*1000:.1f}ms")
                self.logger.info(f"  - Min processing time: {stats.get('min_processing_time', 0)*1000:.1f}ms")

            # Resource usage
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.info(f"Final memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
            except ImportError:
                self.logger.info("Memory usage tracking not available (psutil not installed)")

            self.logger.info("=== System shutdown complete ===")

        except Exception as e:
            self.logger.error(f"Failed to log shutdown statistics: {e}")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Automatic Vein Detection Following System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_vein_detector.py
  python auto_vein_detector.py --config custom_config.json
  python auto_vein_detector.py --help

Configuration:
  The system uses vein_detection_config.json for configuration.
  If the file doesn't exist, a default one will be created.
        """
    )
    parser.add_argument(
        "--config",
        default="vein_detection_config.json",
        help="Path to configuration file (default: vein_detection_config.json)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Vein Detection Following System 1.0"
    )

    args = parser.parse_args()

    print("Vein Detection Following System")
    print("=" * 40)
    print(f"Configuration file: {args.config}")

    # Initialize and run the vein detection engine
    engine = VeinDetectionEngine(args.config)

    if not engine.load_configuration():
        print("\nConfiguration loading failed. Please check the error messages above.")
        print("Fix the configuration issues and try again.")
        return 1

    try:
        engine.setup_logging()
        print("System initialized successfully. Starting detection...")
        engine.run_detection()
        return 0
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
        engine.shutdown_logging()
        return 0
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if engine.logger:
            engine.log_error("Unexpected error in main execution", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())