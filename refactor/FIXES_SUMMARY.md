# SimpleFEM Refactored Code Fixes Summary

**Date**: 2025-12-29
**Session**: Continuation of ROI1 waveform and peak detection alignment

## Problem Statement

The refactored SimpleFEM code was producing different peak detection results compared to the original `simple_roi_daemon.py`:
- Different peak frame indices
- Missing G1/G2 and column_diff values in CSV exports
- Inconsistent ROI1 waveform generation

## Fixes Applied

### 1. Added ROI3 G1/G2 Buffers (roi_capture_manager.py)

**Problem**: CSV fields for G1/G2 values were empty
**Root Cause**: Refactored code only had `roi3_80_160_buffer`, missing G1/G2 buffers

**Solution**:
- Added `_roi3_g1_buffer` and `_roi3_g2_buffer` deques (maxlen=100)
- Added property methods for access
- Updated `reset_buffers()` to clear new buffers

**Files Modified**:
- `refactor/roi_capture_manager.py` (lines 58-59, 385-393, 432-433)

### 2. Populated G1/G2 Buffers in Main Loop (orchestrator.py)

**Problem**: G1/G2 buffers were created but never populated

**Solution**:
- Calculate G1/G2 from `ROI3Statistics.compute_all()` result
- Append to buffers after ROI3 calculation (lines 316-318)

**Files Modified**:
- `refactor/orchestrator.py` (lines 316-318)

### 3. Implemented Complete G1/G2 Override Logic (hybrid_detection_manager.py)

**Problem**: Peak detection was missing G1/G2 override mechanism

**Solution**:
- Completely rewrote `_determine_roi2_color()` method
- Added G1/G2 curve parameters to method signature
- Implemented G1/G2 override logic:
  - Extract G1/G2 values from peak interval
  - Find maximum G1 position (brightest frame)
  - If G1 >= 98% and G2 >= 20%, force red → green
- Implemented column diff override logic
- Return complete result dictionary with all required fields

**Files Modified**:
- `refactor/hybrid_detection_manager.py` (complete rewrite, 348 lines)

### 4. Fixed Configuration Reading Path (hybrid_detection_manager.py)

**Problem**: G1/G2 override always returned False (default value)

**Root Cause**: Wrong configuration path - used `config.get('g1_g2_override', ...)` instead of `config.get('peak_detection', 'g1_g2_override', ...)`

**Solution**:
- Corrected configuration path to `peak_detection.g1_g2_override.*`
- Verified values now read correctly:
  - `enabled`: True
  - `g1_threshold`: 98.0%
  - `g2_threshold`: 20.0%

**Files Modified**:
- `refactor/hybrid_detection_manager.py` (lines 169-173)

### 5. Made ROI1 Buffer Population Conditional (orchestrator.py)

**Problem**: Peak frame indices differed between original and refactored code

**Root Cause**: Refactored code unconditionally populated ROI1 buffer, while original code only does so when `roi1_enabled` is True

**Solution**:
- Check `roi1_peak_detection_enabled` before calculating ROI1 gray value
- Only populate ROI1 buffer when enabled
- Handle None case for `roi1_gray` in cache payload and logging

**Files Modified**:
- `refactor/orchestrator.py` (lines 281-286, 402, 482-495)

## Configuration Structure

The refactored code now correctly reads the nested configuration:

```json
{
  "peak_detection": {
    "g1_g2_override": {
      "enabled": true,
      "g1_threshold": 98.0,
      "g2_threshold": 20.0,
      "use_peak_max": true
    },
    "roi3_column_diff_override": {
      "enabled": true,
      "threshold": 15.0
    }
  },
  "roi1_peak_detection": {
    "enabled": true,
    "threshold": 63.0
  },
  "hybrid_detection": {
    "enabled": true,
    "detection_strategy": "roi1_peaks_roi2_color"
  }
}
```

## Verification

Configuration reading verification:
```python
ROI1 Detection Configuration:
  Enabled: True
  Threshold: 63.0

Hybrid Detection Configuration:
  Enabled: True

G1/G2 Override Configuration:
  Enabled: True
  G1 Threshold: 98.0%
  G2 Threshold: 20.0%

Column Diff Override Configuration:
  Enabled: True
  Threshold: 15.0
```

## Expected Behavior After Fixes

1. **ROI1 Buffer**: Only populated when `roi1_peak_detection.enabled = true`
2. **G1/G2 Fields**: CSV should now show G1 and G2 values (typically ~100% for G1, 0.7-1.2% for G2)
3. **G1/G2 Override**: When both thresholds met, red peaks forced to green
4. **Column Diff Override**: When column diff >= 15.0, red peaks forced to green
5. **Peak Frame Indices**: Should now match original code results

## Remaining Work

1. **Implement Column Diff Buffer**: Currently set to None in hybrid detection call (line 354)
2. **Run Full Video Processing Test**: Verify all peaks match between original and refactored code
3. **Compare CSV Outputs**: Ensure all fields (including G1/G2/column_diff) are populated correctly

## Related Files

**Original Code**:
- `simple_roi_daemon.py` - Main daemon with all original logic
- `simple_fem_config.json` - Configuration file

**Refactored Code**:
- `refactor/orchestrator.py` - Main processing loop
- `refactor/roi_capture_manager.py` - ROI capture and buffer management
- `refactor/hybrid_detection_manager.py` - Hybrid detection with G1/G2 override
- `refactor/roi3_statistics.py` - ROI3 statistics calculation (G1/G2, column diff)
- `refactor/config_manager.py` - Configuration management with nested path support

## Test Cases

**Video 2 Comparison**:
- Original peaks: [93, 349, 408, 498, 979]
- Refactored peaks (before fix): [175, 492, 582, 979, 1062]
- Expected (after fix): Should match original

**CSV Field Verification**:
- Original has complete G1/G2 values (~100%, 0.7-1.2%)
- Original has column diff values (10-15)
- Refactored should now show same values
