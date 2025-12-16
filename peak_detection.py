"""
Peak detection utilities for SimpleFEM.

Design goals:
- Keep public API compatible with previous versions.
- Provide a clear, deterministic definition of green / red peaks:
  * GREEN: average gray value of the 5 frames after the peak region
           minus the average of the 5 frames before the peak region
           is >= differenceThreshold (X).
  * RED  : all other detected peaks.
  * No white state is used any more.
"""

from typing import List, Tuple
import statistics

# Optional external "improved" implementation
try:
    from improved_peak_detection import detect_peaks_improved  # type: ignore

    IMPROVED_DETECTION_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    IMPROVED_DETECTION_AVAILABLE = False


# Global flag to control whether the improved implementation should be used
USE_IMPROVED_DETECTION: bool = False


# ---------------------------------------------------------------------------
#  Basic helpers
# ---------------------------------------------------------------------------

def calculate_frame_difference(
    curve: List[float],
    peak_start: int,
    peak_end: int,
    avgFrames: int = 5,
) -> float:
    """
    Compute the average gray difference before/after a peak.

    Definition:
        frameDifference = average(curve[peak_end+1 : peak_end+6])
                          - average(curve[peak_start-5 : peak_start])

    The slices are clipped to the valid range; if there are not enough
    frames on one side, the corresponding peak edge value is used.
    """
    n = len(curve)
    if n == 0:
        return 0.0

    frame_count = int(avgFrames)
    if frame_count <= 0:
        frame_count = 5

    # Before-peak window
    before_start = max(0, peak_start - frame_count)
    before_end = max(0, peak_start - 1)
    if before_start <= before_end:
        before_vals = curve[before_start : before_end + 1]
        before_avg = sum(before_vals) / len(before_vals)
    else:
        before_avg = curve[peak_start]

    # After-peak window
    after_start = min(n - 1, peak_end + 1)
    after_end = min(n - 1, peak_end + frame_count)
    if after_start <= after_end:
        after_vals = curve[after_start : after_end + 1]
        after_avg = sum(after_vals) / len(after_vals)
    else:
        after_avg = curve[peak_end]

    return float(after_avg - before_avg)


def classify_peak_color(
    frameDifference: float,
    differenceThreshold: float = 0.5,
) -> str:
    """
    Classify peak color using the agreed rule:

    GREEN:
        (avg gray of 5 frames after peak) -
        (avg gray of 5 frames before peak) >= differenceThreshold

    RED:
        All other peaks.
    """
    return "green" if frameDifference >= differenceThreshold else "red"


# ---------------------------------------------------------------------------
#  Threshold-based peak detection (original version)
# ---------------------------------------------------------------------------

def detect_white_peaks_by_threshold(
    curve: List[float],
    threshold: float = 105.0,
    marginFrames: int = 5,
    differenceThreshold: float = 0.5,
    avgFrames: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Simple absolute-threshold peak detection.

    This is the "original" version used mainly for comparison/testing.
    It groups consecutive samples >= threshold into a peak region.

    Returns a list of (start, end, frameDifference).
    """
    n = len(curve)
    if n == 0:
        return []

    peaks: List[Tuple[int, int]] = []
    in_peak = False
    start = 0

    for i, v in enumerate(curve):
        if v >= threshold:
            if not in_peak:
                start = i
                in_peak = True
        else:
            if in_peak:
                peaks.append((start, i - 1))
                in_peak = False

    if in_peak:
        peaks.append((start, n - 1))

    result: List[Tuple[int, int, float]] = []
    for s, e in peaks:
        frame_diff = calculate_frame_difference(curve, s, e, avgFrames=avgFrames)
        result.append((s, e, frame_diff))

    return result


# ---------------------------------------------------------------------------
#  Threshold-based peak detection with proper spacing control
# ---------------------------------------------------------------------------

def detect_white_peaks_by_threshold_improved(
    curve: List[float],
    threshold: float = 105.0,
    marginFrames: int = 5,
    silenceFrames: int = 0,
    differenceThreshold: float = 0.5,
    avgFrames: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Improved absolute-threshold peak detection.

    Differences from the original:
    - Enforces a minimum spacing of `marginFrames` between peaks:
      if two peaks are closer than this, only the higher one is kept.
    - Optionally enforces a "silence" constraint around each peak when
      `silenceFrames > 0`:
        * In the `silenceFrames` samples immediately BEFORE the rising
          edge of a peak, all values must be < threshold.
        * In the `silenceFrames` samples immediately AFTER the falling
          edge of a peak, all values must be < threshold.
      Peaks too close to the boundaries (无法满足前后 X 帧) 将被丢弃。
    - Still returns (start, end, frameDifference).
    """
    n = len(curve)
    if n == 0:
        return []

    # First, build raw contiguous >= threshold segments.
    raw: List[Tuple[int, int]] = []
    in_peak = False
    start = 0

    for i, v in enumerate(curve):
        if v >= threshold:
            if not in_peak:
                start = i
                in_peak = True
        else:
            if in_peak:
                raw.append((start, i - 1))
                in_peak = False

    if in_peak:
        raw.append((start, n - 1))

    if not raw:
        return []

    # Enforce minimal spacing between peaks.
    if marginFrames > 0 and len(raw) > 1:
        filtered: List[Tuple[int, int]] = [raw[0]]
        for s, e in raw[1:]:
            last_s, last_e = filtered[-1]
            spacing = s - last_e
            if spacing >= marginFrames:
                filtered.append((s, e))
            else:
                # keep the region with higher maximum value
                last_max = max(curve[last_s : last_e + 1])
                cur_max = max(curve[s : e + 1])
                if cur_max > last_max:
                    filtered[-1] = (s, e)
        raw = filtered

    # Enforce pre/post "silence" around each peak:
    # - before the rising edge: previous `silenceFrames` samples < threshold
    # - after the falling edge: next `silenceFrames` samples < threshold
    if silenceFrames > 0 and raw:
        silenced: List[Tuple[int, int]] = []
        for s, e in raw:
            # Not enough margin at sequence boundaries -> discard this peak
            if s - silenceFrames < 0 or e + silenceFrames >= n:
                continue

            pre_ok = all(curve[i] < threshold for i in range(s - silenceFrames, s))
            post_ok = all(
                curve[i] < threshold for i in range(e + 1, e + 1 + silenceFrames)
            )

            if pre_ok and post_ok:
                silenced.append((s, e))

        raw = silenced

    if not raw:
        return []

    # Final frame difference computation.
    result: List[Tuple[int, int, float]] = []
    for s, e in raw:
        frame_diff = calculate_frame_difference(curve, s, e, avgFrames=avgFrames)
        result.append((s, e, frame_diff))

    return result


# ---------------------------------------------------------------------------
#  Morphological / width-based detection (currently unused in daemon)
# ---------------------------------------------------------------------------

def detect_white_curve_peaks(
    curve: List[float],
    sensitivity: float = 20.0,
    minPeakWidth: int = 3,
    maxPeakWidth: int = 15,
    minDistance: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Width-constrained peak detection.

    This implementation is intentionally simple and is not used by the
    main daemon at the moment. It is kept for API compatibility.
    """
    n = len(curve)
    if n < minPeakWidth * 2:
        return []

    baseline = statistics.median(curve)
    candidates: List[Tuple[int, int, float]] = []

    for i in range(1, n - 1):
        # strict local maximum
        if not (curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1]):
            continue

        height = curve[i] - baseline
        if height < sensitivity:
            continue

        # expand left until monotonic increase breaks
        l = i
        while l > 0 and curve[l] >= curve[l - 1]:
            l -= 1

        # expand right until monotonic decrease breaks
        r = i
        while r < n - 1 and curve[r] >= curve[r + 1]:
            r += 1

        width = r - l + 1
        if width < minPeakWidth or width > maxPeakWidth:
            continue

        frame_diff = calculate_frame_difference(curve, l, r, avgFrames=5)
        candidates.append((l, r, frame_diff))

    # simple distance-based deduplication
    if not candidates:
        return []

    candidates.sort(key=lambda p: p[0])
    filtered: List[Tuple[int, int, float]] = [candidates[0]]
    for s, e, fd in candidates[1:]:
        last_s, last_e, _ = filtered[-1]
        if s - last_e < minDistance:
            # too close, keep the higher peak
            last_max = max(curve[last_s : last_e + 1])
            cur_max = max(curve[s : e + 1])
            if cur_max > last_max:
                filtered[-1] = (s, e, fd)
        else:
            filtered.append((s, e, fd))

    return filtered


# ---------------------------------------------------------------------------
#  Improved color classification wrapper
# ---------------------------------------------------------------------------

def classify_peak_color_improved(
    frameDifference: float,
    differenceThreshold: float,
    peak_start: int,
    peak_end: int,
    curve: List[float],
) -> str:
    """
    Improved peak color classification.

    For now, the rule is intentionally aligned with `classify_peak_color`:
    - GREEN: frameDifference >= differenceThreshold
    - RED  : otherwise

    Extra arguments are accepted to keep backward compatibility.
    """
    return classify_peak_color(frameDifference, differenceThreshold)


# ---------------------------------------------------------------------------
#  Peak scoring (used only in tests/experiments at the moment)
# ---------------------------------------------------------------------------

def evaluate_peak_score(
    curve: List[float],
    start: int,
    end: int,
    frame_diff: float,
    differenceThreshold: float = 2.1,
) -> float:
    """
    Compute a simple quality score for a peak.

    Higher scores mean "better" peaks.
    """
    if start < 0 or end <= start or end >= len(curve):
        return 0.0

    segment = curve[start : end + 1]
    peak_max = max(segment)
    peak_avg = sum(segment) / len(segment)
    peak_width = end - start + 1

    score = 0.0

    # 1. Height contribution
    score += peak_max * 0.4

    # 2. Color contribution (green preferred over red)
    color = classify_peak_color(frame_diff, differenceThreshold)
    if color == "green":
        score += 50.0
    else:
        score -= 30.0

    # 3. Compactness (narrower peaks get higher score up to a limit)
    score += max(0.0, 20.0 - float(peak_width))

    # 4. Prominence: relative distance between max and mean
    if peak_avg > 0:
        score += (peak_max - peak_avg) / peak_avg * 10.0

    return score


# ---------------------------------------------------------------------------
#  Main API: detect_peaks
# ---------------------------------------------------------------------------

def detect_peaks(
    curve: List[float],
    threshold: float = 105.0,
    marginFrames: int = 5,
    differenceThreshold: float = 0.5,
    silenceFrames: int = 0,
    avgFrames: int = 5,
    use_improved: bool = False,
    **config_params,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Detect peaks and classify them into green (stable) and red (unstable).

    - When an external improved implementation is available and enabled,
      this function delegates to `detect_peaks_improved`.
    - Otherwise it uses the internal improved threshold-based detector
      plus the agreed green/red rule based on frameDifference.
    """
    if not curve:
        return [], []

    # Decide which implementation to use.
    use_improved_algo = use_improved or USE_IMPROVED_DETECTION
    if use_improved_algo and IMPROVED_DETECTION_AVAILABLE:
        # Delegate to external implementation; assumed to respect the
        # same classification semantics.
        # Pass silenceFrames via config_params so external impl can choose
        # to honour it or ignore it.
        if "silenceFrames" not in config_params:
            config_params["silenceFrames"] = silenceFrames
        if "avgFrames" not in config_params:
            config_params["avgFrames"] = avgFrames
        return detect_peaks_improved(
            curve,
            threshold,
            marginFrames,
            differenceThreshold,
            **config_params,
        )

    # Internal improved threshold-based detection.
    peaks_with_diff = detect_white_peaks_by_threshold_improved(
        curve,
        threshold=threshold,
        marginFrames=marginFrames,
        silenceFrames=silenceFrames,
        differenceThreshold=differenceThreshold,
        avgFrames=avgFrames,
    )

    green_peaks: List[Tuple[int, int]] = []
    red_peaks: List[Tuple[int, int]] = []

    for start, end, frame_diff in peaks_with_diff:
        color = classify_peak_color(frame_diff, differenceThreshold)
        if color == "green":
            green_peaks.append((start, end))
        else:  # "red"
            red_peaks.append((start, end))

    return green_peaks, red_peaks


# ---------------------------------------------------------------------------
#  Global toggles and compatibility wrappers
# ---------------------------------------------------------------------------

def enable_improved_detection(enable: bool = True) -> bool:
    """
    Enable or disable the external improved peak detection algorithm.

    Returns True if the requested state is active, False otherwise.
    """
    global USE_IMPROVED_DETECTION

    if enable and not IMPROVED_DETECTION_AVAILABLE:
        print("Warning: improved_peak_detection.detect_peaks_improved not available")
        USE_IMPROVED_DETECTION = False
        return False

    USE_IMPROVED_DETECTION = bool(enable)
    status = "enabled" if USE_IMPROVED_DETECTION else "disabled"
    print(f"Improved peak detection is now {status}")
    return True


def is_improved_detection_enabled() -> bool:
    """Return True if the improved detection pipeline is active."""
    return USE_IMPROVED_DETECTION and IMPROVED_DETECTION_AVAILABLE


def detect_green_peaks(
    curve: List[float],
    threshold: float = 105.0,
    marginFrames: int = 5,
    differenceThreshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Backwards-compatible helper: return only green peak intervals.
    """
    greens, _ = detect_peaks(
        curve,
        threshold=threshold,
        marginFrames=marginFrames,
        differenceThreshold=differenceThreshold,
    )
    return greens


if __name__ == "__main__":  # pragma: no cover - manual quick test
    test_curve = [
        40,
        42,
        45,
        48,
        52,
        108,
        110,
        112,
        109,
        107,
        45,
        43,
        41,
        42,
        44,
        46,
        49,
        53,
        55,
        58,
        60,
        62,
        61,
        59,
        45,
        43,
        41,
        42,
        45,
        110,
        115,
        118,
        116,
        113,
        48,
        46,
        44,
        42,
        41,
    ]

    g, r = detect_peaks(test_curve, threshold=100.0, marginFrames=5, differenceThreshold=0.5)
    print("Green peaks:", g)
    print("Red peaks  :", r)
