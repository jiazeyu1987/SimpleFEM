"""
诊断frame_index差异问题

分析为什么新CSV中frame_index=175但post_peak_frame_end=92
"""

import pandas as pd
import json

# 读取CSV文件
new_csv_path = r"D:\ProjectPackage\SimpleFEM\export\peak_statistics_2（5次发射，1无效)_20251229_095944.csv"
old_csv_path = r"D:\ProjectPackage\SimpleFEM\export_2\peak_statistics_2（5次发射，1无效)_20251228_163652.csv"

print("=" * 80)
print("OLD CSV (Original Code)")
print("=" * 80)
old_df = pd.read_csv(old_csv_path)
print(f"Total peaks: {len(old_df)}")
print("\nFirst peak:")
first_old = old_df.iloc[0]
for col in ['peak_type', 'frame_index', 'post_peak_frame_start', 'post_peak_frame_end',
            'pre_peak_frame_start', 'pre_peak_frame_end']:
    print(f"  {col}: {first_old[col]}")

print(f"\nBuffer calculation:")
frame_index_old = int(first_old['frame_index'])
post_end_old = int(first_old['post_peak_frame_end'])
# curve_start_global_frame = frame_index - len(curve) + 1
# post_peak_frame_end = curve_start_global_frame + post_end (buffer position)
# So: len(curve) = frame_index - post_peak_frame_end + post_end + 1
# But we don't know post_end (buffer position). Let's estimate.

# If peak ended at global frame X and post_end is about 5 frames after peak:
# Then the buffer length can be estimated
estimated_buffer_len_old = frame_index_old - post_end_old + 5
print(f"  frame_index: {frame_index_old}")
print(f"  post_peak_frame_end: {post_end_old}")
print(f"  Estimated buffer length: ~{estimated_buffer_len_old} frames")
print(f"  Buffer start frame: {frame_index_old - estimated_buffer_len_old + 1}")

print("\n" + "=" * 80)
print("NEW CSV (Refactored Code)")
print("=" * 80)
new_df = pd.read_csv(new_csv_path)
print(f"Total peaks: {len(new_df)}")
print("\nFirst peak:")
first_new = new_df.iloc[0]
for col in ['peak_type', 'frame_index', 'post_peak_frame_start', 'post_peak_frame_end',
            'pre_peak_frame_start', 'pre_peak_frame_end']:
    print(f"  {col}: {first_new[col]}")

print(f"\nBuffer calculation:")
frame_index_new = int(first_new['frame_index'])
post_end_new = int(first_new['post_peak_frame_end'])
estimated_buffer_len_new = frame_index_new - post_end_new + 5
print(f"  frame_index: {frame_index_new}")
print(f"  post_peak_frame_end: {post_end_new}")
print(f"  Estimated buffer length: ~{estimated_buffer_len_new} frames")
print(f"  Buffer start frame: {frame_index_new - estimated_buffer_len_new + 1}")

print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)
print(f"Old code: CSV written at frame {frame_index_old}, post_peak_end={post_end_old}")
print(f"         Difference: {frame_index_old - post_end_old} frames")
print(f"New code: CSV written at frame {frame_index_new}, post_peak_end={post_end_new}")
print(f"         Difference: {frame_index_new - post_end_new} frames")
print(f"\nThe new code writes the CSV {frame_index_new - post_end_new - (frame_index_old - post_end_old)} frames later!")

print("\n" + "=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print("In the old code, when a peak completes at frame X, the CSV is written immediately")
print("with frame_index=X and post_peak_frame_end=X.")
print("\nIn the new code, there seems to be a delay. The peak completes at frame 92,")
print("but the CSV is written at frame 175 - that's 83 frames later!")
print("\nPossible causes:")
print("1. Peaks are aggregated and written in batches (delayed recording)")
print("2. Deduplication logic waits to confirm peaks before writing")
print("3. Buffer state at recording time is different from detection time")
print("4. Statistics recording is called at wrong point in processing loop")

print("\n" + "=" * 80)
print("CHECKING PEAK INTERVALS")
print("=" * 80)
# Calculate peak interval from pre/post frames
print("Old CSV peak interval:")
old_peak_start = int(first_old['pre_peak_frame_end']) + 1  # Peak starts after pre-avg
old_peak_end = int(first_old['post_peak_frame_start']) - 1  # Peak ends before post-avg
print(f"  Estimated: frames {old_peak_start} to {old_peak_end}")
print(f"  Width: {old_peak_end - old_peak_start + 1} frames")

print("\nNew CSV peak interval:")
new_peak_start = int(first_new['pre_peak_frame_end']) + 1
new_peak_end = int(first_new['post_peak_frame_start']) - 1
print(f"  Estimated: frames {new_peak_start} to {new_peak_end}")
print(f"  Width: {new_peak_end - new_peak_start + 1} frames")

print("\nNote: These intervals represent when the actual peak occurred,")
print("      not when it was written to CSV!")
