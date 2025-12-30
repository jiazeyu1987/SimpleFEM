from __future__ import annotations

from typing import Any, Optional

from fem_refactor.models import DaemonContext


def shutdown_daemon(
    *,
    analysis_cache: Optional[Any],
    ctx: Optional[DaemonContext],
    intersection_filter: Any,
) -> None:
    if analysis_cache is not None:
        try:
            analysis_cache.close(reason="shutdown")
        except Exception:
            pass

    # 释放视频资源
    if ctx is not None and ctx.video.video_cap is not None:
        ctx.video.video_cap.release()
        print("视频资源已释放")

    # 输出防抖动滤波器最终统计信息
    if intersection_filter:
        try:
            debug_info = intersection_filter.get_debug_info()
            print(f"\n防抖动滤波器最终统计:")
            print(f"  总处理帧数: {debug_info['frame_count']}")

            # 根据滤波器类型显示不同信息
            if "update_count" in debug_info:
                # 阈值式滤波器
                print(f"  更新次数: {debug_info['update_count']}")
                print(f"  忽略次数: {debug_info['ignore_count']}")
                print(f"  稳定率: {debug_info.get('stability_rate', 0):.1f}%")
                print(f"  大运动事件: {debug_info['large_movement_count']}次")
                print(f"  阈值参数: threshold={debug_info['parameters']['movement_threshold']}px")
            else:
                # EMA滤波器
                print(f"  大运动事件: {debug_info['large_movement_count']}次")
                print(f"  边界限制事件: {debug_info['boundary_clamp_count']}次")
                print(f"  稳定事件: {debug_info['stability_count']}次")
                print(
                    f"  EMA参数: alpha={debug_info['parameters']['alpha']}, "
                    f"threshold={debug_info['parameters']['movement_threshold']}px"
                )
        except Exception as e:
            print(f"获取防抖动统计信息失败: {e}")

