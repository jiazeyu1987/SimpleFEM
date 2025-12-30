from __future__ import annotations

from typing import Any, Dict


class AntiJitterManager:
    """
    Build ROI2 intersection anti-jitter filter instance from config.

    Note: Decision logic/prints must remain consistent with the legacy daemon.
    """

    def build(self, config: Dict[str, Any]) -> Any:
        anti_jitter_config = config.get("roi2_anti_jitter", {})
        intersection_filter = None

        if anti_jitter_config.get("enabled", False):
            # 参数验证和标准化
            try:
                algorithm = anti_jitter_config.get("algorithm", "ema")
                movement_threshold = float(anti_jitter_config.get("movement_threshold", 20.0))
                initialization_frames = int(anti_jitter_config.get("initialization_frames", 3))

                if algorithm == "threshold":
                    # 阈值式防抖动
                    from .external.threshold_based_anti_jitter import ThresholdIntersectionFilter

                    intersection_filter = ThresholdIntersectionFilter(movement_threshold, initialization_frames)
                    print("ROI2阈值式防抖动已启用:")
                    print("  - algorithm: threshold (阈值式)")
                    print(f"  - movement_threshold: {movement_threshold}px (小于此值ROI2完全不动)")
                    print(f"  - initialization_frames: {initialization_frames} (前N帧初始化稳定位置)")
                    print(f"  - 策略: 小于{movement_threshold}px变化时ROI2完全静止，超过才更新")
                else:
                    # EMA平滑式防抖动
                    from .external.green_detector import IntersectionFilter

                    ema_config = anti_jitter_config.get("ema", {})
                    alpha = float(ema_config.get("alpha", 0.25))
                    stability_threshold = float(anti_jitter_config.get("stability_threshold", 8.0))

                    # 参数范围验证
                    if not (0.05 <= alpha <= 0.95):
                        print(f"Warning: alpha={alpha} 超出推荐范围[0.05, 0.95]，将自动调整")
                    if movement_threshold < 1.0:
                        print(f"Warning: movement_threshold={movement_threshold} 过小，建议设置为1.0以上")
                    if stability_threshold < 1.0:
                        print(f"Warning: stability_threshold={stability_threshold} 过小，建议设置为1.0以上")
                    if not (stability_threshold < movement_threshold):
                        print(
                            f"Warning: stability_threshold({stability_threshold}) 应该小于 movement_threshold({movement_threshold})"
                        )
                    if initialization_frames < 1 or initialization_frames > 20:
                        print(f"Warning: initialization_frames={initialization_frames} 可能不合适，推荐范围[1, 20]")

                    intersection_filter = IntersectionFilter(
                        alpha, movement_threshold, initialization_frames, stability_threshold
                    )
                    print("ROI2平滑式防抖动已启用:")
                    print("  - algorithm: ema (指数移动平均平滑)")
                    print(f"  - alpha (平滑因子): {alpha} (值越小越平滑)")
                    print(f"  - movement_threshold (运动阈值): {movement_threshold}px (大于此值直接通过)")
                    print(f"  - stability_threshold (稳定阈值): {stability_threshold}px (小于此值强力平滑)")
                    print(f"  - initialization_frames (初始化帧数): {initialization_frames}")

            except (ValueError, TypeError) as e:
                from .external.green_detector import IntersectionFilter

                print(f"Error: 防抖动配置参数无效: {e}")
                print("使用默认参数启用EMA防抖动")
                intersection_filter = IntersectionFilter()  # 使用默认参数
        else:
            print("ROI2防抖动已禁用")

        return anti_jitter_config, intersection_filter
