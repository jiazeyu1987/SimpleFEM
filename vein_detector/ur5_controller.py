#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR5 机械臂控制管理类

负责与 UR5 机械臂的通信、URScript 命令发送和状态管理。
UI 类应该只负责显示和用户交互，所有控制逻辑通过此类实现。
"""

import socket
import threading
import time
import logging
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
from enum import Enum


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoveType(Enum):
    """移动类型枚举"""
    MOVEJ = "movej"  # 关节空间移动（快速）
    MOVEL = "movel"  # 笛卡尔空间移动（直线）


class ConnectionStatus(Enum):
    """连接状态枚举"""
    DISCONNECTED = "未连接"
    CONNECTING = "连接中"
    CONNECTED = "已连接"
    ERROR = "错误"


@dataclass
class Pose:
    """位姿数据类"""
    x: float  # X 位置 (m)
    y: float  # Y 位置 (m)
    z: float  # Z 位置 (m)
    rx: float  # 绕 X 轴旋转 (rad)
    ry: float  # 绕 Y 轴旋转 (rad)
    rz: float  # 绕 Z 轴旋转 (rad)

    def to_list(self) -> List[float]:
        """转换为列表"""
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_list(cls, pose_list: List[float]) -> 'Pose':
        """从列表创建"""
        if len(pose_list) < 6:
            raise ValueError("位姿列表长度必须 >= 6")
        return cls(
            x=pose_list[0],
            y=pose_list[1],
            z=pose_list[2],
            rx=pose_list[3],
            ry=pose_list[4],
            rz=pose_list[5]
        )

    def __str__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f}, {self.rx:.4f}, {self.ry:.4f}, {self.rz:.4f})"


@dataclass
class JointAngles:
    """关节角度数据类"""
    j0: float  # 基座关节 (rad)
    j1: float  # 肩部关节 (rad)
    j2: float  # 肘部关节 (rad)
    j3: float  # 腕部1关节 (rad)
    j4: float  # 腕部2关节 (rad)
    j5: float  # 腕部3关节 (rad)

    def to_list(self) -> List[float]:
        """转换为列表"""
        return [self.j0, self.j1, self.j2, self.j3, self.j4, self.j5]

    @classmethod
    def from_list(cls, angles: List[float]) -> 'JointAngles':
        """从列表创建"""
        if len(angles) < 6:
            raise ValueError("关节角度列表长度必须 >= 6")
        return cls(
            j0=angles[0],
            j1=angles[1],
            j2=angles[2],
            j3=angles[3],
            j4=angles[4],
            j5=angles[5]
        )

    def __str__(self) -> str:
        return f"({self.j0:.4f}, {self.j1:.4f}, {self.j2:.4f}, {self.j3:.4f}, {self.j4:.4f}, {self.j5:.4f})"


class UR5Controller:
    """
    UR5 机械臂控制管理类

    功能：
    - 通过 TCP/IP 连接 UR5 机械臂
    - 发送 URScript 命令进行控制
    - 支持 MOVEJ 和 MOVEL 移动
    - 状态监控和错误处理
    """

    # 默认连接参数
    DEFAULT_HOST = "192.168.1.10"
    DEFAULT_PORT = 30002  # UR5 secondary client port
    TIMEOUT = 5.0

    def __init__(self, host: str = None, port: int = None):
        """
        初始化 UR5 控制器

        Args:
            host: UR5 机械臂 IP 地址
            port: UR5 secondary client 端口（默认 30002）
        """
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT

        # 连接状态
        self._status = ConnectionStatus.DISCONNECTED
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()

        # 当前状态
        self._current_pose: Optional[Pose] = None
        self._current_joints: Optional[JointAngles] = None

        # 回调函数
        self._status_callbacks: List[Callable[[ConnectionStatus], None]] = []
        self._pose_callbacks: List[Callable[[Pose], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []

        logger.info(f"UR5 控制器初始化: {self.host}:{self.port}")

    @property
    def status(self) -> ConnectionStatus:
        """获取连接状态"""
        return self._status

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._status == ConnectionStatus.CONNECTED

    @property
    def current_pose(self) -> Optional[Pose]:
        """获取当前位姿"""
        return self._current_pose

    @property
    def current_joints(self) -> Optional[JointAngles]:
        """获取当前关节角度"""
        return self._current_joints

    def add_status_callback(self, callback: Callable[[ConnectionStatus], None]):
        """添加状态变化回调"""
        self._status_callbacks.append(callback)

    def add_pose_callback(self, callback: Callable[[Pose], None]):
        """添加位姿更新回调"""
        self._pose_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str], None]):
        """添加错误回调"""
        self._error_callbacks.append(callback)

    def _notify_status(self, status: ConnectionStatus):
        """通知状态变化"""
        for callback in self._status_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"状态回调错误: {e}")

    def _notify_pose(self, pose: Pose):
        """通知位姿更新"""
        for callback in self._pose_callbacks:
            try:
                callback(pose)
            except Exception as e:
                logger.error(f"位姿回调错误: {e}")

    def _notify_error(self, error_msg: str):
        """通知错误"""
        for callback in self._error_callbacks:
            try:
                callback(error_msg)
            except Exception as e:
                logger.error(f"错误回调错误: {e}")

    def connect(self, host: str = None, port: int = None) -> bool:
        """
        连接到 UR5 机械臂

        Args:
            host: IP 地址（不指定则使用初始化时的值）
            port: 端口（不指定则使用初始化时的值）

        Returns:
            bool: 连接是否成功
        """
        if host:
            self.host = host
        if port:
            self.port = port

        if self.is_connected:
            logger.warning("已经连接，无需重复连接")
            return True

        try:
            self._notify_status(ConnectionStatus.CONNECTING)
            logger.info(f"正在连接到 {self.host}:{self.port}...")

            # 创建 TCP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.TIMEOUT)

            # 连接到 UR5
            self._socket.connect((self.host, self.port))

            self._status = ConnectionStatus.CONNECTED
            self._notify_status(ConnectionStatus.CONNECTED)

            logger.info("成功连接到 UR5 机械臂")

            # 启动状态监控线程
            self._start_monitor()

            return True

        except socket.timeout:
            error = f"连接超时: {self.host}:{self.port}"
            logger.error(error)
            self._status = ConnectionStatus.ERROR
            self._notify_status(ConnectionStatus.ERROR)
            self._notify_error(error)
            return False

        except ConnectionRefusedError:
            error = f"连接被拒绝: {self.host}:{self.port}"
            logger.error(error)
            self._status = ConnectionStatus.ERROR
            self._notify_status(ConnectionStatus.ERROR)
            self._notify_error(error)
            return False

        except Exception as e:
            error = f"连接失败: {e}"
            logger.error(error)
            self._status = ConnectionStatus.ERROR
            self._notify_status(ConnectionStatus.ERROR)
            self._notify_error(error)
            return False

    def disconnect(self):
        """断开连接"""
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.error(f"关闭 socket 时出错: {e}")
            finally:
                self._socket = None

        self._status = ConnectionStatus.DISCONNECTED
        self._notify_status(ConnectionStatus.DISCONNECTED)
        logger.info("已断开连接")

    def _send_command(self, command: str) -> bool:
        """
        发送 URScript 命令到 UR5

        Args:
            command: URScript 命令字符串

        Returns:
            bool: 发送是否成功
        """
        if not self.is_connected:
            error = "未连接到机械臂，无法发送命令"
            logger.error(error)
            self._notify_error(error)
            return False

        try:
            with self._lock:
                # URScript 命令以换行符结尾
                cmd = command.strip() + "\n"
                self._socket.sendall(cmd.encode('ascii'))
                logger.debug(f"发送命令: {command.strip()}")
                return True

        except Exception as e:
            error = f"发送命令失败: {e}"
            logger.error(error)
            self._notify_error(error)
            return False

    def movej(
        self,
        joints: JointAngles,
        acceleration: float = 1.0,
        velocity: float = 1.0,
        time: float = 0.0
    ) -> bool:
        """
        关节空间移动 (MOVEJ)

        Args:
            joints: 目标关节角度
            acceleration: 加速度 (0.0-2.0)
            velocity: 速度 (0.0-1.0)
            time: 移动时间 (秒)，0 表示自动计算

        Returns:
            bool: 命令是否发送成功
        """
        j = joints.to_list()
        cmd = f"movej([{j[0]}, {j[1]}, {j[2]}, {j[3]}, {j[4]}, {j[5]}], " \
              f"a={acceleration}, v={velocity}"

        if time > 0:
            cmd += f", t={time}"

        cmd += ")"

        logger.info(f"MOVEJ: 目标关节={joints}, a={acceleration}, v={velocity}")
        return self._send_command(cmd)

    def movel(
        self,
        pose: Pose,
        acceleration: float = 1.0,
        velocity: float = 1.0,
        time: float = 0.0
    ) -> bool:
        """
        笛卡尔空间移动 (MOVEL)

        Args:
            pose: 目标位姿
            acceleration: 加速度 (0.0-2.0)
            velocity: 速度 (0.0-1.0)
            time: 移动时间 (秒)，0 表示自动计算

        Returns:
            bool: 命令是否发送成功
        """
        p = pose.to_list()
        cmd = f"movel(p[{p[0]}, {p[1]}, {p[2]}, {p[3]}, {p[4]}, {p[5]}], " \
              f"a={acceleration}, v={velocity}"

        if time > 0:
            cmd += f", t={time}"

        cmd += ")"

        logger.info(f"MOVEL: 目标位姿={pose}, a={acceleration}, v={velocity}")
        return self._send_command(cmd)

    def movep(
        self,
        pose: Pose,
        acceleration: float = 1.0,
        velocity: float = 1.0,
        blend: float = 0.0
    ) -> bool:
        """
        过渡移动 (MOVEP) - 类似 MOVEL 但支持路径混合

        Args:
            pose: 目标位姿
            acceleration: 加速度 (0.0-2.0)
            velocity: 速度 (0.0-1.0)
            blend: 混合半径 (米)

        Returns:
            bool: 命令是否发送成功
        """
        p = pose.to_list()
        cmd = f"movep(p[{p[0]}, {p[1]}, {p[2]}, {p[3]}, {p[4]}, {p[5]}], " \
              f"a={acceleration}, v={velocity}, r={blend})"

        logger.info(f"MOVEP: 目标位姿={pose}, a={acceleration}, v={velocity}, r={blend}")
        return self._send_command(cmd)

    def stopj(self, deceleration: float = 2.0) -> bool:
        """
        停止关节运动

        Args:
            deceleration: 减速度 (0.0-2.0)

        Returns:
            bool: 命令是否发送成功
        """
        cmd = f"stopj({deceleration})"
        logger.info(f"STOPJ: 减速度={deceleration}")
        return self._send_command(cmd)

    def stopl(self, deceleration: float = 2.0) -> bool:
        """
        停止笛卡尔运动

        Args:
            deceleration: 减速度 (0.0-2.0)

        Returns:
            bool: 命令是否发送成功
        """
        cmd = f"stopl({deceleration})"
        logger.info(f"STOPL: 减速度={deceleration}")
        return self._send_command(cmd)

    def set_digital_out(self, pin: int, value: bool) -> bool:
        """
        设置数字输出

        Args:
            pin: 引脚编号 (0-7)
            value: True/False

        Returns:
            bool: 命令是否发送成功
        """
        cmd = f"set_digital_out({pin}, {1 if value else 0})"
        logger.info(f"数字输出: pin={pin}, value={value}")
        return self._send_command(cmd)

    def set_analog_out(self, pin: int, value: float) -> bool:
        """
        设置模拟输出

        Args:
            pin: 引脚编号 (0-3)
            value: 值 (0.0-1.0)

        Returns:
            bool: 命令是否发送成功
        """
        cmd = f"set_analog_out({pin}, {value})"
        logger.info(f"模拟输出: pin={pin}, value={value}")
        return self._send_command(cmd)

    def get_actual_joint_positions(self) -> Optional[JointAngles]:
        """
        获取实际关节角度（需要从实时数据端口读取）

        Returns:
            Optional[JointAngles]: 当前关节角度
        """
        # 注意：这需要连接到实时数据端口 (30003)
        # 这里返回缓存值
        return self._current_joints

    def get_actual_tcp_pose(self) -> Optional[Pose]:
        """
        获取实际 TCP 位姿（需要从实时数据端口读取）

        Returns:
            Optional[Pose]: 当前 TCP 位姿
        """
        # 注意：这需要连接到实时数据端口 (30003)
        # 这里返回缓存值
        return self._current_pose

    def _start_monitor(self):
        """启动状态监控线程"""
        def monitor_thread():
            while self.is_connected:
                try:
                    # TODO: 从实时数据端口 (30003) 读取状态
                    # 这里简化为定期发送查询命令
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"监控线程错误: {e}")
                    break

        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()


if __name__ == "__main__":
    # 测试代码
    controller = UR5Controller()

    # 添加回调
    def on_status_change(status):
        print(f"状态变化: {status.value}")

    def on_error(error):
        print(f"错误: {error}")

    controller.add_status_callback(on_status_change)
    controller.add_error_callback(on_error)

    # 尝试连接
    if controller.connect():
        print("连接成功")

        # 测试移动
        pose = Pose(x=0.3, y=0.2, z=0.4, rx=0, ry=3.14, rz=0)
        controller.movel(pose, acceleration=0.5, velocity=0.3)

        time.sleep(2)
        controller.disconnect()
    else:
        print("连接失败")
