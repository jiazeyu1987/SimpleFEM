# UR5 机械臂控制面板

基于 Python 和 Tkinter 的 UR5 机械臂控制面板，支持 MOVEL/MOVEJ 移动控制。

## 架构设计

采用 MVC 模式，实现 UI 与业务逻辑分离：

```
┌─────────────────────────────────────────────┐
│          UR5ControlPanel (UI)              │
│  - tkinter 图形界面                         │
│  - 用户交互处理                             │
│  - 状态显示                                 │
└──────────────────┬──────────────────────────┘
                   │ 调用
                   ▼
┌─────────────────────────────────────────────┐
│          UR5Controller (Controller)         │
│  - TCP/IP 通信管理                          │
│  - URScript 命令发送                        │
│  - 状态监控                                 │
└──────────────────┬──────────────────────────┘
                   │ TCP/IP
                   ▼
┌─────────────────────────────────────────────┐
│              UR5 机械臂                      │
│  - IP: 192.168.1.10                         │
│  - Port: 30002 (secondary client)           │
└─────────────────────────────────────────────┘
```

## 文件结构

```
vein_detector/
├── ur5_controller.py       # 控制管理类 (Controller)
├── ur5_control_panel.py    # UI 界面类 (View)
├── ur5_main.py            # 主入口文件
├── ur5_config.json        # 配置文件
└── README_UR5.md          # 本文档
```

## 功能特性

### 1. 连接管理
- TCP/IP 连接 UR5 机械臂
- 支持自定义 IP 和端口
- 实时连接状态显示
- 自动重连机制

### 2. 运动控制
- **MOVEL**: 笛卡尔空间直线移动
- **MOVEJ**: 关节空间快速移动
- **MOVEP**: 过渡移动（支持路径混合）
- 可调参数：
  - 加速度 (0.0 - 2.0)
  - 速度 (0.0 - 1.0)
  - 移动时间（可选）

### 3. 位姿控制
- 6 自由度笛卡尔坐标：
  - X, Y, Z (位置，单位：米)
  - RX, RY, RZ (姿态，单位：弧度)

### 4. 关节控制
- 6 关节角度控制 (J0 - J5，单位：弧度)

### 5. IO 控制
- 数字输出 (D0 - D7)
- 模拟输出 (A0 - A3，范围 0.0 - 1.0)

### 6. 快速预设
- **Home**: 标准工作位置
- **Zero**: 关节零点
- **Above**: 上方位置

## 安装依赖

```bash
# 无需额外依赖，使用 Python 标准库
# 需要 Python 3.7+

# 如需完整功能，确保安装了 tkinter (通常随 Python 一起安装)
```

## 使用方法

### 启动控制面板

```bash
# 方式1: 使用主入口
python ur5_main.py

# 方式2: 直接运行 UI
python ur5_control_panel.py
```

### 连接 UR5

1. 在 "连接设置" 区域输入 UR5 的 IP 地址（默认：192.168.1.10）
2. 确认端口号（默认：30002）
3. 点击 "连接" 按钮
4. 等待连接成功提示

### 执行移动

#### MOVEL（笛卡尔直线移动）
1. 在 "笛卡尔位姿" 区域输入目标位姿（X, Y, Z, RX, RY, RZ）
2. 选择移动类型为 "MOVEL"
3. 调整运动参数（加速度、速度）
4. 点击 "执行移动"

#### MOVEJ（关节移动）
1. 在 "关节角度" 区域输入目标关节角度（J0 - J5）
2. 选择移动类型为 "MOVEJ"
3. 调整运动参数（加速度、速度）
4. 点击 "执行移动"

### IO 控制

#### 数字输出
- 勾选/取消勾选 D0-D7 复选框控制数字输出
- True = 高电平，False = 低电平

#### 模拟输出
- 拖动滑块调节 A0-A3 模拟输出值
- 范围：0.0 - 1.0

## URScript 命令参考

控制面板通过发送 URScript 命令控制 UR5：

```python
# 关节移动
movej([j0, j1, j2, j3, j4, j5], a=1.0, v=0.5, t=0)

# 笛卡尔移动
movel(p[x, y, z, rx, ry, rz], a=1.0, v=0.5, t=0)

# 停止
stopj(2.0)  # 关节停止
stopl(2.0)  # 笛卡尔停止

# 数字输出
set_digital_out(pin, value)

# 模拟输出
set_analog_out(pin, value)
```

## 配置文件

`ur5_config.json` 配置说明：

```json
{
  "connection": {
    "default_host": "192.168.1.10",  // 默认 IP
    "default_port": 30002,            // 默认端口
    "timeout": 5.0                    // 连接超时
  },
  "motion": {
    "default_acceleration": 0.5,      // 默认加速度
    "default_velocity": 0.3           // 默认速度
  }
}
```

## API 使用示例

### UR5Controller 类

```python
from ur5_controller import UR5Controller, Pose, JointAngles

# 创建控制器
controller = UR5Controller(host="192.168.1.10", port=30002)

# 连接
if controller.connect():
    # MOVEL 移动
    pose = Pose(x=0.3, y=0.2, z=0.4, rx=0, ry=3.14, rz=0)
    controller.movel(pose, acceleration=0.5, velocity=0.3)

    # MOVEJ 移动
    joints = JointAngles(j0=0, j1=-1.57, j2=1.57, j3=-1.57, j4=-1.57, j5=0)
    controller.movej(joints, acceleration=0.5, velocity=0.3)

    # IO 控制
    controller.set_digital_out(0, True)
    controller.set_analog_out(0, 0.5)

    # 断开连接
    controller.disconnect()
```

### 添加状态回调

```python
def on_status_change(status):
    print(f"状态: {status.value}")

def on_error(error_msg):
    print(f"错误: {error_msg}")

controller.add_status_callback(on_status_change)
controller.add_error_callback(on_error)
```

## 安全注意事项

1. **首次使用前**：确保 UR5 机械臂已正确安装和配置
2. **工作空间限制**：确保目标位置在机械臂工作范围内
3. **速度控制**：首次测试时使用较低速度（0.1 - 0.3）
4. **紧急停止**：使用控制面板上的停止按钮或 UR5 示教器急停
5. **观察区域**：操作时确保有人观察机械臂运动

## 故障排除

### 连接失败
- 检查 IP 地址是否正确
- 确认 UR5 机械臂已启动
- 检查网络连接（ping 192.168.1.10）
- 确认端口 30002 未被占用

### 移动失败
- 确认已连接到机械臂
- 检查目标位姿是否在工作范围内
- 查看日志区域的错误信息

### IO 无响应
- 确认 UR5 软件版本支持 IO 控制
- 检查引脚编号是否正确

## 日志文件

程序运行日志保存在 `ur5_control.log`，包含：
- 连接状态变化
- 命令发送记录
- 错误信息

## 扩展开发

### 添加新的预设位置

在 `ur5_control_panel.py` 中添加：

```python
def _preset_custom(self):
    """自定义预设"""
    self.pose_x.set(0.5)
    self.pose_y.set(0.0)
    self.pose_z.set(0.3)
    # ... 设置其他参数
```

### 添加新的 URScript 命令

在 `UR5Controller` 类中添加方法：

```python
def custom_command(self) -> bool:
    """自定义命令"""
    cmd = "your_urscript_command()"
    return self._send_command(cmd)
```

## 技术支持

如有问题，请检查：
1. UR5 官方文档：https://www.universal-robots.com/
2. URScript 参考手册
3. 日志文件 `ur5_control.log`

## 许可证

SimpleFEM 项目的一部分

## 更新日志

### v1.0.0 (2025-12-30)
- 初始版本
- 支持 MOVEL/MOVEJ 移动
- 支持 IO 控制
- 图形化控制面板
