# UR5 控制面板快捷键功能说明

## 快捷键列表

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| **Ctrl+T** | 切换窗口置顶 | 使控制面板窗口永远在最上面 / 取消置顶 |

## 窗口置顶功能

### 功能描述
按下 `Ctrl+T` 后，控制面板窗口将切换到"置顶模式"，此时窗口会始终显示在其他所有窗口之上，不会被遮挡。再次按下 `Ctrl+T` 将取消置顶模式。

### 使用场景
- **机械臂操作时监控**：需要在操作其他软件时同时观察 UR5 控制面板状态
- **多任务处理**：在使用其他应用控制机械臂时，保持控制面板可见
- **演示教学**：在演示或教学时，确保控制面板始终可见

### 视觉反馈

#### 1. 窗口标题栏
- **置顶模式**：标题显示为 `⬆ UR5 机械臂控制面板`（带有向上箭头标识）
- **普通模式**：标题显示为 `UR5 机械臂控制面板`

#### 2. 状态显示区
在右侧面板的"状态"区域中：
- **置顶模式**：显示 `⬆ 置顶模式`（红色文字）
- **普通模式**：显示 `普通模式`（紫色文字）

#### 3. 日志区域
每次切换都会在日志中记录：
```
[SUCCESS] 窗口已切换到置顶模式
[INFO] 窗口已取消置顶
```

## 快捷键工作原理

### 热键管理器（HotKeyManager）
快捷键功能由 `HotKeyManager` 类实现，该类负责：
- 注册和管理窗口快捷键
- 处理快捷键事件
- 提供可扩展的快捷键接口

### 置顶指示器（TopmostIndicator）
`TopmostIndicator` 类负责：
- 管理窗口置顶状态
- 更新窗口标题栏显示
- 提供状态查询接口

### 集成到控制面板
快捷键功能已完全集成到 `UR5ControlPanel` 中：
- 启动时自动注册 `Ctrl+T` 快捷键
- 状态变化时自动更新 UI 显示
- 日志记录所有状态变化

## 代码示例

### 基本使用

```python
from vein_detector import UR5ControlPanel
import tkinter as tk

# 创建主窗口
root = tk.Tk()

# 创建控制面板（快捷键自动启用）
panel = UR5ControlPanel(root)

# 现在可以按 Ctrl+T 切换窗口置顶
root.mainloop()
```

### 自定义快捷键

```python
from vein_detector import HotKeyManager
import tkinter as tk

root = tk.Tk()
hotkey_manager = HotKeyManager(root)

# 注册自定义快捷键
def my_callback(event=None):
    print("快捷键被触发")

hotkey_manager.register_hotkey('<Control-s>', my_callback, "保存设置")

root.mainloop()
```

### 编程方式控制置顶

```python
from vein_detector import UR5ControlPanel
import tkinter as tk

root = tk.Tk()
panel = UR5ControlPanel(root)

# 编程方式设置置顶
panel.hotkey_manager.set_topmost(True)   # 设置为置顶
panel.hotkey_manager.set_topmost(False)  # 取消置顶

# 查询当前状态
if panel.hotkey_manager.is_topmost:
    print("窗口当前处于置顶模式")

root.mainloop()
```

## 扩展快捷键

如果需要添加更多快捷键，可以在 `_setup_hotkeys` 方法中注册：

```python
def _setup_hotkeys(self):
    """设置快捷键"""
    # 默认快捷键
    self.hotkey_manager.register_default_hotkeys(self._on_topmost_changed)

    # 添加自定义快捷键
    def toggle_connection(event=None):
        """快速连接/断开"""
        if self.controller.is_connected:
            self._on_disconnect()
        else:
            self._on_connect()
        return 'break'

    self.hotkey_manager.register_hotkey(
        '<Control-c>',
        toggle_connection,
        "快速连接/断开"
    )

    self._log("快捷键已注册: Ctrl+T 切换置顶, Ctrl+C 连接/断开", "INFO")
```

## 常见问题

### Q: 快捷键不生效？
**A**: 请确保：
1. 控制面板窗口处于焦点状态
2. 没有其他软件占用了 `Ctrl+T` 快捷键
3. Tkinter 窗口正常响应事件

### Q: 置顶后窗口无法移动？
**A**: 置顶模式不会阻止窗口移动，仍然可以拖动标题栏移动窗口位置。

### Q: 如何禁用快捷键功能？
**A**: 在 `_setup_hotkeys` 方法中注释掉注册代码：
```python
def _setup_hotkeys(self):
    # self.hotkey_manager.register_default_hotkeys(self._on_topmost_changed)
    pass
```

### Q: 可以修改快捷键吗？
**A**: 可以。在 `HotKeyManager` 中修改绑定的快捷键序列，例如将 `<Control-t>` 改为 `<Control-F1>`。

## 技术细节

### Tkinter 置顶实现
```python
# 设置窗口置顶
root.attributes('-topmost', True)

# 取消窗口置顶
root.attributes('-topmost', False)
```

### Tkinter 快捷键绑定
```python
# 绑定快捷键
root.bind('<Control-t>', callback)

# 解绑快捷键
root.unbind('<Control-t>')
```

### 快捷键格式
- `<Control-t>` / `<Control-T>`: Ctrl+T
- `<Control-c>` / `<Control-C>`: Ctrl+C
- `<F1>`: F1 功能键
- `<Escape>`: ESC 键
- `<Alt-f>`: Alt+F
- `<Shift-Delete>`: Shift+Delete

## 更新日志

### v1.1.0 (2025-12-30)
- 新增 `HotKeyManager` 快捷键管理类
- 新增 `TopmostIndicator` 置顶指示器类
- 实现 `Ctrl+T` 窗口置顶切换功能
- 在状态显示区域添加置顶状态和快捷键提示
- 标题栏显示置顶状态标识
