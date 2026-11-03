# 无头服务器可视化

仿真和模型调用在服务器上运行，通过 EGL 渲染；本地浏览器经 SSH 转发查看。
无需桌面环境、X11、VNC 或额外 Web 框架。默认服务仅绑定 `127.0.0.1`。

## 标准运行

服务器终端：

```bash
cd /home/liberow/embodied_agent/Robot-Dock
uv sync --extra mujoco
# 资源未准备时：uv run --group assets python scripts/fetch_assets.py
MUJOCO_GL=egl uv run robot-dock run configs/pokebot.mujoco.json --web-viewer --web-port 18765
```

在服务器 `.env` 设置 RoboAPI 密钥，或按终端提示输入。查看器不会接收或显示密钥。

Windows 另开 PowerShell，保持转发进程运行：

```powershell
ssh -N -L 127.0.0.1:18765:127.0.0.1:18765 gzhu-wangyu-liberow
```

浏览器访问 `http://127.0.0.1:18765/`。如果端口已被其他服务占用，请为
服务器 `--web-port` 和 SSH 转发同时选择一个空闲端口，不要停止不相关服务。

## 行为和边界

- 外部画面约每 3 个动作更新；腕部图像在采集策略观测时更新。
- 旋转、俯仰、距离只调整外部观察相机，模型使用的腕部相机不变。
- HTTP 线程只访问缓存和相机命令邮箱；预览渲染与相机应用在主线程，数据集传感器渲染在独立线程。
- 等待云端推理期间物理仿真继续运行；网页画面与相机请求在下一次渲染时更新。
- 浏览器断开不取消任务。任务结束时设备与模型会话已关闭，网页保留最后帧与结果。
- 结束后相机冻结，页面禁用控制。终端 Ctrl+C 退出查看服务。
- 本版提供画面和外部视角控制，不提供原生 MuJoCo 的物体拖拽、关节修改或任务启动按钮。
- 网页不包含机器人动作控制接口。相机参数校验、请求大小和同源检查均有测试。
- 关闭纯 SSH 转发不代表关闭服务器程序；需要在运行任务的终端退出程序。

