# Emotion_Classifer
人脸表情分类教学平台

使用Efficientnet_Lite0做为预训练模型，Python的http.server为网页服务器，ONNX Runtime Web为前端微调、推理工具，适合在教学环境探究监督学习的特点。训练、推理、数据全部在前端，后端只提供静态文件。局域网内使用摄像头、GPU加速功能要注意浏览器的相关权限unsafely-treat-insecure-origin-as-secure、enable-unsafe-webgpu必须开启。

## 启动方式

### GUI 模式（默认）

双击运行或直接执行脚本，弹出图形界面窗口：

```bash
python server.py
```

启动后自动开启 Web 服务器并打开浏览器，可在界面中修改端口、启停服务器。

### 静默模式

使用 `-p` 参数指定端口号，以静默模式启动（不弹出 GUI 窗口，不依赖 tkinter，日志输出到终端），适合服务器部署：

```bash
python server.py -p 5000
```

- 不导入 tkinter，无需图形环境
- HTTP 请求日志直接打印到终端
- 按 `Ctrl+C` 停止服务器