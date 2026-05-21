# Emotion_Classifer
人脸表情分类教学平台

使用Efficientnet_Lite0做为预训练模型，Python的http.server为网页服务器，ONNX Runtime Web为前端微调、推理工具，适合在教学环境探究监督学习的特点。训练、推理、数据全部在前端，后端只提供静态文件。局域网内使用摄像头、GPU加速功能要注意浏览器的相关权限unsafely-treat-insecure-origin-as-secure、enable-unsafe-webgpu必须开启。