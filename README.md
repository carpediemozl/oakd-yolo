# oakd-yolo
D:\blind_cane_yolov8



新建终端

python -m venv .venv

创建虚拟环境



pip install depthai

pip install opencv-python





window powershell 脚本禁止 管理员模式

Set-ExecutionPolicy RemoteSigned



激活虚拟环境

```
.\.venv\Scripts\Activate.ps1
```



确保有.venv前缀

安装yolov8

pip install ultralytics



测试

yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'

多runs文件夹



datacrosswalk为总的数据集

改为根目录下的traffic_dataset



.yaml文件内容

# 指向你的数据集文件夹

path: ./traffic_dataset

# 训练集和验证集图片目录

train: train/images
val: valid/images

# 类别名称

names:
  0: crosswalk
  1: guide_arrows
  2: blind_path
  3: red_light
  4: green_light



训练

yolo train data=blind_cane_data.yaml model=yolov8n.pt epochs=100 imgsz=640

一轮太慢 且没有用到GPU



nvidia-smi

安装支持GPU的PyTorch



卸载cpu版本的pytorch

pip uninstall torch torchvision torchaudio



CUDA版本是12.6 安装12.1版本兼容

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121



验证

python

import torch
print(torch.cuda.is_available())

输出true

exit()



yolov11为最新版本的yolov8nano

yolo train data=blind_cane_data.yaml model=yolov8n.pt epochs=30 imgsz=640

一轮90s左右

90x30=2700s

2700/60=45min



tensorboard 可视化查看训练结果



### 2. 搭建和测试树莓派硬件平台 (并行核心任务)

这是最重要的并行任务。等到模型训练完，你的硬件平台应该已经“万事俱备，只欠模型”了。

* **准备树莓派系统**:

  *   安装最新版的Raspberry Pi OS。
  *   配置好网络（WiFi/有线），开启SSH和VNC，这样你就可以在你的主电脑上远程控制树莓派了。

* **连接并测试摄像头**:

  * 将摄像头模块正确连接到树莓派的CSI接口。

  * 在树莓派的终端中，运行测试命令，确保摄像头能正常工作并拍照：

    ```bash
    libcamera-still -o test.jpg
    ```

  * 如果能成功生成一张`test.jpg`图片，说明摄像头没问题。这是最容易出问题的环节之一，提前解决它。

* **连接并测试反馈模块**:

  * 连接你的振动马达、蜂鸣器等到树莓派的GPIO引脚上。

  * 编写一个**极简的Python测试脚本**来验证它们能否被驱动。例如，测试连接在GPIO 17号引脚的振动马达：

    ```python
    import RPi.GPIO as GPIO
    import time
    
    MOTOR_PIN = 17 # 假设马达连接在17号引脚
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)
    
    print("测试振动马达... 振动3秒")
    GPIO.output(MOTOR_PIN, GPIO.HIGH) # 启动
    time.sleep(3)
    GPIO.output(MOTOR_PIN, GPIO.LOW)  # 停止
    print("测试结束")
    
    GPIO.cleanup()
    ```

  * 在树莓派上运行这个脚本，如果马达成功振动，说明硬件连接和基础驱动没问题。

---

### 3. 编写与模拟应用层代码 (在主电脑或树莓派上)

你不需要等最终模型出来，就可以开始编写使用模型的应用程序了。

* **搭建树莓派软件环境**:

  * 在树莓派上安装OpenCV和TFLite运行时：

    ```bash
    pip install opencv-python
    pip install tflite-runtime
    ```*   **编写推理脚本框架**:
    ```

  * 将在我们之前讨论过的 `inference.py` 脚本框架写好。包括加载模型、打开摄像头、读取视频帧、图像预处理等部分。

* **创建“模拟模型” (Mock/Dummy Model)**:

  * 这是软件开发的精髓。你可以写一个**假的检测函数**，它不进行真正的推理，而是返回一些预设的检测结果。

  * 例如，在你的 `inference.py` 中：

    ```python
    def run_dummy_inference(frame):
        # 模拟检测到一个在屏幕中间的“人”
        print("正在使用模拟模型...")
        time.sleep(0.1) # 模拟推理耗时
        person = {
            "class_name": "person",
            "confidence": 0.95,
            "box": [280, 150, 80, 200] # [x, y, w, h]
        }
        # 模拟检测到一个在右侧的“障碍物”
        obstacle = {
            "class_name": "obstacle",
            "confidence": 0.80,
            "box": [500, 200, 50, 50]
        }
        return [person, obstacle]
    ```

* **开发核心反馈逻辑**:

  * 现在，你可以用这个 `run_dummy_inference` 函数的输出来**开发和调试你所有的反馈逻辑**。

  * 例如：

    ```python
    detections = run_dummy_inference(frame)
    for det in detections:
        if det["class_name"] == "person":
            x, y, w, h = det["box"]
            if x < frame.shape[1] / 2:
                print("左侧有人，触发左侧马达！")
                # 在这里加入驱动左侧马达的GPIO代码
            else:
                print("右侧有人，触发右侧马达！")
                # 在这里加入驱动右侧马达的GPIO代码
    ```

  * 通过这种方式，你可以**在没有真实模型的情况下，完成90%的应用层代码开发和调试**。

### 总结

在你等待模型训练的几个小时里，你可以：

1.  **偶尔看一眼TensorBoard**，确保训练进展顺利。
2.  **把树莓派的软硬件环境完全搭好**，并测试好所有外设。
3.  **利用“模拟模型”的方法，把核心的应用程序逻辑和反馈机制代码全部写完并调试好**。

这样一来，当你的模型 `best.pt` 训练完成并导出为 `best.tflite` 后，你所要做的就只是把它传到树莓派上，然后用一行真实推理代码替换掉你的模拟函数。整个项目就能立刻无缝地运行起来！这是一种效率极高的并行开发模式。



最终目标是为OAK-D相机生成 .blob 文件，而 .blob 文件是从 **.onnx** 文件转换而来的

不需要转TFLite格式

安装ONNX库

pip install onnx onnxruntime



Model summary (fused): 72 layers, 3,006,623 parameters, 0 gradients, 8.1 GFLOPs



pip install onnx onnxruntime

yolo export model=runs/detect/train4/weights/best.pt format=onnx



报错

**我们手动将** **sympy** **库降级回** **torch** **需要的那个版本**

pip install --force-reinstall sympy==1.13.1



blob文件和inference.py文件



是ONNX文件转还是PT文件转BLOB suoerblob文件和blob文件

yolo train data=blind_cane_data.yaml model=runs/detect/train4/weights/best.pt.pt epochs=2 imgsz=640

tensorboard --logdir="D:\codefield\blind_cane_yolov8\runs\detect\train5"



python train_and_save.py

https://release-assets.githubusercontent.com/github-production-release-asset/434826774/fe1ca5f4-0cb3-4168-8c3c-cc6394a5e698?sp=r&sv=2018-11-09&sr=b&spr=https&se=2025-09-19T10%3A03%3A42Z&rscd=attachment%3B+filename%3Dclash-verge_1.3.8_amd64.deb&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2025-09-19T09%3A03%3A21Z&ske=2025-09-19T10%3A03%3A42Z&sks=b&skv=2018-11-09&sig=dLUZwNjhewBEA0lP2mkvEwCXMt2AIdGQLAz6fwFsfXw%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc1ODI3Mjk3MSwibmJmIjoxNzU4MjcyNjcxLCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.UscOsojeVCY1phe5F9NLr5UA4Xp42zEymLENQRZl0T8&response-content-disposition=attachment%3B%20filename%3Dclash-verge_1.3.8_amd64.deb&response-content-type=application%2Foctet-stream
