# Jetson 模型部署：ONNX 在板上转 TensorRT

Windows 用 OpenCV 读 ONNX 推理；**Jetson 用 TensorRT 读 ``.engine`` 推理**。两者是两条通道，
由 ``modules/vision/config.json`` 的 ``model.backend`` 决定（默认 ``auto``：Windows 走 ONNX，
Jetson 走 TensorRT）。这份文档只讲 Jetson 那半条：怎么把导出的 ONNX 变成板上能用的 engine。

## 为什么 engine 必须在 Jetson 本地生成

``.engine`` 不是模型文件，是**为某一块 GPU + 某一个 TensorRT 版本编译好的推理计划**。
把它从 PC 拷到板上，反序列化会因为版本/架构不匹配而失败（或者更糟：看起来加载成功、
结果全错）。所以：

- 仓库里**不放** ``models/best.engine``（``models/`` 本来也是 gitignore 的）；
- 换成新模型时，engine 一定要在板上重新生成，不能复用 PC 上的产物、也不能复用旧模型的 engine。

## 前置条件（JetPack 5.1 / L4T 35.6.1）

| 项目 | 本机实测 |
|---|---|
| TensorRT | 8.5.2.2（``python3 -c "import tensorrt; print(tensorrt.__version__)"``） |
| ``trtexec`` | ``/usr/src/tensorrt/bin/trtexec``（TensorRT v8502） |
| Python 环境 | ``/home/pi/miniforge3/envs/tensorrt``（Python 3.8.20） |
| pycuda | 2022.1 |

**用 conda 的 ``tensorrt`` 环境跑，不要用系统 ``python3``**：系统 Python 3.8 里没有
``pycuda``（也没装 ``onnx``/``torch``），推理通道起不来。

```bash
cd ~/gong_chuang
PY=/home/pi/miniforge3/envs/tensorrt/bin/python
$PY -c "import tensorrt, pycuda; print('ok')"
```

## 部署步骤

### 1. 在 PC 上训练并导出 ONNX

```powershell
python modules/vision/training/train.py --export-onnx
```

导出的是**静态**输入（``dynamic=False``）：OpenCV 和本项目的预处理都要求固定边长。
把导出的文件放到 ``models/best.onnx``。

导出后先确认模型自己声明的两件事——输入边长和类别表——再决定配置写什么：

```powershell
python -c "import onnx; m=onnx.load('models/best.onnx'); print([(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]); print({p.key:p.value for p in m.metadata_props if p.key in ('names','imgsz')})"
```

当前物料模型（2026-10-03 训练）读到的是：

```
输入  images  [1, 3, 320, 320]
names {0: 'red', 1: 'yellow', 2: 'blue', 3: 'green'}      # 1 红 2 黄 3 蓝 4 绿
imgsz [320, 320]
输出  output0 [1, 8, 2100]                                 # 4 + 4 类
```

### 2. 拷到板子

```powershell
# PC 上，项目根目录
pscp -pw 222222 models\best.onnx pi@192.168.1.7:/home/pi/gong_chuang/models/best.onnx
```

只拷 ONNX（10 MB 上下），**不要拷 engine**。

### 3. 在板上构建 engine

```bash
cd ~/gong_chuang
PY=/home/pi/miniforge3/envs/tensorrt/bin/python
$PY -m modules.vision.training.build_engine
```

脚本会自己从 ``modules/vision/config.json`` 的 ``model`` 块读 ``onnx`` / ``engine`` 路径，
找不到就退回 ``models/best.onnx`` → ``models/best.engine``。默认 FP16 开、工作区 2048 MiB。

构建完它会**接着做一次验收**（加载引擎 + 报形状 + 计时），不需要你再敲一条命令。

本机实测（Orin Nano，320×320 四分类）：

```
trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --memPoolSize=workspace:2048MiB --fp16
构建用时     530 s
engine       6.7 MB
输入张量     (1, 3, 320, 320)
输出张量     (1, 8, 2100)
单帧推理     3.9 ms（50 次平均，含显存拷贝）
端到端       13.3 ms（1280×720 一帧：预处理 + 推理 + 解码 + NMS）
```

### 4. 验收整条链路

```bash
$PY -m modules.vision.training.build_engine --check --image models/top_pic_2.jpg
```

``--check`` 会打印引擎形状、逐帧耗时，并在真实图片上跑一遍**预处理 → 引擎 → 解码 → 类名**，
把每个框的类名和分数列出来。拿它跟 PC 上 OpenCV 通道的结果对照，两边应当一致：

| 图片 | Windows（OpenCV ONNX） | Jetson（TensorRT FP16） |
|---|---|---|
| ``models/top_pic_2.jpg`` | blue 0.956 / green 0.611 | blue 0.956 / green 0.609 |

分数差 0.002 来自 FP16，属于正常范围；**类别名和框位置一致**才算通过。

## 常用命令

| 目的 | 命令 |
|---|---|
| 构建（已存在且比 ONNX 新就跳过） | `python -m modules.vision.training.build_engine` |
| 强制重建（换模型后） | `python -m modules.vision.training.build_engine --force` |
| 关掉 FP16 | `python -m modules.vision.training.build_engine --no-fp16` |
| 不用 trtexec，走 TensorRT Python API | `python -m modules.vision.training.build_engine --python-builder` |
| 只检查已有 engine | `python -m modules.vision.training.build_engine --check` |
| 检查 + 真实图片端到端 | `python -m modules.vision.training.build_engine --check --image models/top_pic_2.jpg` |
| 指定类别表验收 | `... --check --class-names red,yellow,blue,green` |

## 换模型的完整流程

1. 训练、导出新的 ``best.onnx``（保持 ``dynamic=False``）；
2. 按上面那条 ``onnx`` 命令读出**新的输入边长和类别表**；
3. 改 ``modules/vision/config.json``：``model.class_names``（顺序必须等于训练时的类别顺序）、
   ``model.input_size``（没配就由代码去问模型/引擎自己）；
4. 把 ONNX 拷到板子，``--force`` 重建 engine；
5. ``--check --image`` 验收，再跑菜单 18 的真机预览。

## 常见故障

**``加载 engine 失败``／反序列化报版本不匹配**
engine 和板子上的 TensorRT 或 GPU 架构对不上。``--force`` 在板上重建；不要用别的机器
或旧模型的 engine。

**类名和实际颜色对不上（框是准的，字是错的）**
模型的输出通道数是 ``4 + 类别数``，拿它跟配置里的类别表对一下就能发现：4 类的模型是 8 个
通道，``--check`` 会在对不上时直接报出来。类别表在 ``modules/vision/config.json`` 的
``model.class_names``；顺序错了就是整体错位（例如 4 类模型索引 3 是 green，写成年份久的
``obj-yellow`` 就会把绿色的物料报成黄色）。

**``ModuleNotFoundError: No module named 'pycuda'``**
用了系统 ``python3``。换成 ``/home/pi/miniforge3/envs/tensorrt/bin/python``。

**``没找到 trtexec``**
脚本会自动退回 TensorRT 的 Python API 构建；想用 trtexec 就加
``--trtexec /usr/src/tensorrt/bin/trtexec``。构建仍然必须在板上做。

**构建时间很长（几分钟）**
正常。320×320 的模型在本机实测 530 s，其中大部分是 trtexec 的层融合与基准测试。
构建期间别断电；中途 Ctrl+C 会留下不完整的 engine 文件，重新构建时必须加 ``--force``。
