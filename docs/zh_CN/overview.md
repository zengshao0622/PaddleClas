# PaddleClas 代码解析

## 目录
- [1. 项目整体介绍](#1-项目整体介绍)
- [2. 代码解析](#2-代码解析)
    - [2.1 代码总体结构](#21-代码总体结构)
    - [2.2 代码运行逻辑](#22-代码运行逻辑)
- [3. 应用项目介绍](#3-应用项目介绍)
    - [3.1 PULC实用图像分类方案](#31-PULC实用图像分类方案)
    - [3.2 PP-ShiTu图像识别系统](#32-PP-ShiTu图像识别系统)


<a name="1. 项目整体介绍"></a>

## 1. 项目整体介绍
PaddleClas是一个致力于为工业界和学术界提供运用PaddlePaddle快速实现图像分类和图像识别的套件库，能够帮助开发者训练和部署性能更强的视觉模型。同时，PaddleClas提供了数个特色方案：PULC超轻量级图像分类方案、PP-ShiTU图像识别系统、PP系列骨干网络模型库和SSLD半监督知识蒸馏算法。
<div align="center">
<img src="https://user-images.githubusercontent.com/11568925/189267545-7a6eefa0-b4fc-4ed0-ae9d-7c6d53f59798.png"/>
<p>PaddleClas全景图</p>
</div>


<a name="2."></a>

## 2. 代码解析

<a name="2.1"></a>

### 2.1 代码总体结构

项目代码总体总体结构可参考下图：
<div align="center">
  <img src="https://user-images.githubusercontent.com/108920665/195777168-f59c15e2-91d3-4893-9cf4-0f93ce8b1cb6.png"/>
<p>代码结构图</p>
</div>

以下介绍各目录代码的作用。

<a name="2.1.1"></a>

### 2.1.1 benchmark

该目录存放了用于测试PaddleClas不同模型速度指标的shell脚本，如单卡训练速度指标、多卡训练速度指标等。以下是各脚本介绍：

- prepare_data.sh:下载相应的测试数据，并配置好数据路径。
- run_benchmark.sh:执行单独一个训练测试的脚本，具体调用方式，可查看脚本注释。
- run_all.sh: 执行所有训练测试的入口脚本。

具体介绍可以[参考文档](../../benchmark/README.md)。

<a name="2.1.2"></a>

### 2.1.2 dataset
该目录用于存放不同的数据集。数据集文件中应当包含数据集图像、训练集标签文件、验证集标签文件、测试集标签文件；数据集标签文件使用txt格式保存，标签文件中每一行描述一个图像数据，包括图像地址和真值标签，中间用分隔符隔开（默认为空格），格式如下：

```
jpg/image_06765.jpg 0
jpg/image_06755.jpg 0
jpg/image_05145.jpg 1
jpg/image_05137.jpg 1
```

<a name="2.1.3"></a>

### 2.1.3 deploy
该目录包含了PaddleClas模型部署以及PP-ShiTu相关代码。以下文档为部署以及PP-ShiTu相关介绍教程:

- [服务器端C++预测](../../deploy/cpp/readme.md)
- [分类模型服务化部署](../../deploy/paddleserving/readme.md)
- [基于PaddleHub Serving服务部署](../../deploy/hubserving/readme.md)
- [Slim功能介绍](../../deploy/slim/readme.md)
- [端侧部署](../../deploy/lite/readme.md)
- [paddle2onnx模型转化与预测](../../deploy/paddle2onnx/readme.md)
- [PP-ShiTu相关](models/PP-ShiTu/README.md)

<a name="2.1.4"></a>

### 2.1.4 docs
该目录存放了PaddleClas项目的中英文说明文档和相关说明图，包括项目教程、方法介绍、模型介绍、应用实例介绍等。

<a name="2.1.5"></a>

### 2.1.5 ppcls
该目录存放了PaddleClas的核心代码，下表详细介绍了该目录下各文件内容:

<div align="center">

<table border="1" >
<tr>
  <th align="left">文件夹名</th>
  <th align="left">功能</th>
  <th align="left">模块/字段/方法</th>
  <th align="left">模块功能</th>
  <th align="left">详细介绍</th>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="4" align="left">arch</th>
  <td rowspan="4" align="left">模型组网代码</th>
    <td>backbone</th>
    <td>骨干网络模型代码</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  <tr>
    <td>gears</th>
    <td>特征提取网络的Neck和Head部分代码</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>distill</th>
    <td>蒸馏相关的代码</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>slim</th>
    <td>模型剪枝和量化相关代码</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="6" align="left">configs</th>
  <td rowspan="6" align="left">各种模型和方法的参考配置文件，该目录包含大量配置文件可供参考，省略各配置文件的详细介绍，仅介绍配置文件中的常用字段。</th>
    <td>Global</th>
    <td>该部分描述整体的训练配置，包括预训练权重、预训练模型、输出地址、训练设备、训练epoch数、输入图像大小等</th>
    <td> - </th>
  <tr>
    <td>Arch</th>
    <td>该部分描述模型的网络结构参数，构建模型时主要调用该部分参数。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>Loss</th>
    <td>该部分描述损失函数的参数配置，包括训练和验证损失，损失类型，损失权重等，构建损失函数时调用。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>Optimizer</th>
    <td>该部分描述优化器部分的参数配置，构建优化器时调用。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>DataLoader</th>
    <td>该部分描述dataloader部分参数配置，包括训练和验证过程的数据采样策略、数据增广方法等。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>Metric</th>
    <td>该部分描述评价指标，包括训练和验证过程选择的评价指标及其参数配置。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="4" align="left">data</th>
  <td rowspan="4" align="left">数据处理相关代码</th>
    <td>dataloader</th>
    <td>包含针对不同数据集读取(dataset)和不同数据采样方式(sampler)的代码。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  <tr>
    <td>postprocess</th>
    <td>对模型输出结果的后处理，输出对应的类别名、置信度、预测结果等</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>preprocess</th>
    <td>对数据的预处理和数据增广方法。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>utils</th>
    <td>其他常用函数</th>
    <td> - </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="3" align="left">engine</th>
  <td rowspan="3" align="left">模型训练和验证过程逻辑代码</th>
    <td>evaluation</td>
    <td>模型验证过程的不同方法：分类、检索等</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  <tr>
    <td>train</th>
    <td>模型训练过程的主要流程代码。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>engine</th>
    <td>模型训练、验证整体流程串联引擎工具代码，负责构建训练和验证过程的各个模块（如model、dataloader、loss等），并负责启动训练或验证流程。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="2" align="left">loss</th>
  <td rowspan="2" align="left">损失函数相关代码，该目录包含了大量损失函数可供选择，各函数详细介绍省略，仅简单介绍损失函数类中的两个方法。</th>
    <td>_loss_method</th>
    <td>定义构造不同损失函数的计算方法</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  <tr>
    <td>forward</th>
    <td>定义计算损失函数的前向流程</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td>metric</th>
  <td>度量方法相关代码，该部分包含了各种度量方法，可通过设置配置文件的'Metric'字段进行选择和使用。</th>
  <td>-</th>
  <td>-</th>
  <td>-</th>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="2" align="left">optimizer</th>
  <td rowspan="2" align="left">训练优化器和学习率策略相关代码</th>
    <td>learning_rate.py</th>
    <td>不同学习率策略的代码</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  <tr>
    <td>optimizer.py</th>
    <td>不同训练优化器的代码</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
<tr>
  <td rowspan="4" align="left">utils</th>
  <td rowspan="4" align="left">该目录包含了其他常用的函数</th>
    <td>logger</th>
    <td> logger打印相关函数。定义了一个'_logger'，并在需要打印的位置import该文件。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  <tr>
    <td>config</th>
    <td>关于配置文件的操作</th>
    <td> - </th>
  </tr>
  <tr>
    <td>ema</th>
    <td> Exponential Moving Average，指数移动平均，用于根据参数加权历史均值更新当前参数。</th>
    <td> <a href="https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/model_list.md">文档</a> </th>
  </tr>
  <tr>
    <td>save_load</th>
    <td>保存、加载模型参数等操作。</th>
    <td> - </th>
  </tr>
</tr>
<!/ ------------------------------------------------>
</table>

</div>


<a name="2.2"></a>

### 2.2 代码运行逻辑

<a name="3."></a>

## 3. 应用项目介绍

<a name="3.1"></a>

### 3.1 PULC实用图像分类方案

<a name="3.2"></a>

### 3.2 PP-ShiTu图像识别系统
