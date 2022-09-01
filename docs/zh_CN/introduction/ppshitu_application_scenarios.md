# PP-ShiTu应用场景介绍

该文档介绍了PP-ShiTu提供的各种应用场景库简介、下载链接以及使用简介。

------

## 目录

- [1. 应用场景介绍](#1-应用场景介绍)
- [2. 使用说明](#2-使用说明)
  - [2.1 环境配置](#21-环境配置)
  - [2.2 下载、解压场景库数据](#22-下载解压场景库数据)
  - [2.3 准备模型](#23-准备模型)
  - [2.4 场景库识别与检索](#24-场景库识别与检索)
    - [2.4.1 识别单张图像](#241-识别单张图像)
    - [2.4.2 基于文件夹的批量识别](#242-基于文件夹的批量识别)
  - [3 新增场景库图像识别体验](#3-新增场景库图像识别体验)
    - [3.1 根据相似度划分Gallery和Query](#31-根据相似度划分gallery和query)
    - [3.2 根据划分好的Gallery构建index索引](#32-根据划分好的gallery构建index索引)

<a name="1. 应用场景介绍"></a>

## 1. 应用场景介绍

PP-ShiTu应用场景介绍和下载地址如下表所示。

| 场景 |示例图|场景简介|Recall@1|场景库下载地址|原数据集下载地址|
|:---:|:---:|:---:|:---:|:---:|:---:|
| 球类 | --- |各种球类识别 | 0.9769 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/balls-image-classification) |
| 狗识别 | --- | 狗细分类识别，包括69种狗的图像 | 0.9606 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set) |
| 宝石 | --- | 宝石种类识别 | 0.9653 | --- | [原数据下载地址](https://www.kaggle.com/datasets/lsind18/gemstones-images) |
| 动物 | --- |各种动物识别 | 0.9078 | --- | [原数据下载地址](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) |
| 鸟类 | --- |鸟细分类识别，包括400种鸟类各种姿态 | 0.9673 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) |
| 交通工具 | --- |车、船等交通工具粗分类识别 | 0.9307 | --- | [原数据下载地址](https://www.kaggle.com/datasets/rishabkoul1/vechicle-dataset) |
| 花 | --- |104种花细分类识别 | 0.9788 | --- | [原数据下载地址](https://www.kaggle.com/datasets/msheriey/104-flowers-garden-of-eden) |
| 运动种类 | --- |100种运动图像识别 | 0.9413 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/sports-classification) |
| 乐器 | --- |30种不同乐器种类识别 | 0.9467 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/musical-instruments-image-classification) |
| 宝可梦 | --- |宝可梦神奇宝贝识别 | 0.9236 | --- | [原数据下载地址](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) |
| 船 | --- |船种类识别 |0.9242 | --- | [原数据下载地址](https://www.kaggle.com/datasets/imsparsh/dockship-boat-type-classification) |
| 鞋子 | --- |鞋子种类识别，包括靴子、拖鞋等 | 0.9000 | --- | [原数据下载地址](https://www.kaggle.com/datasets/noobyogi0100/shoe-dataset) |
| 巴黎建筑 | --- |巴黎著名建筑景点识别，如：巴黎铁塔、圣母院等 | 1.000 | --- | [原数据下载地址](https://www.kaggle.com/datasets/skylord/oxbuildings) |
| 蝴蝶 | --- |75种蝴蝶细分类识别 | 0.9360 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species) |
| 野外植物 | --- |野外植物识别 | 0.9758 | --- | [原数据下载地址](https://www.kaggle.com/datasets/ryanpartridge01/wild-edible-plants) |
| 天气 | --- |各种天气场景识别，如：雨天、打雷、下雪等 | 0.9924 | --- | [原数据下载地址](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) |
| 坚果 | --- |各种坚果种类识别 | 0.9412 | --- | [原数据下载地址](https://www.kaggle.com/datasets/gpiosenka/tree-nuts-image-classification) |
| 时装 | --- |首饰、挎包、化妆品等时尚商品识别 | 0.9555 | --- | [原数据下载地址](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) |
| 垃圾 | --- |12种垃圾分类识别 | 0.9845 | --- | [原数据下载地址](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) |
| 航拍场景 | --- |各种航拍场景识别，如机场、火车站等 | 0.9797 | --- | [原数据下载地址](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) |
| 蔬菜 | --- |各种蔬菜识别 | 0.8929 | --- | [原数据下载地址](https://www.kaggle.com/datasets/zhaoyj688/vegfru) |
| 商标 | --- |两千多种logo识别 | 0.9313 | --- | [原数据下载地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |



<a name="2. 使用说明"></a>

## 2. 使用说明

<a name="2.1 环境配置"></a>

### 2.1 环境配置
- 安装：请先参考文档[环境准备](../installation/install_paddleclas.md)配置PaddleClas运行环境
- 进入`deploy`运行目录，本部分所有内容与命令均需要在`deploy`目录下运行，可以通过下面命令进入`deploy`目录。
```shell
cd deploy
```


<a name="2.2 下载、解压场景库数据"></a>

### 2.2 下载、解压场景库数据
首先创建存放场景库的地址`deploy/datasets`:

```shell
mkdir datasets
```
下载并解压对应场景库到`deploy/datasets`中。
```shell
cd datasets

# 下载并解压场景库数据
wget {场景库下载链接} && tar -xf {压缩包的名称}
```
以`dataset_name`为例，解压完毕后，`datasets/dataset_name`文件夹下应有如下文件结构：
```shel
├── dataset_name/
│   ├── Gallery/
│   ├── Index/
│   ├── Query/
│   ├── gallery_list.txt/
│   ├── query_list.txt/
│   ├── image_list.txt/
├── ...
```
其中，`Gallery`文件夹中存放的是用于构建索引库的原始图像，`Index`表示基于原始图像构建得到的索引库信息，`Query`文件夹存放的是用于检索的图像列表，`gallery_list.txt`和`query_list.txt`分别为索引库和检索图像的标签文件。

<a name="2.3 准备识别模型"></a>

### 2.3 准备模型
创建存放模型的文件夹`deploy/models`，并下载轻量级主体检测、识别模型，命令如下：
```shell
cd ..
mkdir models
cd models

# 下载检测模型并解压
wget {检测模型下载链接} && tar -xf {检测模型压缩包名称}

# 下载识别 inference 模型并解压
wget {识别模型下载链接} && tar -xf {识别模型压缩包名称}
```

解压完成后，`models`文件夹下有如下文件结构：
```
├── inference_model_name
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── det_model_name
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

<a name="2.4 场景库识别与检索"></a>

### 2.4 场景库识别与检索

以`动物识别`场景为例，展示识别和检索过程（如果希望尝试其他场景库的识别与检索效果，在下载解压好对应的场景库数据和模型后，替换对应的配置文件即可完成预测）。

注意，此部分使用了`faiss`作为检索库，安装方法如下：
```shell
pip install faiss-cpu==1.7.1post2
```

若使用时，不能正常引用，则`uninstall`之后，重新`install`，尤其是在windows下。

<a name="2.4.1 识别单张图像"></a>

#### 2.4.1 识别单张图像

假设需要测试`./datasets/AnimalImageDataset/Query/antelope/0a37838e99.jpg`这张图像识别和检索效果。

首先分别修改配置文件`./configs/inference_general.yaml`中的`Global.det_inference_model_dir`和`Global.rec_inference_model_dir`字段为对应的检测和识别模型文件夹，以及修改测试图像地址字段`Global.infer_imgs`示例如下：

```shell
Global:
  infer_imgs: './datasets/AnimalImageDataset/Query/antelope/0a37838e99.jpg'
  det_inference_model_dir: './models/det_model_name'
  rec_inference_model_dir: './models/inference_model_name'
```

并修改配置文件`./configs/inference_general.yaml`中的`IndexProcess.index_dir`字段为对应场景index库地址：

```shell
IndexProcess:
  index_dir:'./datasets/AnimalImageDataset/Index/SampleAll/index'
```


运行下面的命令，对图像`./datasets/AnimalImageDataset/Query/antelope/0a37838e99.jpg`进行识别与检索

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml

# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.use_gpu=False
```

最终输出结果如下：
```
[{'bbox': [264, 79, 1088, 850], 'rec_docs': 'antelope', 'rec_scores': 0.81452656}]
```
其中`bbox`表示检测出的主体所在位置，`rec_docs`表示索引库中与检测框最为相似的类别，`rec_scores`表示对应的置信度。
检测的可视化结果也保存在`output`文件夹下，对于本张图像，识别结果可视化如下所示。

![](../../images/ppshitu_application_scenarios/rec_result.jpg)

<a name="2.4.2 基于文件夹的批量识别"></a>

#### 2.4.2 基于文件夹的批量识别

如果希望预测文件夹内的图像，可以直接修改配置文件中`Global.infer_imgs`字段，也可以通过下面的`-o`参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.infer_imgs="./datasets/AnimalImageDataset/Query/antelope"
```
终端中会输出该文件夹内所有图像的识别结果，如下所示。
```
...
[{'bbox': [0, 0, 1200, 675], 'rec_docs': 'antelope', 'rec_scores': 0.6153812}]
[{'bbox': [0, 0, 275, 183], 'rec_docs': 'antelope', 'rec_scores': 0.77218026}]
[{'bbox': [264, 79, 1088, 850], 'rec_docs': 'antelope', 'rec_scores': 0.81452656}]
[{'bbox': [0, 0, 188, 268], 'rec_docs': 'antelope', 'rec_scores': 0.637074}]
[{'bbox': [118, 41, 235, 161], 'rec_docs': 'antelope', 'rec_scores': 0.67315465}]
[{'bbox': [0, 0, 175, 287], 'rec_docs': 'antelope', 'rec_scores': 0.68271667}]
[{'bbox': [0, 0, 310, 163], 'rec_docs': 'antelope', 'rec_scores': 0.6706451}]
...
```
所有图像的识别结果可视化图像也保存在`output`文件夹内。

<a name="3 新增场景库图像识别体验"></a>

### 3 新增场景库图像识别体验

该部分内容介绍根据新数据集创建场景库并进行指标验证和识别检索的方法。

<a name="3.1 根据相似度划分Gallery和Query"></a>

#### 3.1 根据相似度划分Gallery和Query

首先制作需要创建场景库数据集的`image_list.txt`，该文件格式如下：
```
# 每一行采用“空格”分割图像路径与标签
image_path_1 label_1
image_path_2 label_1
image_path_3 label_1
image_path_4 label_2
...
```

在配置文件`configs/index_selector.yaml`中的`Datasets`字段中添加新建场景库对应数据集的信息，格式如下：
```shell
Datasets:
  DatasetName:
    infer_path: "path/to/image"
    infer_imgs: "path/to/image_list.txt"
    output_path: "path/to/save/Gallery&Query"
    gallery_num: 20
    sim_thred: 0.88
```

其中，`DatasetName`表示新建场景库的名字，`infer_path`为保存新建场景库对应数据集图像的文件夹路径，`infer_imgs`为`image_list.txt`的路径，`output_path`为划分好gallery和query的保存路径，`gallery_num`为每类gallery图像的最大数量，`sim_thred`为选择query图像时与gallery图像的相似度阈值。

运行如下命令，根据相似度划分gallery和query：
```shell
python shitu_index_selector/index_selector.py -c configs/index_selector.yaml
```

输出结果如下：
```
Mean of gallery num: 20
Mean of query num: 26
```
其中`Mean of gallery num`表示划分的gallery库每类图像数均值，`Mean of query num`表示划分的query库每类图像数均值。

划分好的文件结构如下：
```
├── Gallery
│   ├── class1
│   │      ├──image1.jpg
│   │      ├──image2.jpg
│   │      ├──...
│   ├── class2
│   └── ...
├── Query
│   ├── class1
│   ├── class2
│   └── ...
├── gallery_list.txt
└── query_list.txt
```
其中`Gallery`和`Query`分别为gallery和query库图像，`gallery_list.txt`和`query_list.txt`分别为对应的标签文件。


<a name="3.2 根据Gallery构建index索引"></a>

#### 3.2 根据划分好的Gallery构建index索引
首先在配置文件`configs/sample_indexes.yaml`中的`Datasets`字段中添加新建场景库的信息，格式如下：
```shell
Datasets:
  DatasetName:
    infer_path: "path/to/Gallery/images"
    infer_imgs: "path/to/gallery_list.txt"
    output_dir: "path/to/save/index"
```
其中，`DatasetName`表示新建场景库的名字，`infer_path`为保存新建场景库划分好的gallery图像路径，`infer_imgs`为`gallery_list.txt`的路径，`output_path`为保存构建`index`索引的路径。

然后在配置文件`configs/sample_indexes.yaml`中的`Method`字段中添加构建`index`索引的方法，格式如下：
```shell
Methods:
  SampleAll:
    method_name: SampleAll

  RandomSample_10:
    method_name: RandomSample
    gallery_num: 10
```
其中PaddleClas提供了两种方法：`SampleAll`为根据提供的所有Gallery图像构建`index`索引，`RandomSample`为根据提供的Gallery图像每类随机选取`gallery_num`张图像构建`index`索引，可根据需要选择对应的方法。同时，也可以参考`shitu_index_selector/index_random_sample.py`和`shitu_index_selector/index_sample_all.py`实现其他图像选择方式构建`index`索引。

构建的`index`索引库文件结构如下：
```
├── RandomSample_10
│   ├── gallery
│   │      ├──class1
│   │      │     ├──image1.jpg
│   │      │     ├──...
│   │      ├──class2
│   │      └──...
│   └── index
│          ├──id_map.pkl
│          └──vector.index
└──SampleAll
    └── index
           ├──id_map.pkl
           └──vector.index
```
其中，`RadomSample`方法构建index索引后，会保存随机采样的Gallery图像到`gallery`文件夹。

使用新的`index`索引库进行图像识别，需要修改配置文件中的`IndexProcess.index_dir`字段来更改索引库的路径。

注意：划分Gallery、构建`index`索引库以及进行图像识别时，应当使用同一个识别模型（即三个步骤配置文件中的`Global.rec_inference_model_dir`字段保持一致）。