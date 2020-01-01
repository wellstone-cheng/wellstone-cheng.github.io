---
layout: post
title: tensorflow的安装
date: 2019-10-27 00:05:00
author: wellstone-cheng
img: tensorflow.jpeg
tensorflow: true
tags: [tensorflow]
---

# 1 Tensorflow 安装


## 1.1 基于docker安装
docker方式安装，并以jupyter server运行，可以通过网页远程访问

```
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-py3-jupyter
```
注：可以在jupyter页面通过new Terminal打开终端进行操作。


## 1.2 基于Python安装
- 先安装python，参照python study篇章

- 直接安装tensorflow

```
 pip3 install --user --upgrade tensorflow  # install in $HOME
```

- 验证安装

```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## 1.3 本地基于pip虚拟环境安装
- 先安装python，参照python study篇章

- 安装virtualenv

```
pip3 install -U virtualenv 
```

- 启动虚拟环境

```
        virtualenv --system-site-packages -p python3 ./venv
        source ./venv/bin/activate  # sh, bash, ksh, or zsh
（venv） pip install --upgrade pip
（venv） pip list  # show packages installed within the virtual environment
（venv） deactivate  # don't exit until you're done using TensorFlow
```

- 在virtualenv安装tensorflow

```
（venv）pip install --upgrade tensorflow
  Collecting tensorflow
  Cache entry deserialization failed, entry ignored
  Downloading https://files.pythonhosted.org/packages/de/f0/96fb2e0412ae9692dbf400e5b04432885f677ad6241c088ccc5fe7724d69/tensorflow-1.14.0-cp36-cp36m-manylinux1_x86_64.whl (109.2MB)
```

- 验证安装

```
（venv）python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
2019-11-02 22:41:53.733300: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-02 22:41:53.752820: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3408000000 Hz
2019-11-02 22:41:53.753392: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x50a5470 executing computations on platform Host. Devices:
2019-11-02 22:41:53.753417: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
tf.Tensor(1447.9882, shape=(), dtype=float32)
```
## 1.4本地基于conda虚拟环境安装

- 先安装python，参照python study篇章

- 启动虚拟环境

```
        conda create -n venv pip python=3.7  # select python version
        source activate venv
（venv） pip install --ignore-installed --upgrade packageURL
（venv） source deactivate
```

- 在virtualenv安装tensorflow

```
（venv）pip install --upgrade tensorflow
```

- 验证安装

```
（venv）python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## 1.5 安装过程遇到的问题及解决方案

### 1.5.1 Cache entry deserialization failed

- error details

```
# pip3 install --upgrade tensorflow
Collecting tensorflow
  Cache entry deserialization failed, entry ignored
```
- sulotion：升级pip

```
pip3 install --upgrade pip
```

### 1.5.2 FutureWarning

- error details

环境1 报错信息：

```
# python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
2019-11-03 03:50:03.730110: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-03 03:50:03.750617: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3408000000 Hz
2019-11-03 03:50:03.751483: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4ebcdc0 executing computations on platform Host. Devices:
2019-11-03 03:50:03.751528: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
tf.Tensor(398.2649, shape=(), dtype=float32)
```

环境2 保报错信息：

```
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
```

-Analysis
  环境1： Python 3.6.8 + pip 19.3.1 + numpy 1.17.3 +tensorflow 2.0.0 --> no this error
  环境2： Python 3.6.8 + pip 9.0.1 + numpy 1.17.3 +tensorflow 1.14.0 --> have this error
  
- solution
 将环境2的numpy版本降至1.16.0
 
```
# pip3 install numpy==1.16.0
# python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
Tensor("Sum:0", shape=(), dtype=float32)
```

### 1.5.3 Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

- error details

```
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
2019-11-02 22:41:53.733300: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-02 22:41:53.752820: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3408000000 Hz
2019-11-02 22:41:53.753392: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x50a5470 executing computations on platform Host. Devices:
2019-11-02 22:41:53.753417: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
tf.Tensor(1447.9882, shape=(), dtype=float32)
```
- Analysis

  环境3：Python 3.7.5 + pip 19.3.1 + numpy 1.17.3 +tensorflow 2.0.0 --> have this error
  环境2: Python 3.6.8 + pip 9.0.1 + numpy 1.16.0 +tensorflow 1.14.0 --> no this error
  环境4: Python 3.6.8 + pip 19.3.1 + numpy 1.17.3 +tf-nightly 2.1.0 --> have this error,but in jupyter no error
  
- solution
 更换tensorflow版本？

## 1.6 验证
- Python输入

```
import tensorflow as tf;
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```

- 输出

```
tf.Tensor(-380.3409, shape=(), dtype=float32)
```
