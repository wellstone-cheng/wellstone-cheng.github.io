---
layout: post
title: tensor array list
date: 2019-10-27 00:05:00
author: wellstone-cheng
img: tensorflow.jpeg
tensorflow: true
tags: [tensorflow]
---
## 1 tf 1.x版本
### 1.1 tensor && numpy
**代码例1.1**:
``` python
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
a = np.random.random((5,3))
b = np.random.randint(0,9,(3,1))
c = tf.tensordot(a.astype(np.float),b.astype(np.float),axes=1)
# tensor 转ndarray
with tf.Session() as sess:
    dn = c.eval()
    print(dn)
# ndarray转tensor
tn = tf.convert_to_tensor(dn)
print(tn)
```
打印如下:
``` shell
[[7.41453359][7.53395565][2.91013169][4.53876831][6.85017419]]
Tensor("Const:0", shape=(5, 1), dtype=float64)
```
#### 1.1.1 Numpy2Tensor
**代码例1.2**:
``` python
tn = tf.convert_to_tensor(dn)
```
#### 1.1.2 Tensor2Numpy
**代码例1.3**:
``` python
with tf.Session() as sess:
    dn = c.eval()
```
## 2 tf 2.x版本
### 2.1 tensor && numpy array
**代码例2.1**:
``` python
import numpy as np
import tensorflow as tf

a = np.random.random((5,3))
b = np.random.randint(0,9,(3,1))
c = tf.tensordot(a.astype(np.float),b.astype(np.float),axes=1)
# tensor 转ndarray
dn = c.numpy()
print(dn)
# ndarray转tensor
tn = tf.convert_to_tensor(dn)
print(tn)
```
打印如下:
``` shell
[[ 6.50293042][ 6.81634275][ 5.83577695][11.06985179][13.71887664]]
tf.Tensor([[ 6.50293042][ 6.81634275][ 5.83577695][11.06985179][13.71887664]], shape=(5, 1), dtype=float64)
```
#### 2.1.1 Numpy2Tensor
**代码例2.2**:
``` python
tn = tf.convert_to_tensor(dn)
```
#### 2.1.2 Tensor2Numpy
**代码例2.3**:
``` python
dn = c.numpy()
```
#### 2.1.3 1.x版本与2.x版本区别
* Numpy2Tensor两者相同,都是使用了tf.convert_to_tensor
* Tensor2Numpy两者不一样
  
(1) 1.x版本需使用session,然后通过 .eval() 转换,不能直接使用.numpy(),否则会报错
```
AttributeError: 'Tensor' object has no attribute 'numpy'
```
根本原因:1.x版本报上述错误是由于采用了session模式,而没开启Eager Execution模式;反之, 1.x版本开启Eager Execution模式也是可以使用.numpy()进行转换的,若使用tf.Graph().as_default(),则开启Eager Execution模式的语句需放在tf.Graph().as_default()不能放在tf.Graph().as_default()之前否则也会报上述错误.
(2) 由于2.x版本取消了session机制，开发人员可以直接执行 .numpy()方法转换tensor.
若放在tf.Graph().as_default()下也会报下面错误
```
AttributeError: 'Tensor' object has no attribute 'numpy'
```
根本原因:2.x版本是默认开启了Eager Execution模式,若使用tf.Graph().as_default()则开启了Eager Execution模式的语句是在tf.Graph().as_default()之前.
(3) 综上所述,出现
```
AttributeError: 'Tensor' object has no attribute 'numpy'
```
错误的原因如下:
either a missing call to tf.enable_eager_execution() or the code is being run inside a with tf.Graph().as_default() block
* 若使用session模式,则没有使用Eager Execution模式,所以.numpy会报错.
* 若使用Eager Execution且使用了tf.Graph().as_default,则需把开启Eager Execution模式的语句放在tf.Graph().as_default之下,否则.numpy会报错.