---
layout: post
title: tensorflow migrate from v1 to v2
date: 2019-10-27 00:05:00
author: wellstone-cheng
img: tensorflow.jpeg
tensorflow: true
tags: [tensorflow]
---
<font color='red'>此篇仅使用Tensorflow 低级的API,不适用高级的API(tf.keras)</font>

## TensorFlow 2.0同1.x之间的重要区别

* 在API层面的类、方法有了较大的变化，这个需要在使用中慢慢熟悉
* 取消了Session机制，每一条命令直接执行，而不需要等到Session.run
* 因为取消了Session机制，原有的数学模型定义，改为使用Python函数编写。原来的feed_dict和tf.placeholder，成为了函数的输入部分；原来的fetches，则成为了函数的返回值
* 使用keras的模型体系对原有的TensorFlow API进行高度的抽象，使用更容易
* 使用tf.keras.Model.fit来替代原有的训练循环

## 1. 安装了V2.x的tensorflow,使用V1.x的代码
### 1.1 使用tensorflow.compat.v1全局替换

``` python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```
### 1.2 使用tf.compat.v1.xxxx替代tf.xxxx
修复下列报错
``` shell
AttributeError: module 'tensorflow' has no attribute xxxx
```

注: 从TensorFlow2.0开始，默认情况下启用了Eager Execution;Tensorflow1.x 需添加相关代码执行.
``` python
# 启用动态图机制
tf.enable_eager_execution()
```
## 2. 安装了V2.x的tensorflow,移植V1.x的代码至V2.x
### 2.1 使用迁移工具tf_upgrade_v2转换V1.x代码至V2.x代码
* **命令**
``` shell
# 转化整个工程
tf_upgrade_v2 --intree my_project/ --outtree my_project_v2/ --reportfile report.txt
# 转化单个文件
tf_upgrade_v2 --infile first-tf.py --outfile first-tf-v2.py
```
* **效果**
``` python
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
```
变成如下:
``` python
with tf.compat.v1.name_scope('loss'):
    loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(ys - prediction),axis=[1]))
```
* **结论**
  * 使用tf.compat.v1.xxxx替代tf.xxxx
  * 函数入参变化
  
### 2.2 重构V1.x代码

V1.x代码
``` python
import tensorflow as tf
import numpy as np
x = np.float32(np.random.rand(100, 1))
y = np.dot(x, 0.5) + 0.7
b = tf.Variable(np.float32(0.3))
a = tf.Variable(np.float32(0.3))

y_value = tf.multiply(x, a) + b
loss = tf.reduce_mean(tf.square(y_value - y))
# TensorFlow内置的梯度下降算法，每步长0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 代价函数值最小化的时候，代表求得解
train = optimizer.minimize(loss)

# 初始化所有变量，也就是上面定义的a/b两个变量
init = tf.global_variables_initializer()

# 启动图
sess = tf.Session()
# 真正的执行初始化变量，还是老话，上面只是定义模型，并没有真正开始执行
sess.run(init)
# 重复梯度下降200次，每隔5次打印一次结果
for step in xrange(0, 200):
    sess.run(train)
    if step % 5 == 0:
        print
        step, sess.run(loss), sess.run(a), sess.run(b)
```
重构后
``` python
import tensorflow as tf
import numpy as np
x = np.float32(np.random.rand(100, 1))
y = np.dot(x, 0.5) + 0.7
b = tf.Variable(np.float32(0.3))
a = tf.Variable(np.float32(0.3))

@tf.function
def model(x):
    return a * x + b
# 定义代价函数，也是python函数
def loss(predicted_y, desired_y):
    return tf.reduce_sum(tf.square(predicted_y - desired_y))

# TensorFlow内置Adam算法，每步长0.1
optimizer = tf.optimizers.Adam(0.1)
# 还可以选用TensorFlow内置SGD(随机最速下降)算法，每步长0.001
# 不同算法要使用适当的步长，步长过大会导致模型无法收敛
# optimizer = tf.optimizers.SGD(0.001)

# 重复梯度下降200次，每隔5次打印一次结果
for step in range(0, 200):
    with tf.GradientTape() as t:
        outputs = model(x)  # 进行一次计算
        current_loss = loss(outputs, y)  # 得到当前损失值
        grads = t.gradient(current_loss, [a, b])  # 调整模型中的权重、偏移值
        optimizer.apply_gradients(zip(grads, [a, b]))  # 调整之后的值代回到模型
    if step % 5 == 0:  # 每5次迭代显示一次结果
        print("Step:%d loss:%%%2.5f weight:%2.7f bias:%2.7f " %
              (step, current_loss.numpy(), a.numpy(), b.numpy()))

```
