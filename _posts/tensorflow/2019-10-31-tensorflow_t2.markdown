---
layout: post
title: tensorflow 使用要点
date: 2019-10-27 00:05:00
author: wellstone-cheng
img: tensorflow.jpeg
tensorflow: true
tags: [tensorflow]
---
## 1. tensorflow V2.x使用要点
### 1.1 @tf.function
在TF 2.0里面, 如果需要构建计算图, 我们只需要给python函数加上@tf.function的装饰器使得eager exection默认打开。如果不使用@tf.function，虽然可以使用eager exection去写代码，但是模型的中间结果无法保存，所以无法进行预测.
TF 2.0的其中一个重要改变就是去除tf.Session. 这个改变会迫使用户用更好的方式来组织代码: 不用再用让人纠结的tf.Session来执行代码, 就是一个个python函数, 加上一个简单的装饰器.
#### 1.1.1 用@tf.function装饰器来将<font color='red'>python代码</font>转成<font color='red'>图表示代码</font>
**代码例1.1**:
``` python
import tensorflow as tf
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x)
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y
f()
```
打印如下:
``` shell
PRINT:  tf.Tensor([[10. 10.][11.  1.]], shape=(2, 2), dtype=float32)
TF-PRINT:  [[10 10][11 1]]
```
加入@tf.function后,**代码例1.2**:
``` python
import tensorflow as tf

@tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x)
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```
打印如下
``` shell
PRINT:  Tensor("MatMul:0", shape=(2, 2), dtype=float32)
TF-PRINT:  [[10 10][11 1]]
```
#### 1.1.2 不能在被装饰函数中初始化<font color='red'>tf.Variable</font>
**代码例1.3**:
``` python
import tensorflow as tf

@tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```
打印如下:
``` shell
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
ValueError: tf.function-decorated function tried to create variables on non-first call.
```
#### 1.1.3 用变量作用域继承(对象属性)在函数外初始化的变量
**代码例1.4**:
``` python
import tensorflow as tf

class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10,10],[11.,1.]])
        x = tf.constant([[1.,0.],[0.,1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y
f = F()
f()
```
打印如下
``` shell
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
TF-PRINT:  [[22 22][23 13]]
```
#### 1.1.4 用参数传入的方法使用在函数外初始化的变量
**代码例1.5**:
``` python
import tensorflow as tf

@tf.function
def f(b):
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y
b = tf.Variable(12.)
f(b)
```
打印如下
``` shell
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
TF-PRINT:  [[22 22][23 13]]
```
## 2. tensorflow V1.x使用要点
### 2.1 tf.Session
Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过 **session.run()** 来进行。想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer。
**代码例2.1**:
``` python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

g = tf.Graph() #step1 初始化计算图
with g.as_default(): # 设置为默认计算图
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b # step2 设计计算图
    init_op = tf.global_variables_initializer() # step4 初始化参数
    print("a is ", a)
    print("x is ", x)
    print("b is ", b)
    print("y is ",y)
with tf.Session(graph=g) as sess: # step3 创建并配置tf.Session
    sess.run(init_op) # step5 执行节点
    print("a is ", sess.run(a))
    print("x is ",sess.run(x))
    print("b is ",sess.run(b))
    print("y is ",sess.run(y)) # step6 输出结果
```
输出如下
``` shell
a is  Tensor("Const:0", shape=(2, 2), dtype=float32)
x is  Tensor("Const_1:0", shape=(2, 2), dtype=float32)
b is  <tf.Variable 'Variable:0' shape=() dtype=float32_ref>
y is  Tensor("add:0", shape=(2, 2), dtype=float32)
a is  [[10. 10.][11.  1.]]
x is  [[1. 0.][0. 1.]]
b is  12.0
y is  [[22. 22.][23. 13.]]
```
## 3 图变量
## 4 Eager Execution
### 4.1 传统的tensorflow是基于符号的计算图
在传统的TensorFlow开发中，我们需要首先通过变量和Placeholder来定义一个计算图，然后启动一个Session，通过TensorFlow引擎来执行这个计算图，最后给出我们需要的结果。相信大家在入门阶段，最困惑的莫过于想要打印某些向量或张量的值，在Session之外或未执行时，其值不可打印的问题。TensorFlow采用这种反人性的设计方式，主要是为了生成基于符号的计算图，然后通过C++的计算图执行引擎，进行各种性能优化、分布式处理，从而获取优秀的运行时性能。

见**代码例2.1**

### 4.2 pytorch直接计算结果
与此形成对照的是以PyTorch为代表的动态图方式，其不用生成基于符号表示的计算图，直接计算结果，与我们平常编程的处理方式类似，无疑这种方式学习曲线会低很多。
### 4.3 新版的tensorflow
TensorFlow实际上也注意到了这一问题，在2017年11月(v1.5?)，推出的Eager Execution就是这种动态图机制在TensorFlow中的实现。目前虽然Eager Execution在性能上还没有达到静态计算图的效率，但是由于其编程调试的方便性，会在实际应用中得到越来越广泛的应用。
* tensorflow 1.x代码
**代码例4.1** :
 ``` python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

g = tf.Graph() #step1 初始化计算图
with g.as_default(): # 设置为默认计算图
    tf.enable_eager_execution()  # 开启Eager Execution
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b # step2 设计计算图
    #init_op = tf.global_variables_initializer() # step4 初始化参数
    print("a is ", a)
    print("x is ", x)
    print("b is ", b)
    print("y is ",y)
    print("a.numpy is ", a.numpy())
    print("x.numpy is ", x.numpy())
    print("b.numpy is ", b.numpy())
    print("y.numpy is ",y.numpy())
 ```
 打印如下:
 ``` shell
a is  tf.Tensor([[10. 10.][11.  1.]], shape=(2, 2), dtype=float32)
x is  tf.Tensor([[1. 0.][0. 1.]], shape=(2, 2), dtype=float32)
b is  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=12.0>
y is  tf.Tensor([[22. 22.][23. 13.]], shape=(2, 2), dtype=float32)
a.numpy is  [[10. 10.][11.  1.]]
x.numpy is  [[1. 0.][0. 1.]]
b.numpy is  12.0
y.numpy is  [[22. 22.][23. 13.]]
 ```
由上面的代码可以看出，我们直接采用普通的程序形式，就可以求出矩阵乘法的结果，而且TensorFlow中的Tensor和numpy中的ndarray可以互相无缝转换，非常方便使用。更加有用的是，我们还可以利用使用机器中的GPU。
对比**代码例4.1**和**代码例2.1**的打印结果

* tensorflow 2.0代码
**代码例4.2** :
``` python
import tensorflow as tf
#g = tf.Graph() #step1 初始化计算图
#with g.as_default(): # 设置为默认计算图
#tf.enable_eager_execution()  # 开启Eager Execution
a = tf.constant([[10,10],[11.,1.]])
x = tf.constant([[1.,0.],[0.,1.]])
b = tf.Variable(12.)
y = tf.matmul(a, x) + b # step2 设计计算图
#init_op = tf.global_variables_initializer() # step4 初始化参数
print("a is ", a)
print("x is ", x)
print("b is ", b)
print("y is ",y)
print("a.numpy is ", a.numpy())
print("x.numpy is ", x.numpy())
print("b.numpy is ", b.numpy())
print("y.numpy is ",y.numpy())
```
打印如下:
``` shell
a is  tf.Tensor([[10. 10.][11.  1.]], shape=(2, 2), dtype=float32)
x is  tf.Tensor([[1. 0.][0. 1.]], shape=(2, 2), dtype=float32)
b is  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=12.0>
y is  tf.Tensor([[22. 22.][23. 13.]], shape=(2, 2), dtype=float32)
a.numpy is  [[10. 10.][11.  1.]]
x.numpy is  [[1. 0.][0. 1.]]
b.numpy is  12.0
y.numpy is  [[22. 22.][23. 13.]]
```


 ## 5 Tensorflow VS Pytorch之静态和动态图