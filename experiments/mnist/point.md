
# 知识点：

#### 图像检测
1. 简单横向边界检测
`
# 创建初始化权重参数w
w = np.array([1, 0, -1], dtype='float32')
# 将权重参数调整成维度为[cout, cin, kh, kw]的四维张量
w = w.reshape([1, 1, 1, 3])
# 创建卷积算子，设置输出通道数，卷积核大小，和初始化权重参数
# filter_size = [1, 3]表示kh = 1, kw=3
# 创建卷积算子的时候，通过参数属性param_attr，指定参数初始化方式
# 这里的初始化方式时，从numpy.ndarray初始化卷积参数
conv = Conv2D(num_channels=1, num_filters=1, filter_size=[1, 3],
        param_attr=fluid.ParamAttr(
          initializer=NumpyArrayInitializer(value=w)))

`

2. 实际卷积的方法
`
# 设置卷积核参数
w = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype='float32')/8
w = w.reshape([1, 1, 3, 3])
w = np.repeat(w, 3, axis=1)
conv = Conv2D(num_channels=3, num_filters=1, filter_size=[3, 3],
        param_attr=fluid.ParamAttr(
          initializer=NumpyArrayInitializer(value=w)))
`


3. 实际均值模糊方法
`
w = np.ones([1, 1, 5, 5], dtype = 'float32')/25
conv = Conv2D(num_channels=1, num_filters=1, filter_size=[5, 5], param_attr=fluid.ParamAttr(
          initializer=NumpyArrayInitializer(value=w)))
`

