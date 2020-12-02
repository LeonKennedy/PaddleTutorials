import paddle
import paddle.fluid as fluid
import numpy as np

BATCH_SIZE = 10

def generator():
    for i in range(5):
        yield np.random.uniform(low=0, high=255, size=[784, 784]),

image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
reader.decorate_sample_list_generator(
    paddle.batch(generator, batch_size=BATCH_SIZE))

executor = fluid.Executor(fluid.CPUPlace())
executor.run(fluid.default_startup_program())
for i in range(3):
    reader.start()
    while True:
        try:
            executor.run(feed=None)
        except fluid.core.EOFException:
            reader.reset()
            break