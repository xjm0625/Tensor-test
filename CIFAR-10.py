from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 定义全局变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir # /tmp/cifar10_data
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # 从URL中获得文件名
    filename = DATA_URL.split('/')[-1]
    # 合并文件路径
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 定义下载过程中打印日志的回调函数
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        # 下载数据集
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
        print()
        # 获得文件信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        # 解压缩
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


maybe_download_and_extract()