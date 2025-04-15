import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
import sys

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import fast_hist
from tf_record import read_record, read_bd_rm_record

GPU_ID = '0'

def data_loader_bd_rm_from_tfrecord(batch_size=1):
    paths = open('../dataset/r3d_train.txt', 'r').read().splitlines()

    loader_dict = read_bd_rm_record('../dataset/r3d.tfrecords', batch_size=batch_size, size=512)

    num_batch = len(paths) // batch_size

    return loader_dict, num_batch

class Network(object):
    """docstring for Network"""
    def __init__(self, dtype=tf.float32):
        print('Initial nn network object...')  # 更新为Python 3的print函数
        self.dtype = dtype
		self.pre_train_restore_map = {'vgg_16/conv1/conv1_1/weights':'FNet/conv1_1/W', # {'checkpoint_scope_var_name':'current_scope_var_name'} shape must be the same
									'vgg_16/conv1/conv1_1/biases':'FNet/conv1_1/b',	
									'vgg_16/conv1/conv1_2/weights':'FNet/conv1_2/W',
									'vgg_16/conv1/conv1_2/biases':'FNet/conv1_2/b',	
									'vgg_16/conv2/conv2_1/weights':'FNet/conv2_1/W',
									'vgg_16/conv2/conv2_1/biases':'FNet/conv2_1/b',	
									'vgg_16/conv2/conv2_2/weights':'FNet/conv2_2/W',
									'vgg_16/conv2/conv2_2/biases':'FNet/conv2_2/b',	
									'vgg_16/conv3/conv3_1/weights':'FNet/conv3_1/W',
									'vgg_16/conv3/conv3_1/biases':'FNet/conv3_1/b',	
									'vgg_16/conv3/conv3_2/weights':'FNet/conv3_2/W',
									'vgg_16/conv3/conv3_2/biases':'FNet/conv3_2/b',	
									'vgg_16/conv3/conv3_3/weights':'FNet/conv3_3/W',
									'vgg_16/conv3/conv3_3/biases':'FNet/conv3_3/b',	
									'vgg_16/conv4/conv4_1/weights':'FNet/conv4_1/W',
									'vgg_16/conv4/conv4_1/biases':'FNet/conv4_1/b',	
									'vgg_16/conv4/conv4_2/weights':'FNet/conv4_2/W',
									'vgg_16/conv4/conv4_2/biases':'FNet/conv4_2/b',	
									'vgg_16/conv4/conv4_3/weights':'FNet/conv4_3/W',
									'vgg_16/conv4/conv4_3/biases':'FNet/conv4_3/b',	
									'vgg_16/conv5/conv5_1/weights':'FNet/conv5_1/W',
									'vgg_16/conv5/conv5_1/biases':'FNet/conv5_1/b',	
									'vgg_16/conv5/conv5_2/weights':'FNet/conv5_2/W',
									'vgg_16/conv5/conv5_2/biases':'FNet/conv5_2/b',	
									'vgg_16/conv5/conv5_3/weights':'FNet/conv5_3/W',
									'vgg_16/conv5/conv5_3/biases':'FNet/conv5_3/b'} 

	# basic layer 
	def _he_uniform(self, shape, regularizer=None, trainable=None, name=None):
		name = 'W' if name is None else name + '/W'
		kernel_size = np.prod(shape[:2])
		fan_in = shape[-2] * kernel_size
		s = np.sqrt(1. / fan_in)

		with tf.device('/GPU:' + GPU_ID):  # 更新设备指定语法
			w = tf.Variable(
				initial_value=tf.random.uniform(shape, minval=-s, maxval=s, dtype=self.dtype),
				name=name,
				trainable=trainable
			)
			if regularizer is not None:
				tf.keras.regularizers.register_regularizer(w, regularizer)
		return w

	def _constant(self, shape, value=0, regularizer=None, trainable=None, name=None):
		name = 'b' if name is None else name + '/b'
		with tf.device('/GPU:' + GPU_ID):
			b = tf.Variable(
				initial_value=tf.constant(value, shape=shape, dtype=self.dtype),
				name=name,
				trainable=trainable
			)
			if regularizer is not None:
				tf.keras.regularizers.register_regularizer(b, regularizer)
		return b

	def _conv2d(self, tensor, dim, size=3, stride=1, rate=1, pad='SAME', act='relu', norm='none', G=16, bias=True,
				name='conv'):
		in_dim = tensor.shape[-1]
		size = size if isinstance(size, (tuple, list)) else [size, size]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		rate = rate if isinstance(rate, (tuple, list)) else [1, rate, rate, 1]
		kernel_shape = [size[0], size[1], in_dim, dim]

		w = self._he_uniform(kernel_shape, name=name)
		b = self._constant(dim, name=name) if bias else 0

		# 激活函数
		if act == 'relu':
			tensor = tf.nn.relu(tensor, name=name + '/relu')
		elif act == 'sigmoid':
			tensor = tf.nn.sigmoid(tensor, name=name + '/sigmoid')
		elif act == 'softplus':
			tensor = tf.nn.softplus(tensor, name=name + '/softplus')
		elif act == 'leaky_relu':
			tensor = tf.nn.leaky_relu(tensor, name=name + '/leaky_relu')

		# 归一化
		if norm == 'gn':
			x = tf.transpose(tensor, [0, 3, 1, 2])
			N, C, H, W = x.shape
			G = min(G, C)
			x = tf.reshape(x, [-1, G, C // G, H, W])
			mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
			x = (x - mean) / tf.sqrt(var + 1e-6)

			gamma = tf.Variable(tf.ones([C]), name=name + '/gamma')
			beta = tf.Variable(tf.zeros([C]), name=name + '/beta')
			gamma = tf.reshape(gamma, [1, C, 1, 1])
			beta = tf.reshape(beta, [1, C, 1, 1])

			tensor = tf.reshape(x, [-1, C, H, W]) * gamma + beta
			tensor = tf.transpose(tensor, [0, 2, 3, 1])

		# 卷积操作
		out = tf.nn.conv2d(tensor, w, strides=stride, padding=pad, dilations=rate) + b
		return out

	# 更新其他方法使用tf.Variable代替tf.get_variable
	def _upconv2d(self, tensor, dim, size=4, stride=2, pad='SAME', act='relu', name='upconv'):
		batch_size, h, w, in_dim = tensor.shape
		size = size if isinstance(size, (tuple, list)) else [size, size]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		kernel_shape = [size[0], size[1], dim, in_dim]

		W = self._he_uniform(kernel_shape, name=name)

		if pad == 'SAME':
			out_shape = [batch_size, h * stride[1], w * stride[2], dim]
		else:
			out_shape = [batch_size, (h - 1) * stride[1] + size[0],
						 (w - 1) * stride[2] + size[1], dim]

		out = tf.nn.conv2d_transpose(
			tensor, W, output_shape=out_shape,
			strides=stride, padding=pad, name=name
		)

		if act == 'relu':
			out = tf.nn.relu(out, name=name + '/relu')
		elif act == 'sigmoid':
			out = tf.nn.sigmoid(out, name=name + '/sigmoid')

		return out

	def forward(self, inputs, init_with_pretrain_vgg=False, pre_trained_model='./vgg16/vgg_16.ckpt'):
		# 使用tf.keras的VGG16作为基础网络
		if init_with_pretrain_vgg:
			vgg16 = VGG16(weights='imagenet', include_top=False)
			# 将VGG16权重映射到我们的网络
			for layer in vgg16.layers:
				if 'conv' in layer.name:
					# 实现权重映射逻辑
					pass
		with tf.variable_scope('FNet', reuse=reuse_fnet):
			# feature extraction
			self.conv1_1 = self._conv2d(inputs, dim=64, name='conv1_1') 
			self.conv1_2 = self._conv2d(self.conv1_1, dim=64, name='conv1_2')		
			self.pool1  = self._max_pool2d(self.conv1_2) # 256 => /2			

			self.conv2_1 = self._conv2d(self.pool1, dim=128, name='conv2_1')
			self.conv2_2 = self._conv2d(self.conv2_1, dim=128, name='conv2_2')	
			self.pool2 = self._max_pool2d(self.conv2_2) # 128 => /4		

			self.conv3_1 = self._conv2d(self.pool2, dim=256, name='conv3_1')
			self.conv3_2 = self._conv2d(self.conv3_1, dim=256, name='conv3_2')	
			self.conv3_3 = self._conv2d(self.conv3_2, dim=256, name='conv3_3')	
			self.pool3 = self._max_pool2d(self.conv3_3) # 64 => /8		

			self.conv4_1 = self._conv2d(self.pool3, dim=512, name='conv4_1')	
			self.conv4_2 = self._conv2d(self.conv4_1, dim=512, name='conv4_2')		
			self.conv4_3 = self._conv2d(self.conv4_2, dim=512, name='conv4_3')	
			self.pool4 = self._max_pool2d(self.conv4_3)	# 32 => /16		

			self.conv5_1 = self._conv2d(self.pool4, dim=512, name='conv5_1')	
			self.conv5_2 = self._conv2d(self.conv5_1, dim=512, name='conv5_2')		
			self.conv5_3 = self._conv2d(self.conv5_2, dim=512, name='conv5_3')		
			self.pool5 = self._max_pool2d(self.conv5_3)	# 16 => /32		

			# init feature extraction part from pre-train vgg16
			if init_with_pretrain_vgg:
				tf.train.init_from_checkpoint(pre_trained_model, self.pre_train_restore_map)

			# input size for logits predict
			[n, h, w, c] = inputs.shape.as_list()

		reuse_cw_net = len([v for v in tf.global_variables() if v.name.startswith('CWNet')]) > 0
		with tf.variable_scope('CWNet', reuse=reuse_cw_net):
			# upsample
			up2 = (self._upconv2d(self.pool5, dim=256, act='linear', name='up2_1') # 32 => /16
					+ self._conv2d(self.pool4, dim=256, act='linear', name='pool4_s'))
			self.up2_cw = self._conv2d(up2, dim=256, name='up2_3')

			up4 = (self._upconv2d(self.up2_cw, dim=128, act='linear', name='up4_1') # 64 => /8
					+ self._conv2d(self.pool3, dim=128, act='linear', name='pool3_s'))
			self.up4_cw = self._conv2d(up4, dim=128, name='up4_3')

			up8 = (self._upconv2d(self.up4_cw, dim=64, act='linear', name='up8_1') # 128 => /4
					+ self._conv2d(self.pool2, dim=64, act='linear', name='pool2_s'))
			self.up8_cw = self._conv2d(up8, dim=64, name='up8_2')

			up16 = (self._upconv2d(self.up8_cw, dim=32, act='linear', name='up16_1') # 256 => /2
					+ self._conv2d(self.pool1, dim=32, act='linear', name='pool1_s'))
			self.up16_cw = self._conv2d(up16, dim=32, name='up16_2')

			# predict logits
			logits_cw = self._up_bilinear(self.up16_cw, dim=3, shape=(h, w), name='logits')	

		# decode network for room type detection
		reuse_rnet = len([v for v in tf.global_variables() if v.name.startswith('RNet')]) > 0
		with tf.variable_scope('RNet', reuse=reuse_rnet):
			# upsample
			up2 = (self._upconv2d(self.pool5, dim=256, act='linear', name='up2_1') # 32 => /16
					+ self._conv2d(self.pool4, dim=256, act='linear', name='pool4_s'))
			up2 = self._conv2d(up2, dim=256, name='up2_2')
			up2, _ = self._non_local_context(self.up2_cw, up2, name='context_up2')

			up4 = (self._upconv2d(up2, dim=128, act='linear', name='up4_1') # 64 => /8
					+ self._conv2d(self.pool3, dim=128, act='linear', name='pool3_s'))
			up4 = self._conv2d(up4, dim=128, name='up4_2')
			up4, _ = self._non_local_context(self.up4_cw, up4, name='context_up4')

			up8 = (self._upconv2d(up4, dim=64, act='linear', name='up8_1') # 128 => /4
					+ self._conv2d(self.pool2, dim=64, act='linear', name='pool2_s'))
			up8 = self._conv2d(up8, dim=64, name='up8_2')
			up8, _ = self._non_local_context(self.up8_cw, up8, name='context_up8')

			up16 = (self._upconv2d(up8, dim=32, act='linear', name='up16_1') # 256 => /2
					+ self._conv2d(self.pool1, dim=32, act='linear', name='pool1_s'))
			up16 = self._conv2d(up16, dim=32, name='up16_2')
			self.up16_r, self.a = self._non_local_context(self.up16_cw, up16, name='context_up16')

			# predict logits
			logits_r = self._up_bilinear(self.up16_r, dim=9, shape=(h, w), name='logits')	

			return logits_r, logits_cw	