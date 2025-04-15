import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import imageio.v2 as imageio
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少日志输出

# 输入输出路径配置
parser = argparse.ArgumentParser(description='Floorplan Prediction Demo')
parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='Path to input image')
parser.add_argument('--model_dir', type=str, default='./pretrained',
                    help='Directory containing pretrained model files')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory to save output images')

# 颜色映射
floorplan_map = {
    0: [255,255,255], # background
    1: [192,192,224], # closet
    2: [192,255,255], # batchroom/washroom
    3: [224,255,192], # livingroom/kitchen/dining room
    4: [255,224,128], # bedroom
    5: [255,160, 96], # hall
    6: [255,224,224], # balcony
    7: [255,255,255], # not used
    8: [255,255,255], # not used
    9: [255, 60,128], # door & window
    10:[  0,  0,  0]  # wall
}

def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def imresize(arr, size):
    """图像缩放"""
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        channels = []
        for i in range(3):
            channel = Image.fromarray(arr[:, :, i].astype(np.uint8))
            channel = channel.resize((size[0], size[1]), Image.BILINEAR)
            channels.append(np.array(channel))
        return np.stack(channels, axis=2)
    else:
        img = Image.fromarray(arr.astype(np.uint8))
        return np.array(img.resize((size[0], size[1]), Image.BILINEAR))

def ind2rgb(ind_im, color_map=floorplan_map):
    """将索引图像转换为RGB"""
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))
    
    for i, rgb in color_map.items():
        rgb_im[(ind_im==i)] = rgb
    
    return rgb_im

def load_model(sess, model_dir):
    """加载预训练模型"""
    try:
        meta_path = os.path.join(model_dir, 'pretrained_r3d.meta')
        model_path = os.path.join(model_dir, 'pretrained_r3d')
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model meta file not found at {meta_path}")
        if not os.path.exists(model_path + '.index'):
            raise FileNotFoundError(f"Model data files not found at {model_path}")
        
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, model_path)
        return saver
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def save_results(input_im, prediction, output_dir, base_filename):
    """保存输入图像和预测结果"""
    ensure_dir(output_dir)
    
    # 保存输入图像
    input_path = os.path.join(output_dir, f"{base_filename}_input.png")
    imageio.imwrite(input_path, (input_im * 255).astype(np.uint8))
    
    # 保存预测结果
    pred_path = os.path.join(output_dir, f"{base_filename}_prediction.png")
    imageio.imwrite(pred_path, prediction.astype(np.uint8))
    
    # 保存对比图
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Input Image')
    plt.imshow(input_im)
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Floorplan Prediction')
    plt.imshow(prediction/255.)
    plt.axis('off')
    
    plt.tight_layout()
    compare_path = os.path.join(output_dir, f"{base_filename}_compare.png")
    plt.savefig(compare_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    print(f"Results saved to: {input_path}, {pred_path}, {compare_path}")

def main(args):
    """主处理函数"""
    # 验证输入路径
    if not os.path.exists(args.im_path):
        raise FileNotFoundError(f"Input image not found at {args.im_path}")

    # 加载并预处理图像
    try:
        im = imageio.imread(args.im_path, mode='RGB')
        im = im.astype(np.float32)
        im = imresize(im, (512, 512)) / 255.
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

    # 创建并运行TensorFlow会话
    with tf.Session() as sess:
        try:
            # 初始化变量
            sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

            # 加载模型
            load_model(sess, args.model_dir)

            # 获取输入输出张量
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name('inputs:0')
            room_type_logit = graph.get_tensor_by_name('Cast:0')
            room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

            # 执行推理
            [room_type, room_boundary] = sess.run(
                [room_type_logit, room_boundary_logit],
                feed_dict={x: im.reshape(1, 512, 512, 3)}
            )
            room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

            # 合并结果
            floorplan = room_type.copy()
            floorplan[room_boundary==1] = 9
            floorplan[room_boundary==2] = 10
            floorplan_rgb = ind2rgb(floorplan)

            # 保存结果
            base_name = os.path.splitext(os.path.basename(args.im_path))[0]
            save_results(im, floorplan_rgb, args.output_dir, base_name)

        except tf.errors.OpError as e:
            raise RuntimeError(f"TensorFlow operation error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    try:
        main(FLAGS)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting steps:")
        print("1. Ensure the input image exists")
        print("2. Verify pretrained model files are in correct directory")
        print("3. Check file permissions")
        print("4. Ensure output directory is writable")