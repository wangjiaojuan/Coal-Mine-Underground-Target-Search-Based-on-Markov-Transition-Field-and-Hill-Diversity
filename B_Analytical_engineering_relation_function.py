import math
import numpy as np
import os
graphviz_bin_path = r'D:\ruanjian\Graphviz\bin'  
os.environ["PATH"] += os.pathsep + graphviz_bin_path
import cv2
from scipy import stats
import networkx as nx
#import matplotlib.pyplot as plt
from PIL import Image
import io
def padding_img(img):
    if len(img.shape)==2:
        w,h = img.shape
        c=3
        new_w,new_h,new_c = (math.floor(w/3)+1)*3,(math.floor(h/3)+1)*3,c
        new_img = np.zeros((new_w,new_h,new_c), dtype=np.uint8)
        new_img[new_w-w:,new_h-h:,0] = img
        new_img[new_w-w:,new_h-h:,1] = img
        new_img[new_w-w:,new_h-h:,2] = img
    else:
        w,h,c = img.shape
        new_w,new_h,new_c = (math.floor(w/3)+1)*3,(math.floor(h/3)+1)*3,c
        new_img = np.zeros((new_w,new_h,new_c), dtype=np.uint8)
        new_img[new_w-w:,new_h-h:,:] = img
    return new_img
def split_img(img):
    h,w,_ = img.shape
    Patch_1 = img
    Patch_2,Patch_3 = img[:int(h/2),:,:],img[int(h/2):,:,:]
    Patch_4,Patch_5,Patch_6,Patch_7 = img[:int(h/2),:int(w/2),:],img[:int(h/2),int(w/2):,:],img[int(h/2):,:int(w/2),:],img[int(h/2):,int(w/2):,:]
    Patch_8,Patch_9,Patch_10 = img[:int(h/3),:int(w/3),:],img[:int(h/3),int(w/3):int(2*w/3),:],img[:int(h/3),int(2*w/3):int(w),:]
    Patch_11,Patch_12,Patch_13 = img[int(h/3):int(2*h/3),:int(w/3),:],img[int(h/3):int(2*h/3),int(w/3):int(2*w/3),:],img[int(h/3):int(2*h/3),int(2*w/3):int(w),:]
    Patch_14,Patch_15,Patch_16 = img[int(2*h/3):int(h),:int(w/3),:],img[int(2*h/3):int(h),int(w/3):int(2*w/3),:],img[int(2*h/3):int(h),int(2*w/3):int(w),:]
    return Patch_1,Patch_2,Patch_3,Patch_4,Patch_5,Patch_6,Patch_7,Patch_8,Patch_9,Patch_10,Patch_11,Patch_12,Patch_13,Patch_14,Patch_15,Patch_16
def patch_Statistics_value(patch):
    data = patch.flatten()
    # 计算均值
    s1 = np.mean(data)
    # 计算方差
    s2 = np.var(data)
    # 计算标准差
    s3 = np.std(data)
    # 计算众数
    mode_result = stats.mode(data)
    s4 = int(mode_result.mode)
    # 计算中位数
    s5 = np.median(data)
    # 计算偏度
    s6 = stats.skew(data)
    # 计算峰度
    s7 = stats.kurtosis(data)
    # 计算极差
    s8 = np.max(data) - np.min(data)
    return [s1,s2,s3,s4,s5,s6,s7,s8]
def Statistics_value(Patch_1,Patch_2,Patch_3,Patch_4,Patch_5,Patch_6,Patch_7,Patch_8,Patch_9,Patch_10,Patch_11,Patch_12,Patch_13,Patch_14,Patch_15,Patch_16):
    
    Statistics_value_Patch_1 = patch_Statistics_value(Patch_1)
    Statistics_value_Patch_2 = patch_Statistics_value(Patch_2)
    Statistics_value_Patch_3 = patch_Statistics_value(Patch_3)
    Statistics_value_Patch_4 = patch_Statistics_value(Patch_4)
    Statistics_value_Patch_5 = patch_Statistics_value(Patch_5)
    Statistics_value_Patch_6 = patch_Statistics_value(Patch_6)
    Statistics_value_Patch_7 = patch_Statistics_value(Patch_7)
    Statistics_value_Patch_8 = patch_Statistics_value(Patch_8)
    Statistics_value_Patch_9 = patch_Statistics_value(Patch_9)
    Statistics_value_Patch_10 = patch_Statistics_value(Patch_10)  
    Statistics_value_Patch_11 = patch_Statistics_value(Patch_11)
    Statistics_value_Patch_12 = patch_Statistics_value(Patch_12)
    Statistics_value_Patch_13 = patch_Statistics_value(Patch_13)
    Statistics_value_Patch_14 = patch_Statistics_value(Patch_14)
    Statistics_value_Patch_15 = patch_Statistics_value(Patch_15)
    Statistics_value_Patch_16 = patch_Statistics_value(Patch_16)

    SS = Statistics_value_Patch_1
    SS.extend(Statistics_value_Patch_2)
    SS.extend(Statistics_value_Patch_3)
    SS.extend(Statistics_value_Patch_4)
    SS.extend(Statistics_value_Patch_5)
    SS.extend(Statistics_value_Patch_6)
    SS.extend(Statistics_value_Patch_7)
    SS.extend(Statistics_value_Patch_8)
    SS.extend(Statistics_value_Patch_9)
    SS.extend(Statistics_value_Patch_10)
    SS.extend(Statistics_value_Patch_11)
    SS.extend(Statistics_value_Patch_12)
    SS.extend(Statistics_value_Patch_13)
    SS.extend(Statistics_value_Patch_14)
    SS.extend(Statistics_value_Patch_15)
    SS.extend(Statistics_value_Patch_16)
    return np.array(SS)

def mkdir_multi(path):
    # 判断路径是否存在
    isExists=os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path) 
        print('目录创建成功！')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('目录已存在！')
        return False

def get_high_low_frequency_images(image, radius=30):
    # 分离彩色图像的三个通道
    b, g, r = cv2.split(image)

    # 定义存储高频和低频图像通道的列表
    low_freq_channels = []
    high_freq_channels = []

    for channel in [b, g, r]:
        # 进行傅里叶变换
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        # 获取图像的行数和列数
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # 低通滤波器
        mask_low = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask_low, (ccol, crow), radius, 1, -1)

        # 高通滤波器
        mask_high = 1 - mask_low

        # 应用低通滤波器
        fshift_low = fshift * mask_low
        f_ishift_low = np.fft.ifftshift(fshift_low)
        channel_low = np.fft.ifft2(f_ishift_low)
        channel_low = np.abs(channel_low)

        # 应用高通滤波器
        fshift_high = fshift * mask_high
        f_ishift_high = np.fft.ifftshift(fshift_high)
        channel_high = np.fft.ifft2(f_ishift_high)
        channel_high = np.abs(channel_high)

        # 将处理后的通道添加到相应列表
        low_freq_channels.append(channel_low)
        high_freq_channels.append(channel_high)

    # 合并通道
    low_freq_image = cv2.merge(low_freq_channels).astype(np.uint8)
    high_freq_image = cv2.merge(high_freq_channels).astype(np.uint8)

    return low_freq_image, high_freq_image

def generate_markov_graph(data):
    # 确定状态集合
    states = np.unique(data)
    num_states = len(states)
    # 初始化状态转移矩阵
    transition_matrix = np.zeros((num_states, num_states))

    # 计算状态转移次数
    for i in range(len(data) - 1):
        current_state = data[i]
        next_state = data[i + 1]
        current_index = np.where(states == current_state)[0][0]
        next_index = np.where(states == next_state)[0][0]
        transition_matrix[current_index][next_index] += 1

    # 归一化状态转移矩阵
    for i in range(num_states):
        row_sum = np.sum(transition_matrix[i])
        if row_sum > 0:
            transition_matrix[i] /= row_sum

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点
    for state in states:
        G.add_node(state)

    # 添加边
    for i in range(num_states):
        for j in range(num_states):
            if transition_matrix[i][j] > 0:
                G.add_edge(states[i], states[j], weight=str(transition_matrix[i][j]))

    return G, transition_matrix

def get_markovImg(imge_descriptor,img_path):
    # 生成马尔可夫图和状态转移矩阵
    G, transition_matrix = generate_markov_graph(imge_descriptor)

    # 使用 graphviz 渲染图形
    p = nx.drawing.nx_pydot.to_pydot(G)
    img_data = p.create_png()

    # 将图像数据转换为 PIL 图像
    img_pil = Image.open(io.BytesIO(img_data))

    # 将 PIL 图像转换为 cv2 图像
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 保存图像
    #cv2.imwrite('markov_graph.png', image)

    return image


def get_MarkovTransitionFieldImg(imge_descriptor,img_path):
    # 导入必要的库
    from pyts.image import MarkovTransitionField
    import cv2

    # 初始化 MTF 转换器，设置图像大小为 24x24
    mtf = MarkovTransitionField(image_size=24)

    # 将 1 维数据转换为 MTF 图像
    # 注意：X 需要是 2D 数组，形状为 (n_samples, n_timesteps)
    X_mtf = mtf.fit_transform(imge_descriptor.reshape(1, -1))  # 将 1D 数据转换为 2D

    # 将 MTF 图像归一化到 [0, 255] 范围，并转换为 uint8 类型
    mtf_image = (X_mtf[0] * 255).astype(np.uint8)

    # 将灰度图转换为彩色图（伪彩色）
    # 使用 OpenCV 的 applyColorMap 函数
    colored_image = cv2.applyColorMap(mtf_image, cv2.COLORMAP_JET)

    # 使用 cv2.resize() 调整图像大小
    # 将图像从 24x24 放大到 224*224
    resized_image = cv2.resize(colored_image, (100, 100), interpolation=cv2.INTER_LINEAR)

    return resized_image