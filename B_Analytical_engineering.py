from B_Analytical_engineering_relation_function import*
import cv2


def deal_img(img_path):
    #img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    if img is None:
        print("无法读取图像，请检查图像路径。")
    else:
        # 转换颜色空间以用于 matplotlib 显示
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 获取高频和低频图像
    low_freq_image, high_freq_image = get_high_low_frequency_images(img)
    new_img = padding_img(low_freq_image)
    Patch_1,Patch_2,Patch_3,Patch_4,Patch_5,Patch_6,Patch_7,Patch_8,Patch_9,Patch_10,Patch_11,Patch_12,Patch_13,Patch_14,Patch_15,Patch_16 = split_img(new_img)
    imge_descriptor = Statistics_value(Patch_1,Patch_2,Patch_3,Patch_4,Patch_5,Patch_6,Patch_7,Patch_8,Patch_9,Patch_10,Patch_11,Patch_12,Patch_13,Patch_14,Patch_15,Patch_16)
    markov_image = get_MarkovTransitionFieldImg(imge_descriptor,img_path)
    
    low_freq_image_path = img_path.replace("Datasets","low_freq_image")
    # 提取目录路径
    directory = os.path.dirname(low_freq_image_path)
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    high_freq_image_path = img_path.replace("Datasets","high_freq_image")
    directory = os.path.dirname(high_freq_image_path)
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(low_freq_image_path,low_freq_image)
    cv2.imwrite(high_freq_image_path,high_freq_image)
    
    markov_image_path = img_path.replace("Datasets","markov_image")
    directory = os.path.dirname(markov_image_path)
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(markov_image_path,markov_image)
    return low_freq_image,high_freq_image,markov_image

def compare(Datasets_name):
    # get all_image label
    filepath = Datasets_name+'_paths.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    path=0
    while path<origion_line:
        img_path = origion_file[path][:-1]
        path=path+1
        deal_img(img_path)
        print(path,origion_line)


if __name__ == '__main__':
    Datasets_name = 'mydataset2'#'mydataset' #'CUMT-BelT'
    compare(Datasets_name)

