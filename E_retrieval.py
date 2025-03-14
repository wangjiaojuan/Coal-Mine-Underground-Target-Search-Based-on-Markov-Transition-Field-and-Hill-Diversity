from __future__ import division 
from E_retrieval_relation_function import*
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")

def split_retrieval_datasets(Datasets_name):
    # get all_image label
    filepath = Datasets_name+'_paths.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get retrieval image and database image
    if 'Holidays' in Datasets_name:  
        path=0
        re=[] 
        database=[]
        while path<origion_line:
            re.append(path)
            name_map=ac_img_label(origion_file)
            re_image = origion_file[path].split('\\')[1] #检索图像的类名
            start,end = name_map[re_image]
            num=end-start+1
            start=start+1
            while start<=end:
              database.append(start)
              start=start+1
            path=path+num
    else:
        path=0
        re=[] 
        database=[]
        while path<origion_line:
            re.append(path)
            database.append(path)
            path=path+1
    return re,database,origion_file
def compare(Datasets_name,Top_N,pre_name,after_name):
    file_path = './results/'+Datasets_name+"_"+pre_name.split('/')[1]+"_"+after_name.split('/')[1]+".txt"
    filewrite = open(file_path, "w")
    Result = ['all_feature_shuffleNetV2','all_feature_vgg16','all_feature_resnet18','all_feature_densenet121',
              'all_feature_Alexnet','all_feature_convnext','all_feature_squeezenet',
              'all_feature_googlenet','all_feature_efficientnet','all_feature_mnasnet',
              'all_feature_regnet','all_feature_mobilenet']#'all_feature_HOG',,'all_feature_color','all_feature_lbp','all_feature_GIST','all_feature_Harris'
    
    for feature_name in Result:
        print("........................................comparision is",feature_name)
        print("........................................comparision is",feature_name,file=filewrite)
        # get feature
        pre_feature = np.load(pre_name+Datasets_name.split('_')[0]+'_feature_'+feature_name.split('_')[2]+'.npy')
        after_feature = np.load(after_name+Datasets_name.split('_')[0]+'_feature_'+feature_name.split('_')[2]+'.npy')
        re,database,origion_file = split_retrieval_datasets(Datasets_name)
        pre_ac=0
        after_ac=0
        pre_time=0
        after_time=0
        #########################
        for i in range(len(re)):
            path=re[i]
            #获得本次检索的正确结果的索引
            name_map=ac_img_label(origion_file)
            re_image = origion_file[path].split('\\')[1]
            start,end = name_map[re_image]
            ac_label=[]
            start=start+1
            while start<=end:
              ac_label.append(start)
              start=start+1 
            ##retrieval
            time1 = time.time()
            pre_distance=euclidean_distance(pre_feature,path,database)
            result_pre=query_result(pre_distance,ac_label,database,Datasets_name,Top_N)
            time2 = time.time()
            after_distance=haming_distance(after_feature,path,database)
            result_after=query_result(after_distance,ac_label,database,Datasets_name,Top_N)
            time3 = time.time()

            pre_time = pre_time + (time2 - time1)
            after_time = after_time + (time3-time2)
            pre_ac = pre_ac+result_pre
            after_ac = after_ac+result_after

        pre_time = pre_time/len(re)
        after_time = after_time/len(re)
        pre_ac = pre_ac/len(re)
        after_ac = after_ac/len(re)

        print("pre_ac,after_ac.................is",pre_ac,after_ac)
        print("pre_time,after_time.................is",pre_time,after_time)

        print("pre_ac,after_ac.................is",pre_ac,after_ac,file=filewrite)
        print("pre_time,after_time.................is",pre_time,after_time,file=filewrite)

if __name__ == '__main__':
    pre_name = './preAnalytical_features/'
    after_name = './pre_afterAnalytical_features/'
    for Datasets_name in ['mydataset2']:#'CUMT-BelT','mydataset'
        for Top_N in [20]:
            print('Top_N',Top_N,Datasets_name,pre_name,after_name)
            compare(Datasets_name,Top_N,pre_name,after_name)