import numpy as np
np.set_printoptions(threshold='nan')
import matplotlib.pyplot as plt
#matplotlib inline

#input five parameters
####first three########### 
vgg_model = 16
index_layer = 15
stage = 6
#############################
# Make sure that caffe is on the python path:
caffe_root = '/home/tzeng/caffe/'  # this file is expected to be in {caffe_root}/examples
save_root = '/home/tzeng/overfit_Fruit_Fly_qiansun/vgg/vgg_' + str(vgg_model) + '/layer' + str(index_layer) + '/others/' + str(stage) + '/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import os, sys

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
net = caffe.Classifier(caffe_root + 'models/VGG_ILSVRC_' + str(vgg_model) + '_layers/deploy.prototxt',
                       caffe_root + 'models/VGG_ILSVRC_' + str(vgg_model) + '_layers/VGG_ILSVRC_' + str(vgg_model) + '_layers.caffemodel')
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

#next two parameters
###########################
feature_map = 512
feature_matrix_size = 14
###########################
path = '/home/tzeng/overfit_Fruit_Fly_qiansun/bmps_224_224_resize/img_' + str(stage) +'/'
dirs = os.listdir( path )

# This would print all the files and directories
for file in dirs:
    print file
    temp1 = os.path.join(path,file)
    scores = net.predict([caffe.io.load_image(temp1)])
    out_file1 = open(save_root + str('feature_layer')+ '_'+ file + '.txt', 'w+')
    temp = net.blobs.items()
    a = temp[index_layer]
    #print (a[1].data)
    temp2 = a[1].data
    temp3 = temp2.reshape(1*feature_map*feature_matrix_size*feature_matrix_size)
    #out_file2.write(str(temp3)+'\n') 
    for i in range(0,len(temp3)):
        out_file1.write(str(temp3[i])+'\n')
# print (net.blobs.items())
# for k, v in net.blobs.items():
    # print (k, v.data.shape)



#out_file2 = open('feature_layer.txt', 'w+') 





 

