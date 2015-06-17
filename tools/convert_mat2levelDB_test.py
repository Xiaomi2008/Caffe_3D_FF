import sys, getopt

sys.path.insert(0, '/home/tzeng/caffe_3d/python')
sys.path.insert(0,'/home/tzeng/autoGenelable_multi_lables_proj/code/py-leveldb-read-only/build/lib.linux-x86_64-2.7')

import numpy as np
import hdf5storage
import leveldb
from leveldb import WriteBatch, LevelDB
import os
import imp
#import caffe
#sys.path.append('/home/tzeng/caffe_3d/python/caffe')
#foo = imp.load_source('caffe.io', '/home/tzeng/caffe_3d/python/caffe/__init__.py')
import caffe.io
from caffe.proto import caffe_pb2
print os.path.dirname(caffe_pb2.__file__) 
#from caffe.proto import caffe_pb2
#mat_file ='/home/tzeng/caffe_3d/data/mri_test_pad'
mat_file ='/home/tzeng/caffe_3d/data/test'
out =hdf5storage.loadmat(mat_file,format='7.3')
#print len(out)
#print len(out['data'])
#print out['data'].shape
#print out['data'][0,1].shape
size = out['A'].shape;
size=size[1];
print size
k=1
db_path_data='/home/tzeng/caffe_3d/data/test'
db_data_lb=leveldb.LevelDB(db_path_data, create_if_missing=True, error_if_exists=False)	
batch = leveldb.WriteBatch()
D=out['A']
print D[:,:,0]

D2=out['B']
print D2[:,:,0]


dim_3d=D.shape 
print(dim_3d)
dim_4d=[1]
for i in (dim_3d):
  dim_4d.append(i)

d=np.reshape(D,dim_4d).astype('uint8')
elm_d=np.reshape(D2,dim_4d).astype('float')
datum= caffe.io.elemetwise_array_to_datum(d,elm_d)
db_data_lb.Put('%08d' % 1, datum.SerializeToString())
print d
print elm_d
db_data_lb.Write(batch)

	



# for k in range(size):

 # p =out['data'][0,k]
 # l =out['labels'][0,k]
 # elm_l=out['elm_labels'][0,k]
 
 # #print p

 # dim_3d=p.shape
 # print(dim_3d)
 # dim_4d=[1]
 # #print p[:,32,20]
 # for i in (dim_3d):
  # dim_4d.append(i)

 # d=np.reshape(p,dim_4d).astype('uint8')
 # print " max =%d  min =%d" %(d.max(), d.min())
 # elm_d=np.reshape(elm_l,dim_4d).astype('float')
 # print d.shape
 # labels=[l.astype(int).tolist()]
 # #print type(labels[0])
 # #print labels
 # #datum= caffe.io.array_to_datum(d,labels)
 # datum=caffe.io.elemetwise_array_to_datum(d, elm_d)
 # db_data_lb.Put('%08d' % k, datum.SerializeToString())
# #datum = caffe_pb2.Datum()
# #print datum.label.extend([1,2,3,4,5])
 # #print datum.label;
 # print(k)
# db_data_lb.Write(batch)
#datum.clear_float_data();