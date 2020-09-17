import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import numpy as np
import pickle

if __name__ == "__main__":
    GRAPH_PB_PATH = 'final_frozen_graph.pb' 
    GRAPH_PB_PATH = 'samba_bert.pb'

    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        wts = [n for n in graph_nodes if n.op=='Const']
    data = {}
    for n in wts:
        print ("Name of the node - %s" % n.name)
        print ("Value - ")
        print (tensor_util.MakeNdarray(n.attr['value'].tensor))
        data[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)
    f = open("weight_data.pkl","wb")
    pickle.dump(data,f)
    f.close()

    # with open("weight_data.pkl","rb") as f:
    #     weight_data = pickle.load(f)
    # W = weight_data['gpu/transform2_block/multi_head_attention/conv_1d/he_uniform/W']
    # value = (W[:,:,:512]).reshape((-1,512))
    # # value.reshape((-1,512))
    # print(np.shape(value))