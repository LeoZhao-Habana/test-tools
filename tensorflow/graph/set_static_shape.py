import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import dtypes
import numpy as np

def find_node(graph_def, name):
    node = None
    for n in graph_def.node:
        if n.name == name:
           node = n
           break
    if node == None:
        raise ValueError(f"Node {name} not found)")

    return node

def create_shape_proto(shape):
    shape_proto = tensor_shape_pb2.TensorShapeProto()
    for dim in shape:
        shape_proto.dim.add().size = dim
    return attr_value_pb2.AttrValue(shape=shape_proto)

def set_shape(node, shape):
    node.attr["shape"].CopyFrom(create_shape_proto(shape))

def load_frozen_graph(fname):
    with tf.gfile.GFile(fname, "rb") as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
    return graph_def

def write_graph(graph_def, fname, as_text=False):
    tf.train.write_graph(graph_def, ".", fname, as_text=as_text)

# load model save frozen shape
fname = "/software/data/tf/models/faster_rcnn_resnet101_coco_2018_01_28_dynamic.pb"
graph_def = load_frozen_graph(fname)

# Set the placeholder to a static shape
node = find_node(graph_def, "image_tensor")
set_shape(node, [1, 600, 800, 3])

# Save the frozen graph
write_graph(graph_def, "./faster_rcnn_resnet101_coco_2018_01_28_1x600x800x3.pb")
