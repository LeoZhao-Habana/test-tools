
from google.protobuf import text_format
from tensorflow.core import framework
from tensorflow.python.framework import dtypes
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_shape_pb2

import tensorflow as tf
import os
import sys
import shutil
import argparse
import traceback
import numpy as np
import warnings

from auto_recipe_gen import graph_common as gt
from auto_recipe_gen.einsum_util import *

class GenericOptimizor(object):
    def __init__(self, graph, input_names, output_names, bs, input_shape):
        self.graph_path = graph
        self.graph_def = self.load_frozen_graph()
        self.batch_size = bs
        self.input_names = input_names
        self.output_names = output_names
        self.input_shape = input_shape

    def load_frozen_graph(self):
        with tf.gfile.GFile(self.graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def find_node_by_name(self, name):
        node = None
        node_list=[n.name for n in self.graph_def.node]
        for n in self.graph_def.node:
            if n.name == name:
                node = n
                break
        if node == None:
            raise ValueError(f"Node {name} not found)")
        return node


    def find_node_by_type(self, type):
        node = []
        for n in self.graph_def.node:
            if n.op == type:
                node.append(n)
        if node == None:
            raise ValueError(f"Node {type} not found)")
        return node

    def GeneralOptimize(self):
        print("[Info] Doing general optimization...")

        input_names = self.input_names
        output_names = self.output_names
        graph_d = self.graph_def

        transforms = [
            "remove_nodes(op=Identity, op=StopGradient, op=Assert)",
            "fold_batch_norms",
            "fold_old_batch_norms",
            "fuse_pad_and_conv",
            "strip_unused_nodes",
        ]

        self.graph_def = TransformGraph(
            graph_d, input_names, output_names, transforms)

        return self.graph_def

    def StripUnusedNodes(self):
        print("[Info] Striping unsed nodes...")

        input_names = self.input_names
        output_names = self.output_names
        graph_d = self.graph_def

        transforms = [
            "strip_unused_nodes",
        ]

        self.graph_def = TransformGraph(
            graph_d, input_names, output_names, transforms)

        return self.graph_def

    def Setshape2Frozen(self):
        for name in self.input_names:
            name, _ = name.split(":")
            node = self.find_node_by_name(name)
            shape_proto = tensor_shape_pb2.TensorShapeProto()
            for dim in self.input_shape:
                shape_proto.dim.add().size = dim
            new_shape = attr_value_pb2.AttrValue(shape=shape_proto)
            node.attr["shape"].CopyFrom(new_shape)

        abs_path = os.path.abspath(self.graph_path)
        filepath, filename = os.path.split(abs_path)
        opt_frozen = str(self.batch_size) + "_" + filename
        opt_frozen_path = os.getcwd()

        tf.train.write_graph(self.graph_def, opt_frozen_path, opt_frozen, as_text=False)
        self.graph_path = os.path.join(opt_frozen_path, opt_frozen)
        self.graph_def = self.load_frozen_graph()
        return self.graph_def

    def EinsumReduction(self):
        while True:
            nodes = self.find_node_by_type("Einsum")
            if len(nodes) == 0:
                break
            else:
                node=nodes[0]
                tf.reset_default_graph()
                tf.import_graph_def(self.graph_def, name="")
                tensor = einsum_op(node)
                self.graph_def = tf.get_default_graph().as_graph_def()
                next_node = gt.getNodesInOutput(self.graph_def, node.name)
                for n in next_node:
                    next_op = self.find_node_by_name(n[1])
                    next_op.input[n[0]], _ = (tensor.name).split(":")
                self.StripUnusedNodes()

        return self.graph_def


    def run_pass(self):
        self.GeneralOptimize()
        self.EinsumReduction()
        self.Setshape2Frozen()

class BERTOptimizor(GenericOptimizor):
    pass

class WaveGlowOpimizor(GenericOptimizor):
    pass
