#!/usr/bin/env python
#
# Copyright 2020-2021 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow.core import framework

def isTextProtobuf(filename):
    """ Returns whether a filename is a text protobuf based on the file extension.

    Args:
        filename: string - file name to process.

    Returns:
        true if `filename`'s extension is .pbtxt, false otherwise.
    """

    retval = False

    _, filename_ext = os.path.splitext(filename)
    if filename_ext and filename_ext.lower() == ".pbtxt":
        retval = True

    return retval


def saveGraphProtobufToFile(file_name, graph_d):
    """ Saves a `GraphDef` protocol buffer graph to a file.

    Args:
        file_name: string - name of the file where to write the graph.
        graph_d: The `GraphDef` protocol buffer to save.
    """
    output_file_name_no_dir = os.path.basename(file_name)
    output_file_dir = os.path.dirname(file_name)
    tf.io.write_graph(graph_d, output_file_dir,
                      output_file_name_no_dir, as_text=isTextProtobuf(file_name))


def loadGraphProtobufFromFile(file_name):
    """ Loads a `GraphDef` protocol buffer graph from a file.

    Args:
        file_name: string - name of the file to load.

    Returns:
        A `GraphDef` protocol buffer loaded from the file.
    """
    graph_d = framework.graph_pb2.GraphDef()
    with open(file_name, "rb") as f:
        if isTextProtobuf(file_name):
            # for text file:
            text_format.Merge(f.read(), graph_d)
        else:
            # for binary file:
            graph_d.ParseFromString(f.read())
    return graph_d


def duplicateGraph(graph_d):
    """ Creates a deep copy of a tf GraphDef.

    Args:
        graph_d: A `GraphDef` protocol buffer to duplicate.

    Returns:
        A deep copy of the specified tf GraphDef.
    """

    with tf.Graph().as_default() as tmp_graph:
        _ = tf.import_graph_def(graph_d, name="")
        return tmp_graph.as_graph_def()


def getNodeNames(nodes_d):
    """ Compiles a list of strings representing all the name of
    the nodes in the specified list of nodes.

    Args:
        nodes_d: List of `NodeDef` objects to process.

    Returns:
        A list of strings representing all the name of the nodes in `nodes_d`.
    """
    return [node_d.name for node_d in nodes_d]


def getNodeIndexByName(nodes_d, node_name):
    """ Finds the NodeDef node in list of NodeDef corresponding to
    the specified name.

    Args:
        nodes_d: List of `NodeDef` objects to process.
        node_name: node to find.

    Returns:
        And integer index representing the index of the node in the list
        passed or -1 if not found.
    """

    retval = -1
    for i, node_d in enumerate(nodes_d):
        if node_d.name == node_name:
            retval = i
            break
    return retval


def getNodeInputNamesClean(node_input_names):
    retval = []
    for input_name in node_input_names:
        tensor_idx = input_name.rfind(":")
        if tensor_idx < 0:
            retval.append(input_name)
        else:
            retval.append(input_name[:tensor_idx])
    return retval


def getNodeByName(nodes_d, node_name):
    """ Finds the NodeDef node in list of NodeDef corresponding to
    the specified name.

    Args:
        nodes_d: List of `NodeDef` objects to process.
        node_name: node to find.

    Returns:
        The `NodeDef` node in `nodes_d` corresponding to the specified name,
        or None if name is not found in `nodes_d`.
    """

    retval = getNodeIndexByName(nodes_d, node_name)
    if (retval < 0):
        retval = None
    else:
        retval = nodes_d[retval]
    return retval


def getInputNodeNames(graph_d):
    """ Finds the placeholder nodes (or inputs) in the graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.

    Returns:
        A list of node names corresponding to all nodes that are
        inputs to the graph.
    """

    retval = []
    for node_d in graph_d.node:
        if node_d.op == "Placeholder":
            retval.append(node_d.name)
    return retval


def getOutputNodeNames(graph_d):
    """ Finds the nodes that are leaf nodes (or outputs) in the graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.

    Returns:
        A list of node names corresponding to all nodes that are
        leaf nodes (or outputs) in the graph.
    """

    non_output_node_names = set()
    for node_d in graph_d.node:
        non_output_node_names = non_output_node_names | set(
            getNodeInputNamesClean(node_d.input))
    graph_node_names = set(getNodeNames(graph_d.node))
    return list(graph_node_names - non_output_node_names)


def getNodesInOutput(graph_d, node_name):
    """ Finds all nodes that use the output of specified node as
    their input in the specified graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.
        node_name: String name of node to check.

    Returns:
        A list of node names corresponding to all nodes that use the
        output of specified node as their input.
    """
    retval = []

    for node_d in graph_d.node:
        node_input_names = getNodeInputNamesClean(node_d.input)
        for id, input_name in enumerate(node_input_names):
            if input_name == node_name:
                retval.append([id,node_d.name])
                break

    return retval


def getNodesInSubGraph(graph_d, start_nodes, end_nodes):
    subgraph = []
    for node in start_nodes:
        subgraph.append(node)

    successor = start_nodes
    while len(successor) != 0:
        for node in successor:
            tmp_suc = getNodesInOutput(graph_d, node)
            for suc in tmp_suc:
                if suc in subgraph:
                    continue
                else:
                    subgraph.append(suc)
        successor = tmp_suc

    return subgraph


def isQDQNode(node_d):
    """ Returns whether a node is QDQ or not.

    Args:
        node_d: A `NodeDef` protocol buffer to process.

    Returns:
        A list of all qdq NodeDefs in the `graph_d`.
    """
    return "QuantizeAndDequantize" in node_d.op


def findQDQNodes(graph_d):
    """ Finds all QDQ nodes in the specified GraphDef.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.

    Returns:
        A list of all qdq NodeDefs in the `graph_d`.
    """

    # find all qdq nodes
    qdqs = []
    for node_d in graph_d.node:
        if isQDQNode(node_d):
            qdqs.append(node_d)

    return qdqs


def pruneLeafSubgraphs(graph_d, preserve_leaf_names):
    """ Removes all branches of the graph that end in a leaf not
    listed in preserve_leaf_names.

    Args:
        graph_d: A `GraphDef` protocol buffer to prune.
        preserve_leaf_names: List of String representing the name of the
            leaves that will not be pruned.

    Returns:
        A `GraphDef` equivalent to `graph_d`, but with subgraphs ending
        in leaves not specified in `preserve_leaf_names` removed.
    """

    # duplicate graph
    with tf.Graph().as_default() as tmp_graph:
        _ = tf.import_graph_def(graph_d, name="")
        graph_d = tmp_graph.as_graph_def()

    if preserve_leaf_names:
        # find all output nodes
        output_node_names = set(getOutputNodeNames(graph_d))
        # remove the leafs to preserve from the output list
        output_node_names = list(output_node_names - set(preserve_leaf_names))

        if output_node_names:
            # remove all remaining leaf nodes
            tmp_graph_d = framework.graph_pb2.GraphDef()
            for node_d in graph_d.node:
                if not node_d.name in output_node_names:
                    tmp_graph_d.node.extend([node_d])  # add node to graph
            retval = pruneLeafSubgraphs(tmp_graph_d, preserve_leaf_names)
        else:
            retval = graph_d
    else:  # empty graph because all leaves have been removes
        retval = framework.graph_pb2.GraphDef()

    return retval


def convertTensorflow2NumpyShape(shape_tf):
    """ Converts a tensorflow `TensorShape` to a numpy shape.
    All unknown values for partial shapes will be converted to -1.

    Args:
        shape_tf: A `TensorShape` object to convert.

    Returns:
        A list of values representing a valid numpy style shape.
    """
    retval = [shape_val if shape_val is not None else -
              1 for shape_val in shape_tf.as_list()]
    return retval


def convertNumpy2TensorflowShape(shape_np):
    """ Converts a numpy shape to a tensorflow shape.
    All unknown (-1) values for partial shapes will be converted to None.

    Args:
        shape_np: A list of values representing a valid numpy shape.

    Returns:
        A list of values representing a valid tensorflow style shape.
    """
    retval = [shape_val if shape_val >= 0 else None for shape_val in shape_np]
    return retval


def getInputShape(graph_d, numpy_format=False):
    """ Retrieves the shape of all inputs to specified `GraphDef` object.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.
        numpy_format: boolean - if False (default), shape is given in tensorflow format,
            otherwise, numpy format.

    Returns:
        A mapping string => list: from input tensor name to shape.
    """

    retval = {}

    input_node_names = getInputNodeNames(graph_d)

    tf.import_graph_def(graph_d, name="")
    for input_node_name in input_node_names:
        # find all output tensors for this placeholder, i.e. input:0, input:1, etc.
        try:
            i = 0
            while True:
                input_tensor_name = input_node_name + ":" + str(i)
                next_input_tensor = tf.get_default_graph().get_tensor_by_name(input_tensor_name)
                tensor_shape = next_input_tensor.shape
                if numpy_format:
                    tensor_shape = convertTensorflow2NumpyShape(tensor_shape)
                retval[input_tensor_name] = tensor_shape
                i += 1
        except:
            pass  # reached the end of the placeholder outputs

    return retval

def getInputOutputNodes(frozen_graph):
    """ Finds all input and output nodes in the specified graph.

    Args:
        frozen_graph: TensorFlow frozen graph

    Returns:
        A list of input and output node names.
    """
    predefined_inputs = ['segment', 'mask', 'input_ids']
    graph_d = loadGraphProtobufFromFile(frozen_graph)
    inputs = getInputNodeNames(graph_d)
    outputs = getOutputNodeNames(graph_d)
    nodes = [str for str in inputs if
                 any(sub in str for sub in predefined_inputs)]
    if len(nodes) == len(predefined_inputs):
        return [inputs, outputs]
    else:
        status, inputs = findNodeByName(graph_d, predefined_inputs)
        if status:
            return [inputs, outputs]
        else:
            raise RuntimeError("Cannot find suitable inputs for this tool, please indicate the names of inputs after preprocessing")

def findNodeByName(graph_d, node_name):
    """ Finds nodes specified by name in the specified graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.
        node_name: String name of node to check.

    Returns:
        status - True if all nodes are found, False otherwise
        A list of node names.
    """
    status = False
    all_nodes = list(getNodeNames(graph_d.node))
    retval = [str for str in all_nodes if
                 any(sub in str for sub in node_name)]
    if len(node_name) == len(retval):
        status = True

    return status, retval
