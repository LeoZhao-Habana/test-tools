import tensorflow as tf
import os.path
import time

from tensorflow.python.platform import gfile
from google.protobuf import text_format
import sys
import os, argparse
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

#FLAGS = tf.app.flags.FLAGs

#Input Graph model file location
#tf.app.flags.DEFINE_string('model_dir', './', """Paht to classify_image_graph_def.pb""")

#Output Graph model protobuf as text format & binary format
#tf.app.flags.DEFINE_string('output_graph_txt', './output_graph.pbtxt', """pbtxt""")
#tf.app.flags.DEFINE_string('output_graph_pb', './output_graph.pb', """pb""")

OUTPUT_PB = "./frozen_graph.pb"
FROZEN_META = "./frozen.meta"

def printnodes(nodes):
    for node in nodes:
        print(node.name, node.op)

    return

def getNodeByName(nodes, name):
    for node in nodes:
        if node.name == name:
            return node

def load_meta_graph(fname):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    session = tf.compat.v1.Session(config=config, graph=tf.Graph())
    with session.graph.as_default():
        tf.compat.v1.train.import_meta_graph(fname)
        print(tf.compat.v1.tables_initializer())
        session.run(tf.compat.v1.tables_initializer())
        input_tensor_names = [var.name for var in tf.compat.v1.get_collection('input_dict')]
        output_tensor_names = [var.name for var in tf.compat.v1.get_collection('output_dict')]
        print(input_tensor_names, output_tensor_names)
        graph = tf.get_default_graph()
        input_t = graph.get_tensor_by_name(input_tensor_names[0])
        output_t = graph.get_tensor_by_name(output_tensor_names[0])
        output = session.run("input/StaticRegexReplace:0", feed_dict={input_t: ["你好，中国          ", "红花发的发生的发顺丰"]})

        gd = session.graph_def
        node = getNodeByName(gd.node, "gpu/transform2_block/multi_head_attention/conv_1d_1/B")
        print(node.op, node.name, node.attr)
        return

        #Prune graph
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block/multi_head_attention/MatMul_1")
        node.input[0] = "Tower_0/gpu/transform2_block/multi_head_attention/Softmax"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_1/multi_head_attention/MatMul_1")
        node.input[0] = "Tower_0/gpu/transform2_block_1/multi_head_attention/Softmax"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_2/multi_head_attention/MatMul_1")
        node.input[0] = "Tower_0/gpu/transform2_block_2/multi_head_attention/Softmax"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_3/multi_head_attention/MatMul_1")
        node.input[0] = "Tower_0/gpu/transform2_block_3/multi_head_attention/Softmax"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_4/multi_head_attention/MatMul_1")
        node.input[0] = "Tower_0/gpu/transform2_block_4/multi_head_attention/Softmax"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_5/multi_head_attention/MatMul_1")
        node.input[0] = "Tower_0/gpu/transform2_block_5/multi_head_attention/Softmax"

        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block/add")
        node.input[1] = "Tower_0/gpu/transform2_block/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_1/add")
        node.input[1] = "Tower_0/gpu/transform2_block_1/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_2/add")
        node.input[1] = "Tower_0/gpu/transform2_block_2/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_3/add")
        node.input[1] = "Tower_0/gpu/transform2_block_3/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_4/add")
        node.input[1] = "Tower_0/gpu/transform2_block_4/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_5/add")
        node.input[1] = "Tower_0/gpu/transform2_block_5/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"

        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block/add_1")
        node.input[1] = "Tower_0/gpu/transform2_block/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_1/add_1")
        node.input[1] = "Tower_0/gpu/transform2_block_1/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_2/add_1")
        node.input[1] = "Tower_0/gpu/transform2_block_2/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_3/add_1")
        node.input[1] = "Tower_0/gpu/transform2_block_3/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_4/add_1")
        node.input[1] = "Tower_0/gpu/transform2_block_4/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
        node = getNodeByName(gd.node, "Tower_0/gpu/transform2_block_5/add_1")
        node.input[1] = "Tower_0/gpu/transform2_block_5/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"

        node = getNodeByName(gd.node, "SparseToDense")
        node.input[2] = "GatherNd"


#        for node in session.graph_def.node:
#            print(node.name, node.op, node.input)
        
        input_nodes = ["input/split"]
#        input_nodes = ["add_6", "Tower_0/gpu/Tile"]
        output_nodes = ["Tower_0/gpu/Softmax", "init_all_tables"]
#        output_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph_def,  ["Tower_0/gpu/Softmax"])
        output_graph_def = tf.graph_util.convert_variables_to_constants(session, gd,  output_nodes)

        
        transforms = [
            "remove_nodes(op=StopGradient, op=Assert)",
            "fold_batch_norms",
            "fold_old_batch_norms",
            "fuse_pad_and_conv",
#            "strip_unused_nodes(name=add_6, type_for_name=int64, name=Tower_0/gpu/Tile, type_for_name=float)",
            "strip_unused_nodes(type=string)",
        ]
        output_graph_def = TransformGraph(
            output_graph_def, input_nodes, output_nodes, transforms)
        
        outputGraph_def = optimize_for_inference_lib.optimize_for_inference(
                output_graph_def,
                input_nodes, # an array of the input node(s)
                output_nodes, # an array of output nodes
                tf.string.as_datatype_enum)
        with open(OUTPUT_PB, 'wb') as f:
            f.write(outputGraph_def.SerializeToString())
#        tf.import_graph_def(output_graph_def, name="")
#        graph = tf.get_default_graph()
#        input_t = graph.get_tensor_by_name(input_tensor_names[0])
#        output_t = graph.get_tensor_by_name(output_tensor_names[0])
#        print(input_t, output_t)
#        output = session.run(output_t, feed_dict={input_t: ["ssssssss"]})
#        print(output)

def load_frozen_pb(fname):
    with gfile.FastGFile(fname, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def run_frozen_pb(fname):
    graph_def = load_frozen_pb(fname)
    g = tf.import_graph_def(graph_def, name='')
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    session = tf.compat.v1.Session(config=config, graph=g)
    input_tensor_names = ['input/split:0']
    output_tensor_names = ['Tower_0/gpu/Softmax:0']
    output1_tensor_names = ['add_6:0', 'Tower_0/gpu/Tile:0', 'Tower_0/gpu/pos_embedding/mul:0']
    print(tf.get_default_graph().get_operation_by_name('init_all_tables'))
#    session.run(tf.get_default_graph().get_operation_by_name('init_all_tables'))
    session.run('init_all_tables')
    output = session.run(output_tensor_names, feed_dict={input_tensor_names[0]: ["ssssssss"]})
    output = session.run(output1_tensor_names, feed_dict={input_tensor_names[0]: ["ssssssss"]})

    batch = 256
    input_s = []
    for i in range(batch):
        input_s.append("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")

    iteration = 200
    '''
    time_s = time.time()
    for i in range(iteration):
        output = session.run(output_tensor_names, feed_dict={input_tensor_names[0]: input_s})
    time_e = time.time()
    time1 = (time_e - time_s)/iteration
    '''
    time_s = time.time()
    for i in range(iteration):
        output1, output2, output3 = session.run(output1_tensor_names, feed_dict={input_tensor_names[0]: input_s})
#        print(output1, output2, output3)
    time_e = time.time()
    time2 = (time_e - time_s)/iteration
#    print("time1 to execute=", time1 , "s, throughput=", batch/time1)
    print("time2 to execute=", time2 , "s, throughput=", batch/time2)
#    print("time1 / time2 =", time1 / time2)
#    print(output)


def convert_pb_to_pbtxt(graph_def):
    with gfile.FastGFile(FLAGS.output_graph_txt, 'wb') as f:
        f.write(text_format.MessageToString(graph_def))
        # MessageToString(message, as_utf8=False, as_one_line=False)  Convert protobuf message to text format

    return graph_def

#  with gfile.FastGFile(FLAGS.output_graph_pb, 'wb') as f:
#   f.write(graph_def.SerializeToString())  #serializes the message and returns it as a string. Note that the bytes are binary, not text; we only use the str type as a convenient container.

def list_all_graph_nodes(graph_def):
    for node in graph_def.node:
        print(node.name, node.op, node.input, node.output, node.attr)

def frozen_to_saved_model(frozen_pb, saved_model_dir, input_nodes, output_nodes):
    config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

with gfile.FastGFile('path_to/frozen_inference_graph.pb', 'rb') as f: # 加载冻结图模型文件
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入图定义
    sess.run(tf.global_variables_initializer())

    # 建立tensor info bundle
    input_img = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name('ImageTensor:0'))
    output = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name('SemanticPredictions:0'))
    print(output)

    export_path_base = "export_path"
    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes('1'))

    # Export model with signature
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'inputs': input_img},
          outputs={'outputs': output},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'a_signature':
              prediction_signature
          },
          main_op=tf.tables_initializer())

    builder.save()

def saved_model_to_frozen(saved_model_dir, output_nodes, pb_name, saved_tags=tag_constants.SERVING):
    input_saved_model_dir = saved_model_dir
    output_node_names = output_nodes
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = saved_tags
    output_graph_filename = pb_name

    freeze_graph.freeze_graph(input_graph_filename,
                              input_saver_def_path,
                              input_binary,
                              checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_filename,
                              clear_devices,
                              "", "", "",
                              input_meta_graph,
                              input_saved_model_dir,
                              saved_model_tags)

def ckpt_to_saved(fname, export_dir):
    graph = tf.Graph()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(fname)
        sess.run(tf.compat.v1.tables_initializer())
#        loader.restore(sess, trained_checkpoint_prefix)

        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING], strip_default_attrs=True)
        builder.save()

#model_filename = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')

if __name__ == "__main__":
    load_meta_graph(FROZEN_META)
    exit()
    run_frozen_pb(OUTPUT_PB)
    exit()
    metafile = sys.argv[1]
    export_dir = sys.argv[2]
    pb_name = "leo_frozen.pb"
    output_name = "Tower_0/gpu/Softmax"

    ckpt_to_saved(metafile, export_dir)
    freeze_saved_model(export_dir, output_name, pb_name)


