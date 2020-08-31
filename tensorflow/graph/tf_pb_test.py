import tensorflow as tf
import os.path
import time

import sys
import os, argparse
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tf_graph_utils import *
import numpy as np


def PruneGraph(graph_def):
    #Prune graph
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/multi_head_attention/MatMul_1")
    node.input[0] = "Tower_0/gpu/transform2_block/multi_head_attention/Softmax"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/multi_head_attention/MatMul_1")
    node.input[0] = "Tower_0/gpu/transform2_block_1/multi_head_attention/Softmax"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/multi_head_attention/MatMul_1")
    node.input[0] = "Tower_0/gpu/transform2_block_2/multi_head_attention/Softmax"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/multi_head_attention/MatMul_1")
    node.input[0] = "Tower_0/gpu/transform2_block_3/multi_head_attention/Softmax"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/multi_head_attention/MatMul_1")
    node.input[0] = "Tower_0/gpu/transform2_block_4/multi_head_attention/Softmax"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/multi_head_attention/MatMul_1")
    node.input[0] = "Tower_0/gpu/transform2_block_5/multi_head_attention/Softmax"

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/add")
    node.input[1] = "Tower_0/gpu/transform2_block/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/add")
    node.input[1] = "Tower_0/gpu/transform2_block_1/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/add")
    node.input[1] = "Tower_0/gpu/transform2_block_2/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/add")
    node.input[1] = "Tower_0/gpu/transform2_block_3/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/add")
    node.input[1] = "Tower_0/gpu/transform2_block_4/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/add")
    node.input[1] = "Tower_0/gpu/transform2_block_5/multi_head_attention/conv_1d_1/conv_1d_operation/BiasAdd"

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/add_1")
    node.input[1] = "Tower_0/gpu/transform2_block/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/add_1")
    node.input[1] = "Tower_0/gpu/transform2_block_1/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/add_1")
    node.input[1] = "Tower_0/gpu/transform2_block_2/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/add_1")
    node.input[1] = "Tower_0/gpu/transform2_block_3/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/add_1")
    node.input[1] = "Tower_0/gpu/transform2_block_4/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/add_1")
    node.input[1] = "Tower_0/gpu/transform2_block_5/ffn_mlp/conv_1d_1/conv_1d_operation/BiasAdd"

    node = getNodeByName(graph_def, "SparseToDense")
    node.input[2] = "GatherNd"

    tf.constant([1], name="Tower_0/gpu/transform2_block/ln_layer/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/ln_layer/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block/ln_layer/moments/variance/reduction_indices"
    tf.constant([1], name="Tower_0/gpu/transform2_block/ln_layer_1/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/ln_layer_1/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block/ln_layer_1/moments/variance/reduction_indices"

    tf.constant([1], name="Tower_0/gpu/transform2_block_1/ln_layer/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/ln_layer/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_1/ln_layer/moments/variance/reduction_indices"
    tf.constant([1], name="Tower_0/gpu/transform2_block_1/ln_layer_1/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/ln_layer_1/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_1/ln_layer_1/moments/variance/reduction_indices"

    tf.constant([1], name="Tower_0/gpu/transform2_block_2/ln_layer/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/ln_layer/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_2/ln_layer/moments/variance/reduction_indices"
    tf.constant([1], name="Tower_0/gpu/transform2_block_2/ln_layer_1/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/ln_layer_1/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_2/ln_layer_1/moments/variance/reduction_indices"

    tf.constant([1], name="Tower_0/gpu/transform2_block_3/ln_layer/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/ln_layer/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_3/ln_layer/moments/variance/reduction_indices"
    tf.constant([1], name="Tower_0/gpu/transform2_block_3/ln_layer_1/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/ln_layer_1/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_3/ln_layer_1/moments/variance/reduction_indices"

    tf.constant([1], name="Tower_0/gpu/transform2_block_4/ln_layer/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/ln_layer/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_4/ln_layer/moments/variance/reduction_indices"
    tf.constant([1], name="Tower_0/gpu/transform2_block_4/ln_layer_1/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/ln_layer_1/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_4/ln_layer_1/moments/variance/reduction_indices"

    tf.constant([1], name="Tower_0/gpu/transform2_block_5/ln_layer/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/ln_layer/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_5/ln_layer/moments/variance/reduction_indices"
    tf.constant([1], name="Tower_0/gpu/transform2_block_5/ln_layer_1/moments/variance/reduction_indices")
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/ln_layer_1/moments/variance")
    node.input[1] = "Tower_0/gpu/transform2_block_5/ln_layer_1/moments/variance/reduction_indices"

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/ln_layer/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block/ln_layer_1/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/ln_layer/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_1/ln_layer_1/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/ln_layer/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_2/ln_layer_1/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/ln_layer/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_3/ln_layer_1/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/ln_layer/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_4/ln_layer_1/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))

    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/ln_layer/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))
    node = getNodeByName(graph_def, "Tower_0/gpu/transform2_block_5/ln_layer_1/moments/variance/reduction_indices")
    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto([1], tf.int32, [2])))

    return graph_def
	
def export_pb_from_meta_graph(input_meta, output_pb, input_nodes, output_nodes, placeholder_dtypes, strip_unused_nodes):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    tf.reset_default_graph()
    session = tf.compat.v1.Session(config=config, graph=tf.Graph())
    with session.graph.as_default():
        tf.compat.v1.train.import_meta_graph(input_meta)
        session.run(tf.compat.v1.tables_initializer())
        '''
        input_tensor_names = [var.name for var in tf.compat.v1.get_collection('input_dict')]
        output_tensor_names = [var.name for var in tf.compat.v1.get_collection('output_dict')]
        print(input_tensor_names, output_tensor_names)
        graph = tf.get_default_graph()
        input_t = graph.get_tensor_by_name(input_tensor_names[0])
        output_t = graph.get_tensor_by_name(output_tensor_names[0])
        output = session.run(output_t, feed_dict={input_t: ["ssssssss"]})
        print(output)
        '''
		
        new_graph_def = session.graph_def

        # Prune graph
        new_graph_def = PruneGraph(new_graph_def)
        
        
	# Convert variables to constants
        new_graph_def = tf.graph_util.convert_variables_to_constants(session, new_graph_def,  output_nodes)

	# Transform graph 

        transforms = [
            "remove_nodes(op=StopGradient, op=Assert)",
            "fold_batch_norms",
            "fold_old_batch_norms",
            "fuse_pad_and_conv",
            strip_unused_nodes,
        ]
        new_graph_def = TransformGraph(
            new_graph_def, input_nodes, output_nodes, transforms)
                
	# optimize for inference graph
        new_graph_def = optimize_for_inference_lib.optimize_for_inference(
                new_graph_def,
                input_nodes, # an array of the input node(s)
                output_nodes, # an array of output nodes
#                [tf.string.as_datatype_enum, tf.float32.as_datatype_enum, tf.int64.as_datatype_enum])
                placeholder_dtypes)
        with open(output_pb, 'wb') as f:
            f.write(new_graph_def.SerializeToString())

        '''
        tf.import_graph_def(output_graph_def, name="")
        graph = tf.get_default_graph()
        input_t = graph.get_tensor_by_name(input_tensor_names[0])
        output_t = graph.get_tensor_by_name(output_tensor_names[0])
        print(input_t, output_t)
        output = session.run(output_t, feed_dict={input_t: ["ssssssss"]})
        print(output)
        '''

def GenRandStr1(len):
    import base64
    return base64.b64encode(os.urandom(len))

def GenRandStr2(randomlength):
    from random import Random
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    length = len(chars) - 1
    random = Random()
    for i in range(randomlength):
        str+=chars[random.randint(0, length)]
    return str

def PrepareInputs(batch, seqlen):
    input_s = []
    for i in range(batch):
    #    input_s.append("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
        input_s.append("你好你发斯蒂芬斯蒂芬斯蒂芬斯蒂芬斯蒂芬是否水电费的你好你发斯蒂芬斯蒂芬斯蒂芬斯蒂芬斯蒂芬是否水电费的你好你发斯蒂芬斯蒂芬斯蒂芬斯蒂芬斯蒂芬是否水电费的>你好你发斯蒂芬斯蒂芬斯蒂芬斯蒂芬斯蒂芬是否水电费的")
    #    input_s.append("ssssss")
    #    input_s.append(GenRandStr2(seqlen))

#    print(input_s)
    return input_s, batch

def run_frozen_pb(fname, input_names, output_names, input_f, batch, seqlen, iteration):
    tf.reset_default_graph()
    graph_def = load_frozen_pb(fname)
    graph = tf.import_graph_def(graph_def, name='')
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    session = tf.compat.v1.Session(config=config, graph=graph)
    try:
        session.run(tf.get_default_graph().get_operation_by_name('init_all_tables'))
    except Exception:
        print("no init_all_tables")

    #warm-up
    inputs, _ = input_f(batch, seqlen)
    feed_dict = {}
    for i in range(len(input_names)):
        feed_dict[input_names[i]] = inputs
    session.run(output_names, feed_dict=feed_dict)

    time_s = time.time()
    for i in range(iteration):
        output = session.run(output_names, feed_dict=feed_dict)
        print(output[0][0])
        print(output[0].shape)
#        np.save('input1.npy', output1) 
#        print(output2.shape)
#        np.save('input2.npy', output2) 
#        print(output3.shape)
#        np.save('input3.npy', output3) 
    time_e = time.time()
    time0 = (time_e - time_s)

    session.close()
    print("Time to execute=", time0 , "s, throughput=", batch*iteration/time0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tf_pb_test.py export_graph test_graph")
        exit()

    print(sys.argv[1], sys.argv[2])
    OUTPUT_PB1 = "./frozen_graph.pb"
    OUTPUT_PB2 = "./frozen_graph_preprocess.pb"
    OUTPUT_PB3 = "./frozen_graph_goya.pb"
    batch = 24 
    # Export frozen graphs
    if sys.argv[1] == "True":
        FROZEN_META = "./frozen.meta"

        # export whole graph
        input_nodes = ["input/split"]
        output_nodes = ["Tower_0/gpu/Softmax", "init_all_tables"]
        placeholder_dtypes = [tf.string.as_datatype_enum]
        strip_unused_nodes = "strip_unused_nodes(type=string)"
        export_pb_from_meta_graph(FROZEN_META, OUTPUT_PB1, input_nodes, output_nodes, placeholder_dtypes, strip_unused_nodes)

        # export preprocessing graph
        input_nodes = ["input/split"]
        output_nodes = ["input/strided_slice_1", "init_all_tables"]
        placeholder_dtypes = [tf.string.as_datatype_enum]
        strip_unused_nodes = "strip_unused_nodes(type=string)"
        export_pb_from_meta_graph(FROZEN_META, OUTPUT_PB2, input_nodes, output_nodes, placeholder_dtypes, strip_unused_nodes)

        # export goya graph
        input_nodes = ["input/strided_slice_1"]
        output_nodes = ["Tower_0/gpu/Softmax"]
        placeholder_dtypes = [tf.int64.as_datatype_enum, tf.float32.as_datatype_enum, tf.float32.as_datatype_enum]
        strip_unused_nodes = "strip_unused_nodes(name=add_6, type_for_name=int64, shape_for_name=\"%d, 100\", \
            name=Tower_0/gpu/pos_embedding/mul, type_for_name=float32, shape_for_name=\"%d, 100, 512\", \
            name=Tower_0/gpu/Tile, type_for_name=float32, shape_for_name=\"%d, 100, 512\")" % (batch, batch, batch)
        export_pb_from_meta_graph(FROZEN_META, OUTPUT_PB3, input_nodes, output_nodes, placeholder_dtypes, strip_unused_nodes)

    # Run frozen pb
    if sys.argv[2] == "True":
        seqlen = 100
        iteration = 1

        def load_input_random(bs, seqlen):
            if bs == None:
                bs = 1
            add_arr = np.random.randint(low=0, high=30522, size=[bs, seqlen], dtype=np.int64)
            mul_arr = np.random.random([bs, seqlen, 512])
            mask_arr = np.ones([bs, seqlen, 512], dtype=np.float32)

            return [add_arr, mul_arr, mask_arr], batch

        # Run whole graph
        input_tensor_names = ['input/split:0']
        output_tensor_names = ['Tower_0/gpu/Softmax:0']
#        run_frozen_pb(OUTPUT_PB1, input_tensor_names, output_tensor_names, PrepareInputs, batch, seqlen, iteration)

        # Run preprocessing
        input_tensor_names = ['input/split:0']
#        output_tensor_names = ['add_6:0', 'Tower_0/gpu/pos_embedding/mul:0', 'Tower_0/gpu/Tile:0']
        output_tensor_names = ["input/strided_slice_1:0"]
        run_frozen_pb(OUTPUT_PB2, input_tensor_names, output_tensor_names, PrepareInputs, batch, seqlen, iteration)

        # Run goya graph
        input_tensor_names = ['add_6:0', 'Tower_0/gpu/pos_embedding/mul:0', 'Tower_0/gpu/Tile:0']
        output_tensor_names = ['Tower_0/gpu/Softmax:0']
#        run_frozen_pb(OUTPUT_PB1, input_tensor_names, output_tensor_names, load_input_random, batch, seqlen, iteration)

