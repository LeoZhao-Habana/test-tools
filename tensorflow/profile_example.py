import os
import tempfile
import json

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

#first way to save trace
class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


batch_size = 100

inputs = tf.placeholder(tf.float32, [batch_size, 784])
targets = tf.placeholder(tf.float32, [batch_size, 10])

with tf.variable_scope("layer_1"):
    fc_1_out = fc(inputs, num_outputs=500, activation_fn=tf.nn.sigmoid)
with tf.variable_scope("layer_2"):
    fc_2_out = fc(fc_1_out, num_outputs=784, activation_fn=tf.nn.sigmoid)
with tf.variable_scope("layer_3"):
    logits = fc(fc_2_out, num_outputs=10)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

if __name__ == '__main__':
    mnist_save_dir = os.path.join(tempfile.gettempdir(), 'MNIST_data')
    mnist = input_data.read_data_sets(mnist_save_dir, one_hot=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        #second way by using profiler
        profiler = model_analyzer.Profiler(graph=sess.graph)

        #run options
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        #first way to save trace
        many_runs_timeline = TimeLiner()
        runs = 5
        for i in range(runs):
            batch_input, batch_target = mnist.train.next_batch(batch_size)
            feed_dict = {inputs: batch_input,
                         targets: batch_target}

            sess.run(train_op,
                     feed_dict=feed_dict,
                     options=options,
                     run_metadata=run_metadata)

            #first way to save trace with timeline
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)

            #second way by using profiler
            profiler.add_step(step=i, run_meta=run_metadata)
        many_runs_timeline.save('timeline_03_merged_%d_runs.json' % runs)

        #Second way to show trace & profile
        #graph view
        #统计内容为每个graph node的运行时间和占用内存
        profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
        option_builder.ProfileOptionBuilder.time_and_memory())
        #输出方式为timeline, 输出文件夹必须存在
        profile_graph_opts_builder.with_timeline_output(timeline_file='mnist_profiler.json')
        #定义显示sess.Run() 第1步的统计数据
        profile_graph_opts_builder.with_step(1)
        #显示视图为graph view
        profiler.profile_graph(profile_graph_opts_builder.build())

        #scope view
        #统计内容为所有trainable Variable Op
        profile_scope_opt_builder = option_builder.ProfileOptionBuilder(
        option_builder.ProfileOptionBuilder.trainable_variables_parameter())
        #显示的嵌套深度为4
        profile_scope_opt_builder.with_max_depth(4)
        #显示字段是params，即参数
        profile_scope_opt_builder.select(['params'])
        #根据params数量进行显示结果排序
        profile_scope_opt_builder.order_by('params')
        #显示视图为scope view
        profiler.profile_name_scope(profile_scope_opt_builder.build())

        #op view
        profile_op_opt_builder = option_builder.ProfileOptionBuilder()
        #显示字段：op执行时间，使用该op的node的数量。 注意：op的执行时间即所有使用该op的node的执行时间总和。
        profile_op_opt_builder.select(['micros','occurrence'])
        #根据op执行时间进行显示结果排序
        profile_op_opt_builder.order_by('micros')
        #过滤条件：只显示排名top 5
        profile_op_opt_builder.with_max_depth(4)
        #显示视图为op view
        profiler.profile_operations(profile_op_opt_builder.build())

        profile_op_opt_builder = option_builder.ProfileOptionBuilder()
        #显示字段：op占用内存，使用该op的node的数量。 注意：op的占用内存即所有使用该op的node的占用内存总和。
        profile_op_opt_builder.select(['bytes','occurrence'])
        #根据op占用内存进行显示结果排序
        profile_op_opt_builder.order_by('bytes')
        #过滤条件：只显示排名最靠前的5个op
        profile_op_opt_builder.with_max_depth(4)
        #显示视图为op view
        profiler.profile_operations(profile_op_opt_builder.build())

        #code view
        profile_code_opt_builder = option_builder.ProfileOptionBuilder()
        #过滤条件：显示minist.py代码。
        profile_code_opt_builder.with_max_depth(1000)
        profile_code_opt_builder.with_node_names(show_name_regexes=['mnist.py.*'])
        #过滤条件：只显示执行时间大于10us的代码
        profile_code_opt_builder.with_min_execution_time(min_micros=10)
        #显示字段：执行时间，且结果按照时间排序
        profile_code_opt_builder.select(['micros'])
        profile_code_opt_builder.order_by('micros')
        #显示视图为code view
        profiler.profile_python(profile_code_opt_builder.build())
