import tensorflow as tf
import os

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

def freeze_graph(model_dir, output_node_names, frozen_graph_path):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
        frozen_graph_path: the output path
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return None

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(frozen_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        #print("%d ops in the final graph." % len(output_graph_def.node))

def optimize_graph(input, output, frozen_graph, input_names, output_names):
    if not gfile.Exists(input):
        print("Input graph file '" + input + "' does not exist!")
        return -1

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(input, "rb") as f:
        data = f.read()
        if frozen_graph:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data.decode("utf-8"), input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names.split(","),
        output_names.split(","),
        dtypes.float32.as_datatype_enum)

    if frozen_graph:
        f = gfile.FastGFile(output, "w")
        f.write(output_graph_def.SerializeToString())
    else:
        graph_io.write_graph(output_graph_def,
            os.path.dirname(output),
            os.path.basename(output))
