import tensorflow as tf
from latent_3d_points.src.autoencoder import Configuration as Conf, model_saver_id
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.ae_templates import default_train_params, mlp_architecture_ala_iclr_18
from os import path as osp

n_pc_points = 2048                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
load_epoch = 350
ae_loss = 'emd'                             # Loss to optimize: 'emd' or 'chamfer'
experiment_name = 'shapenet_{}_{}'.format(n_pc_points, bneck_size)
model_path = osp.join('data/', experiment_name)

# Restore the model
loaded_model_path = osp.join(model_path, model_saver_id + '-' + str(int(load_epoch)))

# Freeze the model
checkpoint = tf.train.get_checkpoint_state(model_path)
input_checkpoint = checkpoint.model_checkpoint_path

with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # get graph definition
    gd = sess.graph.as_graph_def()
    # fix batch norm nodes
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    output_node_name = '{}_2/Max'.format(experiment_name) # Max pooling node

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        gd,  # The graph_def is used to retrieve the nodes
        [output_node_name]  # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model-{}.pb".format(load_epoch)

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
