import tensorflow as tf


def data_prep_with_saved_model(
    saved_model_dir, 
    data_filenames, 
    batch_size,
    data_prep_start_node="serialized_example:0",
    data_prep_end_node="DatasetToSingleElement:0"
    ):

    # SavedModel with data pre-processing steps
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], saved_model_dir) 

    # Extract graph definition
    gdef = sess.graph_def

    # Load TFRecord files then generate a Dataset of batch
    dataset = tf.data.TFRecordDataset(data_filenames)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    dataset_b = iterator.get_next()

    # Preprocess data
    data_out, = tf.import_graph_def(
                        gdef,
                        input_map={data_prep_start_node: dataset_b},
                        return_elements=[data_prep_end_node])

    return data_out










