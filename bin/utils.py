import tensorflow as tf


def read_records(filename):
    reader = tf.python_io.tf_record_iterator(filename)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            tf.logging.info("read: %d", len(records))
    return records


def batch_records(records, batch_nb, batch_size):
    data_length = len(records)
    start, end = batch_indices(batch_nb, data_length, batch_size)
    return records[start:end]


def batch_indices(batch_nb, data_length, batch_size):
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)

  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift

  return start, end


def lambda_preprocess_from_saved_model(data_prep_path, 
                                        input_node = 'serialized_example', 
                                        output_node = 'DatasetToSingleElement'):

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], data_prep_path)
        input_preprocess = sess.graph.get_operation_by_name(input_node).values()[0]
        output_preprocess = sess.graph.get_operation_by_name(output_node).values()[0]

        preprocess_fn = lambda record : sess.run(output_preprocess, {input_preprocess: record})
    return preprocess_fn

