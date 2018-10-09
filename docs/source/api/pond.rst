tf-encrypted
============

The main way to interface through tf-encrypted is via a `Protocol` object.

A typical program to do a prediction will look like this:

    # get named players from hostmap configuration
    server0 = config.get_player('server0')
    server1 = config.get_player('server1')
    crypto_producer = config.get_player('crypto_producer')

    # perform secure operations using the Pond protocol
    with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

        # get input from inputters as private values
        inputs = [prot.define_private_input(inputter) for inputter in inputters]

        # sum all inputs and multiply by count inverse (ie divide)
        result = reduce(lambda x, y: x + y, inputs) * (1 / len(inputs))

        # send result to receiver who can finally decrypt
        result_op = prot.define_output([result], result_receiver)

        with tfe.Session() as sess:
            tfe.run(sess, result_op, tag='average')


  API
  ====

.. automodule:: tensorflow_encrypted.protocol
  :members:
