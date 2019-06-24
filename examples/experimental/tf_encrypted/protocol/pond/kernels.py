M = 107


def _pond_add_public_public(replica_context, x, y):
  return replica_context.experimental_run_v2(lambda x, y: x + y, args=(x, y))

def _pond_add_private_private(replica_context, x, y):
  return replica_context.experimental_run_v2(lambda x, y: (x + y) % M, args=(x, y))

def _pond_add_public_private(replica_context, x, y):
  # TODO
  raise NotImplementedError()

def _pond_add_private_public(replica_context, x, y):
  return _pond_add_public_private(replica_context, y, x)
