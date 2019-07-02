RFC: Toward TF Encrypted 1.0
==================================
This is a living design doc for TF Encrypted's first big refactor.

T.O.C.
1. Problems with current implementation
2. Overview
3. Details
4. Protocol Requirements


### Problems with current implementation
- Too much logic is wrapped up in the Pond protocol, and the file is too large
- It is not clear which methods a new protocol needs to override to conform to the public TFE API (this API is largely implicitly defined and not stated explicitly anywhere)
- BackingTensors being wrapped up in factories are impossible to test
- Excessive code duplication: each operation is implemented at least once for each BackingTensor and each protocol



### Overview
The guiding philosophy of the refactor will be to replace TFE's major features with native TensorFlow abstractions that are covered by the TF Stability Guarantees for the lifetime of TF 2.0.  The goals of doing so include:
- Upgrading TF Encrypted to TensorFlow 2.0
- Leverage as much of the existing TensorFlow ecosystem with minimal rewriting and/or maintenance
- Continue to rely on the stability guarantee of the TensorFlow Public API
- Improve our own code modularity, extensibility, and sustainability and reduce code duplication

Concretely, we will replace our own type system of custom tensors with a light set of hooks over existing tensor types in TensorFlow.  The current Protocol class would be replaced by custom subclasses of `tf.distribute.Strategy` and `tf.distribute.StrategyExtended` that define the necessary protocol operations and handle any distributed computation that's required by them.

By doing so, we hope to satisfy each of the above goals, as well as all potential future requirements detailed by @mortendahl (see below)

### Details
- `Protocol` class replaced by custom implementations of `tf.distribute.Strategy` and `tf.distribute.StrategyExtended`
  - The Strategy abstraction functions as our entry-point into the rest of the higher-level TF APIs
  - The current strategy scope will determine which protocol kernel is used behind the scenes when calling a tfe op, e.g. `tfe.matmul`, similar to how it works now with Protocols
  - For n-party protocols, this API provides a convenient set of tools for orchestrating distributed computation in a way that can induce out-of-the-box support for both tf.keras and the Estimator API
  - Note these need not operate in the multi-party setting: see [`tf.distribute.OneDeviceStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy) for an example of a single device strategy
- Replace `BackingTensor` (`BackingVariable`) with `tf.Tensor` (`tf.Variable`)
  - Allows us to use `Strategy` instead of `Protocol` (since Strategies must run on native TF Tensors/Variables)
  - Allows us to use the `tf.custom_gradient` strategy to [bring back autograd](https://github.com/tf-encrypted/tf-encrypted/issues/244)
  - Custom `BackingTensor` types can be implemented through `tf.VariantTensor`, similar to how the [`tf-big`](https://github.com/tf-encrypted/tf-big) project uses libgmp behind the scenes.  It may also be possible to use Tensors from another Graph to source the VariantTensor, which would allow us to reproduce our existing CRTTensor as well as the NativeTensor with an explicit modulus
  - Concretely, we will interface these plain TF Tensors & Variables with our protocols through a collection of light hooks, including but not limited to:
    - `dispatch_id` for marking public/private tensors (other dispatch_ids can be protocol-specific, e.g. masked in Pond/SecureNN)
    - `encoding_config` for identifying which fixed-point (or quantized) encoding that generated the tensor’s current representation
    - `encoded_tensor._encode(config)` to convert between different encodings, used internally and for the public `tfe.cast` (to allow for lifting and casting as desired when mixing operations among protocols)

### Protocol Requirements
From issue [#588](https://github.com/tf-encrypted/tf-encrypted/issues/558):
- supporting any number of participants
  - n parties who each only trust themselves
  - [x] `tf.distribute.Strategy` is agnostic to number of parties, and can even easily allow for sets of heterogenous devices through the `ClusterResolver` API
- converting from one protocol to another
  - layers in NN handled by different protocols as in Gazelle
  - [x] Because each encoding is recognizable form its `encoding_config` hook, we can do lifting and casting as requested by each protocol.  Switching between protocol operations is simple given that `tfe.<Op>` commands can be routed to the correct kernel based on its inputs' dispatch_ids and the current Strategy scope.
- mixing plaintext and encrypted computations
  - FL with secure aggregation
  - [x] Our own (privacy-aware) operations will live under the tfe namespace, so we do not override the existing tf namespace and all its operations.  Additionally, with Strategies,  users can choose to run native TenosrFlow on various machines as long as they are in the Strategy's ReplicaContext and are careful about keeping track of which values are public/private and floating/fixed-point representations
- step-wise execution with networking handled externally
  - FL with bulletin boards to disconnect data owners
  - third party MPC libraries
  - [ ] this one is harder -- I'm not sure what this looks like, so I can't make guarantees, but Strategies can be very lightweight
- concurrent protocols between different subsets of players
  - an aggregator in PATE might run different protocols with each teacher
  - [x] this is easy if expected to be specified by the user -- create separate Strategies for each player, all of them will execute concurrently if staying in the same graph, or eagerly if not.

### Open questions
- Can we trick TensorFlow into thinking all of our custom BackingTensors are tf.Tensors?
  - `CRTTensor`:
    - `VariantTensor` or `ResourceTensor` pointing to a `TensorArray` in another Graph?
  - `NativeTensor`:
    - `VariantTensor` or `ResourceTensor` pointing to a custom tensor type that automatically performs the modulus every time a new tensor is created?
