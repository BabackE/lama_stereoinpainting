import torch
print(f"cuda={torch.cuda.is_available()}")
print(f"nccl={torch.cuda.nccl.version()}")

import tensorflow as tf
print(f"tensorflow={tf.config.list_physical_devices('GPU')}")

