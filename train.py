import os
import tensorflow as tf
from object_detection import model_lib_v2
from object_detection.utils import config_util

# Set the paths
pipeline_config_path = 'C:/Users/younes.kebour/Documents/CITC/WELLEAT/Tray Detection/pipeline.config'
model_dir = 'C:/Users/younes.kebour/Documents/CITC/WELLEAT/Tray Detection/model'

# Set the training parameters
num_train_steps = 50000
sample_1_of_n_eval_examples = 1

# Configure logging
tf.get_logger().setLevel('INFO')

# Read the pipeline config file with explicit encoding
configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
model_config = configs['model']

# Run the training
model_lib_v2.train_loop(
    pipeline_config_path=pipeline_config_path,
    model_dir=model_dir,
    train_steps=num_train_steps,
    sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
    checkpoint_every_n=1000,
    record_summaries=True
)