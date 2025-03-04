# Welleat


export PYTHONPATH=$PYTHONPATH:/path/to/models
python models/research/object_detection/model_main_tf2.py \    --pipeline_config_path=models/research/object_detection/pipeline.config \    --model_dir=object_detection/training \    --alsologtostderr