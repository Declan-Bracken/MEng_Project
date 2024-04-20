# Train a yolo model
from roboflow import Roboflow
rf = Roboflow(api_key="QlMV765HIGubgz2flItC")
project = rf.workspace("transcript-detr-training").project("transcript-table-detector")
version = project.version(6)
dataset = version.download("coco")

