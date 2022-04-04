#https://imageai.readthedocs.io/en/latest/customdetection/
from imageai.Detection.Custom import DetectionModelTrainer


trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="objdetimg")
trainer.setTrainConfig(object_names_array=["pin"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()