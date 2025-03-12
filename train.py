from ultralytics import YOLO

# Create model
model = YOLO('yolov8n-cls.pt')  # load pretrained classification model

# Train the model with your dataset
results = model.train(
    data='C:/Users/Mandi/yolo_hotdog_classifier/dataset',  # path containing train and val folders
    epochs=50,
    imgsz=224,
    batch=16,  # restored normal batch size
    name='hotdog_classifier',
    val=True  # enable validation
)
