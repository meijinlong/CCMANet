from ultralytics import YOLO

# 加载一个预训练的 YOLO11n 模型
model = YOLO("/home/YOLO/ultralytics-main1/yolo11n.yaml").load("/home/YOLO/ultralytics-main1/yolo11n.pt")

# 在 crowdhuman 数据集上训练模型 100 个周期
train_results = model.train(
    data="dior.yaml",  # 数据集配置文件路径
    epochs=300,  # 训练周期数
    imgsz=800,  # 训练图像尺寸
    batch=8,   #batch_size
    device=0,  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
)

## 评估模型在验证集上的性能
# metrics = model.val()
