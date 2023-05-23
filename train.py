import data_setup, engine, model_builder, utils
import torch.nn as nn
import torch.optim as optim
import torch
from timeit import default_timer as timer 
from torchvision import transforms
import torchvision

#heyperparameters
BATCH_SIZE = 64
EPOCHS     = 80
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def instance_train_model(train_model:str,
                         imgage_size:int):

    train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((imgage_size, imgage_size)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor()
])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((imgage_size, imgage_size)),
        transforms.ToTensor(),
    ])

    #Base Model
    if train_model=='base_model':
        model = model_builder.EmotionModel()
    #ResNet 50 Model
    if train_model=='resnet50_model':
        model = model_builder.ResNetGray()
    #EfficientNet
    if train_model=='EfficientNet_model':
        model = model_builder.build_efficientnet_b2()
    #ViT Model
    if train_model=='EfficientNet_model':
        model = model_builder.MyViT()
    #pretrained ViT
    if train_model=='EfficientNet_model':
        model = model_builder.PretrainedViTModel()

    return model,train_transform,val_transform


model,train_transform,val_transform = instance_train_model("base_model",48)
#construct dataloader
train_dataloader, test_dataloader = data_setup.create_dataloaders(batch_size=BATCH_SIZE,
                                                                  train_transform=train_transform,
                                                                  val_transform=val_transform)

#criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(params=efficientnet.parameters(), 
#                              lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
#                              betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
#                              weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k
#create writerSummary

writer = utils.create_writer(experiment_name="64_batch",
                               model_name="base_model",
                               extra="80_epochs")
#train 
start_time = timer()
result = engine.train(model = model,
            train_dataloader= train_dataloader,
            test_dataloader = test_dataloader,
            optimizer       = optimizer,
            loss_fn         = criterion,
            epochs          = EPOCHS,
            device          = device,
            writer          = writer)


end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")



#save model
utils.save_model(model=model,
               target_dir="models",
               model_name="base_model_80_epoch.pth")

#export onnx 
utils.onnx_export(model=model,
                  image_size=48,
                  target_dir="onnx_model",
                  model_name = "base_model_80_epoch.onnx")
#plot history
utils.plot_loss_curves(result)