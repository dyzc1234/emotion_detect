import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import mobilenet_v2
from efficientnet_pytorch import EfficientNet
from torchinfo import summary
import torchvision
#--------------------------------------------------------------------------base model--------------------------------------------------------------------------
class EmotionModel(nn.Module):

    def __init__(self, num_classes=7, l2_coef=0.01):
        super(EmotionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        self.l2_coef = l2_coef

    def forward(self, x):
        x = self.features(x)
        # print(x.shape,1)
        x = x.view(x.size(0), -1)
        # print(x.shape,2)
        x = self.classifier(x)
        # print(x.shape,3)

        return x

    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)**2
        return self.l2_coef * l2_reg

    def loss(self, input, target):
        return F.cross_entropy(input, target) + self.l2_regularization()


#--------------------------------------------------------------------------resnet model--------------------------------------------------------------------------
class ResNetGray(nn.Module):
    def __init__(self, dropout_rate=0.2, weight_decay=1e-3):
        super(ResNetGray, self).__init__()
        # 加载ResNet18预训练模型
        resnet = models.resnet50(pretrained=True)
        
        # 修改第一层以接受单通道输入
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改平均池化层以适应输入尺寸为 [512, 1, 48, 48] 的图像
        resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        
        # 修改全连接层以输出7个类别，并添加 L2 正则化项
        self.fc = nn.Linear(2048, 7)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        self.fc.weight_decay = weight_decay
        
        # 去掉原模型的最后一层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # 添加 dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x


#--------------------------------------------------------------------------inception model--------------------------------------------------------------------------

# # 加载预训练的Inception v3模型
# inception = models.inception_v3(pretrained=True)

# # 修改输入层以接受64x64灰度图像
# inception.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
# inception.Conv2d_1a_3x3.bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

# # 修改辅助分类器以输出7个类别
# num_classes = 7
# inception.AuxLogits.fc = nn.Linear(768, num_classes)

# # 修改主分类器以输出7个类别
# inception.fc = nn.Linear(2048, num_classes)



#--------------------------------------------------------------------------mobilenet model--------------------------------------------------------------------------

def create_mobilenet(num_classes=7):
    model = mobilenet_v2(pretrained=False)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    return model

#--------------------------------------------------------------------------efficientnet model--------------------------------------------------------------------------


def build_efficientnet_b2():
    # 加载预训练的EfficientNet-B0模型
    # model_name = 'efficientnet-b2'
    # # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    # efficientnet = EfficientNet.from_pretrained(model_name)
    # efficientnet = torchvision.models.efficientnet_b0(weights=weights)

    # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    efficientnet = torchvision.models.efficientnet_b2(weights=weights)
    efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
    efficientnet.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=7),
    )
    

    return efficientnet




#--------------------------------------------------------------------------Transform model--------------------------------------------------------------------------
#Patch Embedding turns a 2D input into a 1D sequence learnable embedding vector.

class PatchEmbedding(nn.Module):
    '''
     Args:
        in_channels (int): Number of color channels for the input images. Defaults to 1.
        patch_size (int): Size of patches to convert input image into. Defaults to 4.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 256.
    '''
    def __init__(self,
                 in_channels:int=1,
                 patch_size:int=4,
                 embedding_dim:int=256):
        super().__init__()

        #create a layer to turn an image into pathes
        self.patcher = nn.Conv2d(in_channels = in_channels,
                                 out_channels = embedding_dim,
                                 kernel_size = patch_size,
                                 stride = patch_size,
                                 padding = 0)
        #create a layer to flatten the patch features maps to a single dimension
        self.flatten = nn.Flatten(start_dim = 2,
                                  end_dim = 3)
        self.patch_size = patch_size
    def forward(self,x):
        image_resolution = x.shape[-1]
        assert image_resolution%self.patch_size==0,f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        # print("x.shape",x.shape)
        
        x_patched = self.patcher(x)
        # print("x_patched.shape",x_patched.shape)

        x_flattened = self.flatten(x_patched) 
        # print("x_flattened.shape",x_flattened.shape)

        return x_flattened.permute(0,2,1)#adjust the dimension [batch_size,P^2*C,N]->[batch_size,N,P^2*C]
    
#Transformer Encoder contains MSA and MLP block 

#MSA Block

class MSABlock(nn.Module):
    '''
    creat muti-head self-attention block ("MSA" for short)
    '''
    def __init__(self,
                 embedding_dim:int=256,
                 num_heads:int=12,
                 attention_drop:float=0):#viT paper doesnt use drop here
        super().__init__()
        
        #crete norm layer (LN)
        self.normlayer = nn.LayerNorm(normalized_shape = embedding_dim)

        #create Muti-Head Attention (MSA) layer
        self.mutihead_atten = nn.MultiheadAttention(embed_dim = embedding_dim,
                                                   num_heads = num_heads,
                                                   dropout = attention_drop,
                                                   batch_first = True) 
    def forward(self,x):

        x = self.normlayer(x)
        atten_output,_ = self.mutihead_atten(query=x,
                                           key=x,
                                           value=x,
                                           need_weights=False)#just need output of the layer other than the weights
        
        return atten_output

class MLPBlock(nn.Module):
    def __init__(self,
                embed_dim:int,
                mlp_size:int = 3072,
                dropout:float=0.1):
        super().__init__()
        
        #create norm layer (LN)
        self.normlayer = nn.LayerNorm(normalized_shape = embed_dim)

        #create MutiLayer perceptron layers(MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_features = embed_dim,
                      out_features = mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features = mlp_size,
                      out_features = embed_dim),
            nn.Dropout(p=dropout),
            )
    def forward(self,x):
        x = self.normlayer(x)
        x = self.mlp(x)

        return x
    
class TransformerEncoderBlockMyself(nn.Module):
    def __init__(self,
                 embedding_dim:int,
                 num_heads:int,
                 mlp_size:int,
                 mlp_drop:float,
                 atten_drop:float):
        super().__init__()

        #MSA block
        self.msa_block = MSABlock(embedding_dim=embedding_dim,
                                  num_heads=num_heads,
                                  attention_drop=atten_drop)
        #MLP block
        self.mlp_block = MLPBlock(embed_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_drop)
        
    def forward(self,x):

        #create residual connection for MSA block
        x = self.msa_block(x) + x

        #create residual connection for MLP block
        x = self.mlp_block(x) + x

        return x

class MyViT(nn.Module):
    def __init__(self,
                 img_size:int=48,
                 int_channels:int=1,
                 patch_size:int=4,
                 num_transformer_layers:int=12,
                 embedding_dim:int=256,
                 mlp_size:int=3072,
                 num_heads:int=16,
                 atten_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_class:int=7):
        super().__init__()

        assert img_size % patch_size == 0,f"Image size must divisable by patch size,image size:{img_size},patch size:{patch_size}"
        
        #calculate num of patches
        self.num_patches = (img_size**2) // (patch_size**2)
        #create learnable class embedding 
        self.class_embedding = nn.Parameter(data = torch.randn(1,1,embedding_dim),
                                           requires_grad = True)
        #create learnable position
        self.position_embedding = nn.Parameter(data = torch.rand(1,self.num_patches+1,embedding_dim),
                                               requires_grad = True)
        #create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        #create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=int_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        #create Transformer encoder   "*" means all
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlockMyself(embedding_dim=embedding_dim,
                                                                                 num_heads=num_heads,
                                                                                 mlp_size=mlp_size,
                                                                                 mlp_drop=mlp_dropout,
                                                                                 atten_drop=atten_dropout) for _ in range(num_transformer_layers)]) 
        #create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = embedding_dim),
            nn.Linear(in_features = embedding_dim,
                      out_features = num_class)
        )

    def forward(self,x):

        batch_size = x.shape[0]#batch first

        #create class token to match the batch size
        class_token = self.class_embedding.expand(batch_size,-1,-1) # "-1" means to infer the dimension (try this line on its own)

        x = self.patch_embedding(x)

        #concat class embedding and patch embedding
        x = torch.cat((class_token,x),dim=1)

        #add position embedding
        x = self.position_embedding + x

        #embedding dropout
        x = self.embedding_dropout(x)

        #transformer encoder
        x = self.transformer_encoder(x)

        #classifier
        x = self.classifier(x[:,0])# 0 index is class 

        return x
#--------------------------------------------------------------------------pretrained transformer model--------------------------------------------------------------------------

def PretrainedViTModel():
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False
    
    # Change the classifier head (set the seeds to ensure same initialization with linear head)
# 修改模型的输入层
    pretrained_vit.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16, bias=True)
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=7)

    return pretrained_vit
#--------------------------------------------------------------------------pytorch transformer model--------------------------------------------------------------------------

torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = 256,#hidden size D
                                                             nhead = 16,
                                                             dim_feedforward = 3072,#MLP size
                                                             dropout = 0.1,
                                                             activation = 'gelu',
                                                             batch_first = True,
                                                             norm_first = True)#Norm first after MLP/MSA?
#--------------------------------------------------------------------------summary model--------------------------------------------------------------------------

# mymodel = build_efficientnet_b2()

# summary(model=mymodel, 
#         # input_size=(512, 1, 48, 48), # make sure this is "input_size", not "input_shape"
#         input_size=(32,1,224,224), # make sure this is "input_size", not "input_shape"

#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# ) 
