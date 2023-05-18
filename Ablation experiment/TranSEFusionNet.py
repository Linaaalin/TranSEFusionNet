import torch
from torch import nn
import numpy as np
import copy
import math

__all__ = ['UNet', 'NestedUNet','U_Net','AttU_Net','TransattU_Net','ResUnet']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):   #[8,3,96,96]
        out = self.conv1(x) #[8,32,96,96]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,**kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])  #3,32,32
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])    #32,64,64
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])    #64,128,128
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])    #128,256,256
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])    #256,512,512

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3]) #768,256,256
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2]) #384,128,128
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1]) #192,64,64
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0]) #96,32,32

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, num_classes, input_channels=1, deep_supervision=False):  #num_classes, input_channels=3, deep_supervision=False
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.deep_supervision = deep_supervision

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(input_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out



class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs ):#num_classes, input_channels=1
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.deep_supervision = deep_supervision

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(input_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out



class Sblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Sblock, self).__init__()

        self.sconv = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, d, p):  #d[512,32,32]  p[512,32,32]
        t = torch.cat((d,p), dim=1)
        t = self.sconv(t)
        t = self.relu(t + p)
        return t


class SelfAttention(nn.Module):
    def __init__(self, num_heads=12, embed_dim=768, dropout=0):
        super().__init__()
        self.num_attention_heads = 12  #12
        self.attention_head_size = 64#int(embed_dim / num_heads)  #768/12 = 64
        self.all_head_size = 768  #12*64 = 768
        #self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)  #0.0
        self.proj_dropout = nn.Dropout(dropout)  #0.0

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x): #[4,256,768]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #(4,256,12,64)?
        x = x.view(*new_x_shape) #[4,256,12,64]
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states): #[4,256,768]
        mixed_query_layer = self.query(hidden_states) #[4,256,768]
        mixed_key_layer = self.key(hidden_states)  #[4,256,768]
        mixed_value_layer = self.value(hidden_states)  #[4,256,768]

        query_layer = self.transpose_for_scores(mixed_query_layer)  #[4,12,256,64]
        key_layer = self.transpose_for_scores(mixed_key_layer)   #[4,12,256,64]
        value_layer = self.transpose_for_scores(mixed_value_layer)   ##[4,12,256,64]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #attention_scores4,12,256,256
        #query_layer:4,12,256,64  key_layer:4,12,256,64
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  #[4,12,256,256]
        attention_probs = self.softmax(attention_scores)  #[4,12,256,256]
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)   #[4,12,256,256]

        context_layer = torch.matmul(attention_probs, value_layer)   #[4,12,256,64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  #4,256,12,64
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#4,256,768
        context_layer = context_layer.view(*new_context_layer_shape)  #4,256,768
        attention_output = self.out(context_layer)  #4,256,768
        attention_output = self.proj_dropout(attention_output)#4,256,768
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, in_features=768 ,mid_features=3072, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x): #[4,256,768]
        x = self.fc1(x)    #[4,256,3072]
        x = self.act(x)    #[4,256,3072]
        x = self.drop(x)  #4,256,3072
        x = self.fc2(x)   #[4,256,768]
        x = self.drop(x)  #[4,256,768]
        return x

class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim=768, cube_size=(256,256), patch_size=16, dropout=0.1,
                 n_patches=256):
        super().__init__()
        # self.n_patches = int((cube_size[0] * cube_size[1] ) / ( patch_size * patch_size))
        self.n_patches = n_patches
        # self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=1024, out_channels=embed_dim,
                                          kernel_size=(1,1), stride=(1,1))
        # self.conv = nn.Conv2d(input=1024,out=1,kernel_size=1,stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # 输入x是：  [4,1024,16,16]
        # print(x.size)
        # x = self.conv(x)
        x = self.patch_embeddings(x)  # 经过Patch_embedding函数（就是个卷积）后变成  [4,768,16,16]

        x = x.flatten(2)  # 把后面合并后变成[4,768,256]
        x = x.transpose(-1,-2)  # 在做转换后变成[4,256,768]
        #x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings  #[4,256,768] 然后和position_embeddings做和，这个position_embeddings就是
        # Transformer中的Position_encoding，这里只是用Torch.zero来定义，nn.Parameter表示这个参数可学习
        # （Position_encoding有用来学习得到和手工设置两个方法，这里用的是前者）
        embeddings = self.dropout(embeddings)#[4,256,768]
        return embeddings

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768 , num_heads=12, dropout=0.1 , cube_size=(256,256), patch_size=(16,16),
                 mlp_dim=768):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        #self.mlp_dim = int((cube_size[0] * cube_size[1] ) / ( patch_size * patch_size))
        self.mlp_dim = mlp_dim
        self.mlp = Mlp(embed_dim, 3072)
        self.attn = SelfAttention(num_heads, embed_dim, dropout=0)

    def forward(self, x):  # [4,256,768]看网络结构图更好理解
        h = x  # 先保留了x的值，为了做跳跃连接  [4,256,768]
        x = self.attention_norm(x)  # 然后对x做attention_norm [4,256,768]
        x, weights = self.attn(x)  # 然后对x做multi_head_Attention
        x = x + h  # 接着是Residual   #h=[4,256,768]
        h = x  # 保留现在的x，为了做第二个跳跃连接
        x = self.mlp_norm(x)  # 然后做Norm，接到MLP层（就是FC）
        x = self.mlp(x)
        x = x + h  #[4,256,768] 最后做Residual   Tensor维度的信息：进去x是[4,256,768]，后面每一步操作都是[4,256,768]
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dims, embed_dim, cube_size=(256,256), patch_size=(16,16), num_heads=12, num_layers=12,
                 dropout=0.1, extract_layers=12, vis=0):
        super().__init__()
        self.vis = vis
        self.embeddings = Embeddings(input_dims, embed_dim, cube_size, patch_size, dropout)  # 首先做Embedding
        self.layer = nn.ModuleList()  # layer用了nn.ModuleList定义，其中包含了12个Transformer Block
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(12):  # num_layers=12
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x): #(4,1024,16,16)
        extract_layers = []
        hidden_states = self.embeddings(x)  #[4,256,768]

        for depth, layer_block in enumerate(self.layer):  # forward函数中也做了12次
            hidden_states, _ = layer_block(hidden_states)  #[4,256,768]
            # if depth + 1 in 12:
        # extract_layers.append(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded
    # def forward(self,hidden_states):
    #     attn_weights = []
    #     hidden_states = self.embeddings(hidden_states)
    #     for layer_block in self.layer:
    #         hidden_states,weights = layer_block(hidden_states)
    #         if self.vis:
    #             attn_weights.append(weights)
    #     encoded = self.encoder_norm(hidden_states)
    #     return  encoded



class Conv2dReLU(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, input_ch, output_ch):
        super(Conv2dReLU, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  #[4,768,16,16]
        x = self.conv(x)
        return x


class Reshape(nn.Module):
    def __init__(self,input_ch,output_ch):
        super(Reshape, self).__init__()

        self.Conv2dRelu = Conv2dReLU(
                input_ch,
                output_ch,
                # kernel_size,
                # padding,
                # stride,
            )
    def forward(self, hidden_states):  #[4,256,768]
        B, n_patch, hidden = hidden_states.size()  #[4,256,768]
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))#16,16
        x = hidden_states.permute(0, 2, 1) #[4,768,256]
        x = x.contiguous().view(B, hidden, h, w) #[4,768,16,16]
        x = self.Conv2dRelu(x)  #[4,512,16,16]
        return x


class TransattU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self,num_classes=1,input_channels=3, deep_supervision=False, input_dims=1, embed_dim=768,patch_size=4,
                 num_heads=12,dropout=0.1,input_ch=768 ,output_ch=1024):
        super(TransattU_Net, self).__init__()

        self.deep_supervision = deep_supervision
        self.input_dims = input_dims
        # self.outputdim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers =[3,6,9,12]
        # self.input_ch = input_ch
        # self.output_ch = output_ch
        self.tranformer = \
            Transformer(
                input_dims,
                embed_dim=768,
                cube_size=(256,256),
                patch_size=(16,16),
                num_heads=12,
                dropout=0.1,
                extract_layers=12,
            )
        self.reshape = Reshape(
            input_ch=768,
            output_ch=1024,
        )

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(input_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.tranformer = Transformer(input_dims, embed_dim, patch_size, num_heads, dropout)
        self.reshape = Reshape(input_ch, output_ch)

        self.Up5 = up_conv(filters[4], filters[3])#64,128,256,512,1024
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.s1 = Sblock(in_ch=filters[4], out_ch=filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.s2 = Sblock(in_ch=filters[3], out_ch=filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.s3 = Sblock(in_ch=filters[2], out_ch=filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.s4 = Sblock(in_ch=filters[1], out_ch=filters[0])

        self.Conv = nn.Conv2d(filters[0],num_classes, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):#[4,1,256,256]
        e1 = self.Conv1(x)  #[4,64,256,256]

        e2 = self.Maxpool1(e1) #[4,64,128,128]
        e2 = self.Conv2(e2)   #[4,128,128,128]

        e3 = self.Maxpool2(e2)  #[4,128,64,64]
        e3 = self.Conv3(e3)    #[4,256,64,64]

        e4 = self.Maxpool3(e3)  #[4,256,32,32]
        e4 = self.Conv4(e4)    #[4,512,32,32]

        e5 = self.Maxpool4(e4)  #[4,512,16,16]
        e5 = self.Conv5(e5)     #[4,1024,16,16]

        y = self.tranformer(e5)  #[4,256,768]
        y1 = self.reshape(y)    #[4,1024,16,16]

        # print(x5.shape)
        d5 = self.Up5(y1)  #[4,512,32,32]
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)  #[4,512,32,32]
        d5 = torch.cat((x4, d5), dim=1)  #[4,1024,32,32]
        d5 = self.Up_conv5(d5)   #[4,512,32,32]

        a1 = self.Up5(y1)       #[4,512,32,32]
        s1 = self.s1(d=a1, p=d5)

        d4 = self.Up4(d5)  #[4,256,64,64]
        x3 = self.Att4(g=d4, x=e3)  #[4,256,64,64]
        d4 = torch.cat((x3, d4), dim=1)  #[4,512,64,64]
        d4 = self.Up_conv4(d4)  #[4,256,64,64]

        a2 = self.Up4(s1)
        s2 = self.s2(d=a2, p=d4)


        d3 = self.Up3(d4)   #[4,128,128,128]
        x2 = self.Att3(g=d3, x=e2)   #[4,128,128,128]
        d3 = torch.cat((x2, d3), dim=1)  #[4,256,128,128]
        d3 = self.Up_conv3(d3)  #[4,128,128,128]

        a3 = self.Up3(s2)
        s3 = self.s3(d=a3, p=d3)

        d2 = self.Up2(d3)  #[4,64,256,256]
        x1 = self.Att2(g=d2, x=e1)  #[4,64,256,256]
        d2 = torch.cat((x1, d2), dim=1)  #[4,128,256,256]
        d2 = self.Up_conv2(d2)  #[4,64,256,256]

        a4 = self.Up2(s3)
        s4 = self.s4(d=a4, p=d2)

        out = self.Conv(s4)

        # out = self.Conv(d2)  #[4,1,256,256]

        #  out = self.active(out)

        return out



class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,**kwargs ):
        super(ResUnet, self).__init__()

        filters = [64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output

