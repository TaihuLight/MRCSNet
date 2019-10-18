import torch
import torch.nn as nn
import numpy as np
    
B=32

# Reshape + Concat ==> get initial reconstructed image
# [Reshape each 1*B*B reconstructed vector to a B*B image block, then concatenate the blocks to the initial reconstructed image]

class ReshapeLayer(nn.Module):
    def __init__(self, *args):
        super(ReshapeLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        num_samples = x.size()[0]
        Bsize=self.shape[0]
# You just call the .view and .permute methods on the output you want to reshape in the forward function of the custom model.
# https://discuss.pytorch.org/t/difference-between-2-reshaping-operations-reshape-vs-permute/30749/4
        Xresd = []
        for dim0 in range(num_samples):            
            dc2=[]
            Xbatch=x[dim0,:,:,:]
            for dim2 in range(Xbatch.size()[1]):
                dc3=[]
                for dim3 in range(Xbatch.size()[2]):  
                    tmp=Xbatch[:,dim2,dim3]
                    tmp=tmp.view(Bsize,Bsize).permute(1,0) # view as 1-channel BxB image block[===maybe wrongn for permute****]
                    dc3.append(tmp)
                dctmp2=torch.cat(dc3,dim=1)
                dc2.append(dctmp2)
            catt=torch.cat(dc2,dim=0)  # view as a 1-channel image
            catt = catt.view(1,1,catt.size()[0],catt.size()[1])
            Xresd.append(catt)
            
        XresT=torch.cat(Xresd,dim=0)
        return XresT

    
class DWConvLeakyReLU_block(nn.Module):
    def __init__(self):
        super(DWConvLeakyReLU_block, self).__init__()
#       [Depthwise convolution in Pytorch] https://www.aiuai.cn/aifarm618.html
#       https://pytorch.org/docs/0.4.1/nn.html?highlight=nn%20conv2d#torch.nn.Conv2d
        self.dwtconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.conv2d11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)  
        self.leakyrelu = nn.LeakyReLU(0.001) 

    def forward(self, x):
        out = self.dwtconv(x)
        out = self.leakyrelu(out)
        out = self.conv2d11(out)
        out = self.leakyrelu(out)
        return out
    
class SORTRes_block(nn.Module):
    def __init__(self):
        super(SORTRes_block, self).__init__()        
        self.depthwconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
#         self.resconv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
#         self.leakyrelu = nn.LeakyReLU(0.4, inplace=True)  
#         self.leakyrelu = nn.LeakyReLU(0.001) 
        self.prelu = nn.PReLU(num_parameters=1, init=0.10) 
#         self.actv_swish = Swish() 

    def forward(self, x):
        out1 = self.depthwconv(x)
#         out2 = self.leakyrelu(out1)
        out2 = self.prelu(out1)        
        out3 = self.depthwconv(out2)
        
        out = x + out3
#         out = self.leakyrelu(out) 
        out = self.prelu(out) 
                
        return out

# The model
class CSNet_init(nn.Module):    
    
    def __init__(self, csrate):
        '''
        - csrate_Mdim is computed according to MR[0.01,0.04,0.1,0.25,0.4,0.5]
        '''
        super().__init__()
        
        csnb=int(float(csrate)*B*B)        
        self.cssample = nn.Conv2d(in_channels=1, out_channels=csnb, kernel_size=B, stride=B, padding=0, bias=False)
        self.initresc = nn.Conv2d(in_channels=csnb, out_channels=B*B, kernel_size=1, stride=1, padding=0, bias=False)
        self.reshape_concat = self.make_ShapeConcatLayer(ReshapeLayer,[B,B],1)
       
    def make_ShapeConcatLayer(self, block, paras, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(paras))
        return nn.Sequential(*layers) 
    
    def make_Block(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        outcsy = self.cssample(x)
        out = self.initresc(outcsy)
        
        out_initrec = self.reshape_concat(out)
#         print(out_initrec.size())
             
        return out_initrec, outcsy, out
       
# The model
class CSNet(nn.Module):    
    
    def __init__(self, csrate):
        '''
        - csrate_Mdim is computed according to MR[0.01,0.04,0.1,0.25,0.4,0.5]
        '''
        super().__init__()
        
        csnb=int(float(csrate)*B*B)        
        self.cssample = nn.Conv2d(in_channels=1, out_channels=csnb, kernel_size=B, stride=B, padding=0, bias=False)
        self.initresc = nn.Conv2d(in_channels=csnb, out_channels=B*B, kernel_size=1, stride=1, padding=0, bias=False)
        self.reshape_concat = self.make_ShapeConcatLayer(ReshapeLayer,[B,B],1)
                
        self.firstresc = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) 
#         self.leakyrelu = nn.LeakyReLU(negative_slope=0.001) 

#         self.leakyrelu = nn.LeakyReLU(0.4, inplace=True) 
        self.prelu = nn.PReLU(num_parameters=1, init=0.10)
#         self.actv_swish = Swish() 
        
#         self.blkresc = self.make_Block(DWConvLeakyReLU_block) 
        self.blkresc = self.make_Block(SORTRes_block)  
        self.lastresc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
       
    def make_ShapeConcatLayer(self, block, paras, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(paras))
        return nn.Sequential(*layers) 
    
    def make_Block(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
#         print(x.size())
        outcsy = self.cssample(x)
#         print(outcsy.size())
        out = self.initresc(outcsy)
        
        out_initrec = self.reshape_concat(out)
#         print(out_initrec.size())
        
        out= self.firstresc(out_initrec)
#====================================
#         out= self.leakyrelu(out)
        out= self.prelu(out)
#====================================      
        
        num_of_layer=4
        for _ in range(num_of_layer):
            out= self.blkresc(out)
            
        out = self.lastresc(out+out_initrec)
             
        return out, outcsy, out_initrec
    

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

def weight_init(m):
#     print("layer ---------",m)
    if type(m) == nn.Conv2d:
#       “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015)
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.ConvTranspose2d):
        c1, c2, h, w = m.weight.data.size()
        weight = get_upsample_filter(h)
        m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
        if m.bias is not None:
            m.bias.data.zero_()
     

        
    
    
