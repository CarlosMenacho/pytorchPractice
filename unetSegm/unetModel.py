import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
        super(CNNBlock, self ).__init__()
        
        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        
    def forward(self, x):
        x = self.seq_block(x)
        return x
    
class CNNBlocks(nn.Module):
    def __init__(
        self,n_conv, in_channels, out_channels, padding):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            in_channels = out_channels
            
    def forward(self,x):
        for layer in self.layers: 
            x = layer(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding , downhill = 4):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()
        
        for _ in range(downhill):
            self.enc_layers += [
                CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
                nn.MaxPool2d(2,2)                
            ]
            in_channels = out_channels
            out_channels *= 2
            
        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding))
        
    def forward(self,x):
        route_connection = []
        
        for layer in self.enc_layers: 
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x=layer(x)
        return x, route_connection
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, exit_channels, padding, uphill = 4):
        super(Decoder,self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()
        
        for i in range(uphill):
            self.layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                CNNBlocks(n_conv = 2, in_channels = in_channels, out_channels = out_channels, padding=padding)
            ]
            in_channels //=2
            out_channels //=2
            
        self.layers.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=padding)
        )
    def forward(self, x, routes_connection):
        routes_connection.pop(-1)
        
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):                
                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[2])
                x =  torch.cat([x,routes_connection.pop(-1)], dim=1)
                x=layer(x)
            else: 
                x=layer(x)
        return x

class UNET(nn.Module):
    def __init__(self, in_channels, first_out_channels, exit_channels, downhills, padding =  0 ):
        super(UNET, self).__init__()
        self.encoder = Encoder(in_channels, first_out_channels, padding=padding, downhill=downhills)
        self.decoder = Decoder(first_out_channels*(2**downhills), first_out_channels*(2**(downhills-1)), exit_channels, padding=padding, uphill=downhills)
        
    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out,routes)
        return out    