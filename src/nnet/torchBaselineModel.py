import torch
import torch.nn as nn
import copy

from src.nnet.wavelet import MaxEnergySelector

class ChannelAttention(nn.Module):
    
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B, _, _, _ = x.size()

        avg = self.avg_pool(x).view(B, -1)
        max = self.max_pool(x).view(B, -1)

        avg_out = self.shared_mlp(avg)
        max_out = self.shared_mlp(max)

        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return x * out.expand_as(x)
    
    

class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # Channels first
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out.expand_as(x)
    

class CBAMBlock(nn.Module):

    
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    
    def __init__(self, num_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class BaselineModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        """
        VGGnet-like Model Architecture
    
        """
        super(BaselineModel, self).__init__()

        self.convblock1 = nn.Sequential(
            MaxEnergySelector(num_selected_channels=num_channels), 
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3,3), stride=(1,1), padding = 1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding = 1),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.ReLU()
        )
      
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = 1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride = (1,1), padding = 1),
            CBAMBlock(num_channels=64, reduction_ratio=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.ReLU(),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = 1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = 1),
            CBAMBlock(num_channels=128, reduction_ratio=8,kernel_size=5),
            nn.MaxPool2d(kernel_size=(5,5)),
            nn.ReLU(),
        )


        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=1024), # change this if wavelet transform is used and number of channels changes
            nn.Dropout(p=0.15),
            nn.Linear(in_features=1024, out_features=num_classes)
        )


    def forward(self, x):
        """ Forward pass """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

class EarlyStopping:
    """
    Custom early stopping module to prevent overfitting
    """
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.status = f"Early stopping triggered after {self.counter} epochs. Saving best weights: {self.restore_best_weights}"
                if self.restore_best_weights == True:
                    model.load_state_dict(self.best_model)
                return self.early_stop
        return self.early_stop