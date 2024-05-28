import torch
import torch.nn as nn
import timm
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class UNet(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = timm.create_model(backbone_name, features_only=True, pretrained=pretrained)
        
        encoder_channels = self.encoder.feature_info.channels() # Get the number of channels at each stage of the encoder
        decoder_channels = [256, 128, 64, 32, 16]
        
        self.decoder4 = self._decoder_block(encoder_channels[3], decoder_channels[0])
        self.decoder3 = self._decoder_block(encoder_channels[2] + decoder_channels[0], decoder_channels[1])
        self.decoder2 = self._decoder_block(encoder_channels[1] + decoder_channels[1], decoder_channels[2])
        self.decoder1 = self._decoder_block(encoder_channels[0] + decoder_channels[2], decoder_channels[3])
        
        self.final_conv = nn.Conv2d(decoder_channels[3], n_classes, kernel_size=1)
        
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
    def forward(self, x):
        encoder_features = self.encoder(x)
        
        decoder4 = self.decoder4(encoder_features[3])
        decoder3 = self.decoder3(torch.cat([decoder4, encoder_features[2]], dim=1))
        decoder2 = self.decoder2(torch.cat([decoder3, encoder_features[1]], dim=1))
        decoder1 = self.decoder1(torch.cat([decoder2, encoder_features[0]], dim=1))
        
        out = self.final_conv(decoder1)
        
        return out
    
def create_unet_model(pretrained=True, backbone_name="resnet34", n_classes=1):
    # Example usage:
    model = UNet(backbone_name=backbone_name, pretrained=pretrained, n_classes=n_classes)
    return model
    
if __name__ == "__main__":
    model = create_unet_model()
    x = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = model(x)
    print(output.shape)  # Should be [1, 1, 224, 224]