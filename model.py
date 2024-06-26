import torch
from torch import nn
import torch.nn.functional as F
import timm


class EZVSL(nn.Module):
    def __init__(self, tau, dim):
        super(EZVSL, self).__init__()
        self.tau = tau

        # Vision model (ViT)
        self.imgnet = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.img_proj = nn.Conv2d(768, 2048, kernel_size=(1, 1))

        # Audio model (ViT)
        self.aud_initial = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.aud_proj_1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.audnet = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.aud_proj_2 = nn.Conv2d(768, 1, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.AdaptiveMaxPool2d((1, None))
        self.aud_proj_3 = nn.Linear(self.audnet.num_features, 2048)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for net in [self.img_proj, self.aud_proj_1, self.aud_proj_2, self.aud_proj_3]:
            nn.init.trunc_normal_(net.weight, mean=0.0, std=0.01)
            nn.init.zeros_(net.bias)

    def forward(self, image, audio):
        # Image
        img = self.imgnet.forward_features(image)
        img = img.transpose(1, 2)[:, :, 1:].view(-1, 768, 14, 14)
        img = self.img_proj(img)
        img = F.normalize(img, dim=-3)

        # Audio
        aud = self.aud_initial(audio) #in(N,1,257,276) out(N,3,257,276)
        aud = F.interpolate(aud, size=(224, 224), mode='bilinear', align_corners=False)  #(N,3,224,224)
        aud = self.aud_proj_1(aud)  #(N,3,224,224)
        aud = self.audnet.forward_features(aud)  #(N,197,768)
        aud = self.maxpool(aud).squeeze(dim=-2) #(N,768)
        aud = self.aud_proj_3(aud) #(N,2048)
        aud = F.normalize(aud, dim=-1)

        return img, aud