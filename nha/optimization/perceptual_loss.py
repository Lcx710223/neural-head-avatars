"""
Code heavily inspired by https://github.com/JustusThies/NeuralTexGen/blob/master/models/VGG_LOSS.py
"""
###LCX20250702怀疑感知损失没有调用，设置LCX-DEBUG:
import torch
from torchvision import models
from torchvision.transforms import Normalize
from collections import namedtuple
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parents[2]/"deps")))
from InsightFace.recognition.arcface_torch.backbones import get_model

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, X):
        """
        assuming rgb input of shape N x 3 x H x W normalized to -1 ... +1
        :param X:
        :return:
        """
        X = self.normalize(X * .5 + .5)

        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class ResNet18(torch.nn.Module):
    def __init__(self, weight_path):
        super(ResNet18, self).__init__()
        net = get_model("r18")
        print("[PERCDEBUG] loading weights from", weight_path)
        res = net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
        print("[PERCDEBUG] missing_keys:", res.missing_keys)
        print("[PERCDEBUG] unexpected_keys:", res.unexpected_keys)
        net.eval()
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.prelu = net.prelu
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        print("[PERCDEBUG][ResNet18] forward called")
        self.eval()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        relu1_2 = x.clone()
        x = self.layer2(x)
        relu2_2 = x.clone()
        x = self.layer3(x)
        relu3_3 = x.clone()
        x = self.layer4(x)
        relu4_3 = x.clone()
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        # 调试输出每一层均值和方差
        #print(f"[PERCDEBUG][ResNet18] relu1_2 mean={relu1_2.mean().item():.6f}, std={relu1_2.std().item():.6f}")
        #print(f"[PERCDEBUG][ResNet18] relu2_2 mean={relu2_2.mean().item():.6f}, std={relu2_2.std().item():.6f}")
        #print(f"[PERCDEBUG][ResNet18] relu3_3 mean={relu3_3.mean().item():.6f}, std={relu3_3.std().item():.6f}")
        #print(f"[PERCDEBUG][ResNet18] relu4_3 mean={relu4_3.mean().item():.6f}, std={relu4_3.std().item():.6f}")
        return out

class ResNetLOSS(torch.nn.Module):
    def __init__(self, criterion=torch.nn.L1Loss(reduction='mean')):
        super(ResNetLOSS, self).__init__()
        self.model = ResNet18("assets/InsightFace/backbone.pth")
        self.model.eval()
        self.criterion = criterion
        self.criterion.reduction = "mean"

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, fake, target, content_weight=1.0, style_weight=1.0):
        print("[LCX-DEBUG] ResNetLOSS.forward called")
        vgg_fake = self.model(fake)
        vgg_target = self.model(target)
        ###LCX20250702 调试输出：每一层的均值和方差
        for i, (f, t) in enumerate(zip(vgg_fake, vgg_target)):
            print(f"[LCX-DEBUG] layer {i} fake mean={f.mean().item():.6f}, std={f.std().item():.6f}, target mean={t.mean().item():.6f}, std={t.std().item():.6f}")
        content_loss = self.criterion(vgg_target.relu2_2, vgg_fake.relu2_2)
        print("[LCX-DEBUG] content_loss:", content_loss.item())
        gram_style = [gram_matrix(y) for y in vgg_target]
        style_loss = 0.0
        for j, (ft_y, gm_s) in enumerate(zip(vgg_fake, gram_style)):
            gm_y = gram_matrix(ft_y)
            l = self.criterion(gm_y, gm_s)
            style_loss += l
            print(f"[LCX-DEBUG] style_loss layer {j}:", l.item())
        print("[LCX-DEBUG] total style_loss:", style_loss)
        total_loss = content_weight * content_loss + style_weight * style_loss
        print("[LCX-DEBUG] total perceptual loss:", total_loss.item() if hasattr(total_loss, "item") else total_loss)
        return total_loss
