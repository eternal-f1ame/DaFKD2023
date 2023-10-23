import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import SequenceSummary
from typing import Optional, Dict
from model.nlp.discriminator import Bone
from model.nlp.utils import ClassifierOutput, CustomAttention

def MTL(dom, class_num, pretrained=False, path=None, **kwargs):
    if dom == 'cv':
        model = MTLResNet(Bottleneck, [2,2,2], class_num, **kwargs)
    elif dom == 'nlp':
        from transformers import BertModel
        bert = BertModel.from_pretrained('bert-base-uncased')
        model = MTLBert(bert, class_num, **kwargs)
    else:
        raise NotImplementedError
    return model

class Reshape(nn.Module):
    def __init__(self,*args):
        super(Reshape,self).__init__()
        self.shape = args
    
    def forward(self,x):
        return x.view(x.size(0),-1)

class MTLResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, groups=1, width_per_group=64):
        super(MTLResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.sharedlayer = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True), # batchsize, 16, 28, 28
            self._make_layer(block, 16, layers[0]),
            self._make_layer(block, 32, layers[1], stride=2),
            self._make_layer(block, 64, layers[2], stride=2)# batchsize,256,7,7

        )

        self.classify = nn.Sequential(   
            nn.AdaptiveAvgPool2d((1, 1)) ,  
            Reshape(),
            nn.Linear(64 * block.expansion, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) ,  
            Reshape(),
            nn.Linear(64 * block.expansion,1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sharedlayer(x)
        out_c = self.classify(x)
        out_d = self.discriminator(x)
        return out_c,out_d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv3x3(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MTLBert(Bone):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 4,
        dropout_rate: Optional[float] = 0.15,
        ce_ignore_index: Optional[int] = -100,
        epsilon: Optional[float] = 1e-8,
        gan_training: bool = False,
        **kwargs,
    ):
        super(MTLBert, self).__init__()
        self.num_labels = num_labels
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout")
            else None
        )
        self.dropout = nn.Dropout(dropout_rate if classifier_dropout is None else classifier_dropout)
        if gan_training:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels + 1)
            self.discriminator = nn.Linear(self.encoder.config.hidden_size, 1)  # 1 for real, 0 for fake
        else:
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.epsilon = epsilon
        self.gan_training = gan_training
        if self.gan_training:
            print("Training with GAN mode on!")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        external_states: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> (ClassifierOutput, ClassifierOutput):

        outputs = self.safe_encoder(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state[:, 0]  # get CLS embedding
        if external_states is not None:
            sequence_output = torch.cat([sequence_output, external_states], dim=0)

        sequence_output_drop = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_drop)
        fake_probs, fake_logits = None, None
        if self.gan_training:
            logits_d = self.discriminator(sequence_output_drop)
            fake_logits = logits[:, [-1]]
            fake_probs = self.sigmoid(fake_logits)
            logits = logits[:, :-1]
            logits_d = logits_d[:, [-1]]
        probs = self.sigmoid(logits)
        probd = self.sigmoid(logits_d)

        loss_c = self.compute_loss(
            logits=logits, probs=probs, fake_probs=fake_probs, labels=labels, labeled_mask=labeled_mask
        )
        loss_d = self.compute_loss(logits=logits_d, fake_logits=fake_logits)
        return ClassifierOutput(
            loss=loss["real_loss"],
            fake_loss=loss["fake_loss"],
            logits=logits,
            fake_logits=fake_logits,
            probs=probs,
            fake_probs=fake_probs,
            hidden_states=sequence_output,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        fake_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        fake_probs: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.FloatTensor]:
        real_loss = torch.FloatTensor([0]).to(self.encoder.device)
        fake_loss = torch.FloatTensor([0]).to(self.encoder.device)
        if labels is not None:
            if labeled_mask is not None:
                labeled_mask = labeled_mask.bool()
                logits = logits[labeled_mask]
                labels = labels[labeled_mask]
            if logits.shape[0] > 0:
                real_loss = self.loss_fct(logits, labels.float())
        if self.gan_training:
            fake_loss = -torch.mean(torch.log(fake_probs + self.epsilon))
        return {"real_loss": real_loss, "fake_loss": fake_loss}
