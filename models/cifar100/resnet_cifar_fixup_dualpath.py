#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Resnet for CIFAR10

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

@inproceedings{DBLP:conf/cvpr/HeZRS16,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  pages     = {770--778},
  publisher = {{IEEE} Computer Society},
  year      = {2016}
}

"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter as P
from models.cifar100.routes import FeedforwardGateII, RLFeedforwardGateII

from distiller.modules import EltwiseAdd, EltwiseMult

__all__ = ['resnet20_cifar100_fixup_dualpath', 'resnet32_cifar100_fixup_dualpath', 'resnet44_cifar100_fixup_dualpath', 'resnet56_cifar100_fixup_dualpath',
            'resnet20_cifar100_fixup_dualpath_rl', 'resnet32_cifar100_fixup_dualpath_rl', 'resnet44_cifar100_fixup_dualpath_rl', 'resnet56_cifar100_fixup_dualpath_rl']

NUM_CLASSES = 100

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, layer_index=0):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

        self.gain, self.biases = P(torch.ones(1,1,1,1)), nn.ParameterList([P(torch.zeros(1,1,1,1)) for _ in range(4)])
        self.add = EltwiseAdd(inplace=False)
        self.mul = EltwiseMult(inplace=False)
        self.layer_index = layer_index

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.add( x, self.biases[0] )
            out = self.add( self.conv1(out), self.biases[1] ) 
            out = self.add( self.relu(out), self.biases[2] )

        if self.block_gates[1]:
            out = self.add( self.mul( self.gain, self.conv2(out) ), self.biases[3] )

        if self.downsample is not None:
            residual = self.downsample(self.relu(x))

        return self.add(out, residual)

class DualpathBlock(nn.Module):
    expansion = 1

    def __init__(self, basic_block, gate, fullbit_block, inplanes, outplanes, return_gate_states=False, is_first_block=False):
        super(DualpathBlock, self).__init__()
        self.basic_block = basic_block
        self.gate = gate
        self.fullbit_block = fullbit_block

        self.return_gate_states = return_gate_states
        self.is_first_block = is_first_block

    def forward(self, inputs):
        x, gate_collector = inputs
        quantized = self.basic_block(x)
        mask, gprob = self.gate(quantized)

        if not self.is_first_block:
            fullbit = self.fullbit_block(x)
        else:
            fullbit = self.fullbit_block(x)

        out = (1-mask).expand_as(quantized) * quantized \
                + mask.expand_as(quantized) * fullbit

        if self.return_gate_states:
            gate_collector.append(mask.squeeze().data)

        return out, gate_collector

class ResNetCifarDualpath(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES, return_gate_states=False):
        self.return_gate_states = return_gate_states
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 64  # 64
        self.layer_index = 0
        super(ResNetCifarDualpath, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 64, layers[0], pool_size=32)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 128, layers[1], stride=2, pool_size=16)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 256, layers[2], stride=2, pool_size=8)

        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        self.init_weights()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, pool_size=32):
        downsample = downsample_1 = None
        gate = FeedforwardGateII
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            downsample_1 = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(
                block(
                basic_block = BasicBlock(layer_gates[0], self.inplanes, planes, stride, downsample, self.layer_index),
                gate = gate(pool_size=pool_size, channel=planes*block.expansion),
                fullbit_block = BasicBlock(layer_gates[0], self.inplanes, planes, stride, downsample_1, self.layer_index),
                inplanes = self.inplanes,
                outplanes = planes, 
                return_gate_states = self.return_gate_states,
                is_first_block = True)
            )
        self.layer_index += 1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                basic_block = BasicBlock(layer_gates[i], self.inplanes, planes, layer_index = self.layer_index),
                gate = gate(pool_size=pool_size, channel=planes*block.expansion),
                fullbit_block = BasicBlock(layer_gates[i], self.inplanes, planes, layer_index = self.layer_index),
                inplanes = self.inplanes,
                outplanes = planes, 
                return_gate_states = self.return_gate_states,
                is_first_block = False)
                )
            self.layer_index += 1

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x1, gate_collector_1 = self.layer1((x, []))
        x2, gate_collector_2 = self.layer2((x1, []))
        x3, gate_collector_3 = self.layer3((x2, []))

        x = self.relu(self.bn1(x3))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.return_gate_states:
            return [x, gate_collector_1 + gate_collector_2 + gate_collector_3]
        return x


def resnet20_cifar100_fixup_dualpath(**kwargs):
    model = ResNetCifarDualpath(DualpathBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar100_fixup_dualpath(**kwargs):
    model = ResNetCifarDualpath(DualpathBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar100_fixup_dualpath(**kwargs):
    model = ResNetCifarDualpath(DualpathBlock, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar100_fixup_dualpath(**kwargs):
    model = ResNetCifarDualpath(DualpathBlock, [9, 9, 9], **kwargs)
    return model



class ResNetCifarDualpathRL(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES, return_gate_states=False):
        self.return_gate_states = return_gate_states

        self.saved_actions = []
        self.rewards = []
        self.gate_instances = []

        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 64  # 64
        self.layer_index = 0
        super(ResNetCifarDualpathRL, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 64, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 256, layers[2], stride=2)

        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        self.init_weights()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = downsample_1 = None
        gate = RLFeedforwardGateII
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            downsample_1 = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(
                block(
                basic_block = BasicBlock(layer_gates[0], self.inplanes, planes, stride, downsample, self.layer_index),
                gate = gate(pool_size=pool_size, channel=planes*block.expansion),
                fullbit_block = BasicBlock(layer_gates[0], self.inplanes, planes, stride, downsample_1, self.layer_index),
                inplanes = self.inplanes,
                outplanes = planes, 
                return_gate_states = self.return_gate_states,
                is_first_block = True)
            )
        self.layer_index += 1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                basic_block = BasicBlock(layer_gates[i], self.inplanes, planes, layer_index = self.layer_index),
                gate = gate(pool_size=pool_size, channel=planes*block.expansion),
                fullbit_block = BasicBlock(layer_gates[i], self.inplanes, planes, layer_index = self.layer_index),
                inplanes = self.inplanes,
                outplanes = planes, 
                return_gate_states = self.return_gate_states,
                is_first_block = False)
                )
            self.layer_index += 1

        for l in layers:
            self.gate_instances.append(l.gate)

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x1, gate_collector_1 = self.layer1((x, []))
        x2, gate_collector_2 = self.layer2((x1, []))
        x3, gate_collector_3 = self.layer3((x2, []))

        x = self.relu(self.bn1(x3))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # collect all actions
        for inst in self.gate_instances:
            self.saved_actions.append(inst.saved_action)

        if self.return_gate_states:
            return [x, gate_collector_1 + gate_collector_2 + gate_collector_3]
        return x


def resnet20_cifar100_fixup_dualpath_rl(**kwargs):
    model = ResNetCifarDualpathRL(DualpathBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar100_fixup_dualpath_rl(**kwargs):
    model = ResNetCifarDualpathRL(DualpathBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar100_fixup_dualpath_rl(**kwargs):
    model = ResNetCifarDualpathRL(DualpathBlock, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar100_fixup_dualpath_rl(**kwargs):
    model = ResNetCifarDualpathRL(DualpathBlock, [9, 9, 9], **kwargs)
    return model
