# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.
#  Copyright (c) 2019, Hubert Siuzdak
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import torch

from encoder import Encoder
from decoder import Decoder
from conv import Conv


class Discriminator(torch.nn.Module):
    def __init__(self, num_speakers):
        super(Discriminator, self).__init__()

        self.discriminator_conv1 = Conv(64, 64, kernel_size=4, stride=2)
        self.discriminator_conv2 = Conv(64, 64, kernel_size=4, stride=2)
        self.discriminator_conv3 = Conv(64, 64, kernel_size=2)
        self.fc1 = torch.nn.Linear(64 * 20, 64)
        self.fc2 = torch.nn.Linear(64, num_speakers)
        self.discriminator_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.discriminator_conv1(x))
        x = torch.nn.functional.relu(self.discriminator_conv2(x))
        x = torch.nn.functional.relu(self.discriminator_conv3(x))
        x = x.view(-1, 64 * 20)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return self.discriminator_softmax(x)


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Model(torch.nn.Module):
    def __init__(self, model_config):
        num_speakers = model_config["n_speakers"]
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.domain_classifier = Discriminator(num_speakers)
        self.decoders = torch.nn.ModuleList([Decoder(**model_config) for _ in range(num_speakers)])

    def forward(self, sample, n_model, alpha):
        encoded = self.encoder(sample)
        reverse_feature = ReverseLayer.apply(encoded, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        decoded = self.decoders[n_model](encoded, sample)
        return decoded, domain_output

    def get_latent_input(self, features):
        with torch.no_grad():
            encoded = self.encoder(features)
        return encoded
