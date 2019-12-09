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
import math

from conv import Conv


class ResidualBlock(torch.nn.Module):
    def __init__(self, en_residual_channels, en_dilation_channel, n_layers, max_dilation):
        super(ResidualBlock, self).__init__()

        self.n_layers = n_layers
        self.en_dilate_layers = torch.nn.ModuleList()
        self.en_residual_layers = torch.nn.ModuleList()

        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(self.n_layers):
            dilation = 2 ** (i % loop_factor)

            self.en_dilate_layers.append(
                Conv(
                    en_residual_channels,
                    en_dilation_channel,
                    kernel_size=2,
                    dilation=dilation,
                    w_init_gain='tanh',
                    is_causal=True
                )
            )

            self.en_residual_layers.append(
                Conv(
                    en_dilation_channel,
                    en_residual_channels,
                    kernel_size=1
                )
            )

    def forward(self, sample):
        for i in range(self.n_layers):
            current = sample
            sample = torch.nn.functional.relu(sample, True)
            sample = self.en_dilate_layers[i](sample)
            sample = torch.nn.functional.relu(sample, True)
            sample = self.en_residual_layers[i](sample)
            sample = sample + current
        return sample


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        n_in_channels = 256
        n_residual_channels = 64
        max_dilation = 128
        n_layers = 16
        self.embed = torch.nn.Embedding(n_in_channels, n_residual_channels)
        self.en_residual = ResidualBlock(n_residual_channels, n_residual_channels, n_layers, max_dilation)
        self.conv1x1 = Conv(n_residual_channels, n_residual_channels, kernel_size=1, w_init_gain='relu')
        self.avg_pooling_layer = torch.nn.AvgPool1d(kernel_size=800, stride=200)

    def forward(self, sample):
        sample = self.embed(sample)
        sample = sample.transpose(1, 2)
        sample = self.en_residual(sample)
        sample = self.conv1x1(sample)
        sample = self.avg_pooling_layer(sample)

        return sample
