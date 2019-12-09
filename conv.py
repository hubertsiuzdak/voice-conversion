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


class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, padding=0, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, padding=padding, bias=bias)

        torch.nn.init.xavier_uniform(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.kernel_size > 1:
            padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
            signal = torch.nn.functional.pad(signal, padding)
        return self.conv(signal)


def export_weights(self):
    """
    Returns a dictionary with tensors ready for nv_wavenet wrapper
    """
    model = {}
    # We're not using a convolution to start to this does nothing
    model["embedding_prev"] = torch.cuda.FloatTensor(self.n_out_channels,
                                                     self.n_residual_channels).fill_(0.0)

    model["embedding_curr"] = self.embed.weight.data
    model["conv_out_weight"] = self.conv_out.conv.weight.data
    model["conv_end_weight"] = self.conv_end.conv.weight.data

    dilate_weights = []
    dilate_biases = []
    for layer in self.dilate_layers:
        dilate_weights.append(layer.conv.weight.data)
        dilate_biases.append(layer.conv.bias.data)
    model["dilate_weights"] = dilate_weights
    model["dilate_biases"] = dilate_biases

    model["max_dilation"] = self.max_dilation

    res_weights = []
    res_biases = []
    for layer in self.res_layers:
        res_weights.append(layer.conv.weight.data)
        res_biases.append(layer.conv.bias.data)
    model["res_weights"] = res_weights
    model["res_biases"] = res_biases

    skip_weights = []
    skip_biases = []
    for layer in self.skip_layers:
        skip_weights.append(layer.conv.weight.data)
        skip_biases.append(layer.conv.bias.data)
    model["skip_weights"] = skip_weights
    model["skip_biases"] = skip_biases

    model["use_embed_tanh"] = False

    return model


def get_cond_input(self, features, speaker_id):
    """
    Takes in features and gets the 2*R x batch x # layers x samples tensor
    """
    self.eval()
    with torch.no_grad():
        sample = self.first_layer(features)
        for i, (dilation_layer, dense_layer) in enumerate(zip(self.en_dilation_layer_stack, self.en_dense_layer_stack)):
            current = sample
            sample = torch.nn.functional.relu(sample, True)
            sample = dilation_layer(sample)
            sample = torch.nn.functional.relu(sample, True)
            sample = dense_layer(sample)
            _, _, current_length = sample.size()
            current_in_sliced = current[:, :, -current_length:]
            sample = sample + current_in_sliced

        sample = self.bottleneck_layer(sample)
        sample = torch.nn.functional.relu(sample, True)

        speaker_id_embedding = self.speaker_embed(speaker_id)
        speaker_id_embedding = speaker_id_embedding.transpose(1, 2)

        sample = sample.add(speaker_id_embedding)

        cond_input = self.pre_upsample(sample)
        cond_input = self.post_layer(cond_input)
        cond_input = self.upsample(cond_input)

        cond_input = self.cond_layers(cond_input)
        cond_input = cond_input.view(cond_input.size(0), self.n_layers, -1, cond_input.size(2))
        '''
        # TODO(rcosta): trim conv artifacts. mauybe pad spec to kernel multiple
        cond_input = self.upsample(features)
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        cond_input = cond_input[:, :, :-time_cutoff]
        cond_input = self.cond_layers(cond_input).data
        cond_input = cond_input.view(cond_input.size(0), self.n_layers, -1, cond_input.size(2))
        # This makes the data channels x batch x num_layers x samples'''
        cond_input = cond_input.permute(2, 0, 1, 3)
    return cond_input
