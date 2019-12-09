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
import random

import utils


class Loader(torch.utils.data.Dataset):

    def __init__(self, training_files, decoder_ind, segment_length, mu_quantization, sampling_rate):

        audio_files = utils.files_to_list(training_files)

        self.audio_files = audio_files
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.segment_length = segment_length
        self.mu_quantization = mu_quantization
        self.sampling_rate = sampling_rate
        self.decoder_ind = decoder_ind

    def __getitem__(self, index):

        # Read audio        
        filename = self.audio_files[index]
        # Create torch tensor
        audio, sampling_rate = utils.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

            # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = utils.mu_law_encode(audio / utils.MAX_WAV_VALUE, self.mu_quantization)
        return audio, self.decoder_ind

    def __len__(self):
        return len(self.audio_files)
