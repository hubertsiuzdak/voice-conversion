_This project is obsolete and no longer maintained. If you're looking for a voice conversion model see [this comment](https://github.com/hubertsiuzdak/voice-conversion/issues/4#issuecomment-954865171)._
___
[![Open Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hubertsiuzdak/voice-conversion/blob/master/demo/Voice_Conversion.ipynb)
# Adversarial Voice Conversion

This repo contains implementation of voice conversion system based on WaveNet autoencoders. 
In order to convert speaker identity, multiple decoders are used so that each one corresponds to a particular voice.
The encoder is shared and adversarial domain adaptation technique is employed so that latent space produced by encoder remains speaker-invariant. 
No spectral features are used - the training is perfmormed end-to-end on waveforms. This approach is similar to [A Universal Music Translation Network](https://research.fb.com/publications/a-universal-music-translation-network/) introduced by Facebook AI Research.
This code is built on top of the [NV-Wavenet](https://github.com/NVIDIA/nv-wavenet) model which serve as a decoder in this architecture. 
The encoder was added thus creating full autoencoder pathway. Simple discriminator with the gradient reversal layer was added in order to achieve domain adaptation effect.

![Architecture](/docs/architecture.png)

## Usage

You can find demo on the Colab. See the [link](https://colab.research.google.com/github/hubertsiuzdak/voice-conversion/blob/master/demo/Voice_Conversion.ipynb). 

## Open Source License

```
Copyright (c) 2017, NVIDIA CORPORATION
Copyright (c) 2019, Hubert Siuzdak
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
## Citation
```
@inproceedings{adversarialvoiceconversion,
    title     = {Voice conversion using deep adversarial learning},
    author    = {Hubert Siuzdak, Jakub Gałka},
    booktitle = {9th Language & Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics},
    year      = {2019},
    pages     = {40-43}
}
```

