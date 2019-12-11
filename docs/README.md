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

You can find demo on the Colab. See the link at the top.



