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
import argparse
import json
import logging
import os
import random
import torch
from torch.utils.data import DataLoader

from loader import Loader
from model import Model
from utils import to_gpu

logging.basicConfig(level=logging.INFO)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = model_config["n_out_channels"]

    def forward(self, inputs, targets):
        """
        inputs are batch by num_classes by sample
        targets are batch by sample
        torch CrossEntropyLoss needs
            input = batch * samples by num_classes
            targets = batch * samples
        """
        targets = targets.view(-1)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.contiguous()
        inputs = inputs.view(-1, self.num_classes)
        return torch.nn.CrossEntropyLoss()(inputs, targets)


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())

    print("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    model_for_saving = Model(model_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train(output_directory, epochs, learning_rate, alpha, iters_per_checkpoint, num_workers, batch_size, pin_memory,
          seed, checkpoint_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    criterion = CrossEntropyLoss()
    domain_loss_criterion = torch.nn.NLLLoss()
    model = Model(model_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if one exists
    iteration = 1
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1

    train_files = ["train_files_{}.txt".format(i) for i in range(model_config["n_speakers"])]
    trainsets = [Loader(file, i, **data_config) for i, file in enumerate(train_files)]
    train_loaders = [DataLoader(trainset, num_workers=num_workers, batch_size=batch_size, shuffle=True, sampler=None,
                                pin_memory=pin_memory, drop_last=True) for trainset in trainsets]
    lengths = [len(i) for i in train_loaders]

    # Get output_directory ready
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory)

    model.train()
    epoch_offset = max(0, int(iteration / max(lengths))) + 1
    iterators = [iter(cycle(loader)) for loader in train_loaders]

    # ================ MAIN TRAINING LOOP! ===================
    reduced_recon_loss = 0.0
    reduced_domain_loss = 0.0
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for _ in range(max(lengths)):
            random.shuffle(iterators)
            for iterator in iterators:
                model.zero_grad()
                audio, decoder_ind = next(iterator)
                audio = to_gpu(audio)
                audio_pred, domain_output = model(audio, decoder_ind[0], alpha)

                domain_loss = domain_loss_criterion(domain_output, decoder_ind.long().cuda())
                recon_loss = criterion(audio_pred, audio)
                loss = recon_loss + domain_loss
                reduced_recon_loss += recon_loss.item()
                reduced_domain_loss += domain_loss.item()

                loss.backward()
                optimizer.step()

                print("{}:\trecon_loss: {:.9f} \t domain_loss: {:.9f}".format(iteration, recon_loss.item(),
                                                                              domain_loss.item()))

                if (iteration % 100 == 0):
                    print("\navg_recon_loss: {:.9f}\tavg_domain_loss: {:.9f}\n".format(reduced_recon_loss / 100,
                                                                                       reduced_domain_loss / 100))
                    reduced_recon_loss = 0.0
                    reduced_domain_loss = 0.0

                if (iteration % iters_per_checkpoint == 0):
                    checkpoint_path = "{}/wavenet_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

                iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    if torch.cuda.device_count() > 1:
        print("WARNING: Multiple GPUs detected but no distributed group set")
        print("Only running 1 GPU.")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(**train_config)
