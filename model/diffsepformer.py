import os
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
import numpy as np
import csv
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast

from utils.tools import transformer_source, std_normal, calc_diffusion_step_embedding
from model.modules import Denoiser
from model.losses import MultiResolutionSTFTLoss

# Logger info
logger = logging.getLogger(__name__)


class Encoder_mix(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Encoder_mix, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        batch_size, signal_len, num_spks = x.shape
        x = x.view([batch_size, 1, signal_len, num_spks])
        if num_spks == 2:
            source_1, source_2 = x.split([1, 1], dim=3)
            source_1 = source_1.squeeze(dim=3)
            source_2 = source_2.squeeze(dim=3)
            source_1 = self.conv(source_1)
            source_2 = self.conv(source_2)
            y = torch.stack([source_1, source_2],dim=0)

        elif num_spks == 3:
            source_1, source_2, source_3= x.split([1, 1, 1], dim=3)
            source_1 = source_1.squeeze(dim=3)
            source_2 = source_2.squeeze(dim=3)
            source_3 = source_3.squeeze(dim=3)
            source_1 = self.conv(source_1)
            source_2 = self.conv(source_2)
            source_3 = self.conv(source_3)
            y = torch.stack([source_1, source_2, source_3],dim=0)

        return y


def swish(x):
    return x * torch.sigmoid(x)


def DiffNet(hparams):
    hparams["Encoder_mix"] = Encoder_mix(
        in_channels = hparams["Encoder"].conv1d.in_channels,
        out_channels = hparams["Encoder"].conv1d.out_channels,
        kernel_size = hparams["Encoder"].conv1d.kernel_size[0],
        stride = hparams["Encoder"].conv1d.stride[0],
        padding = hparams["Encoder"].conv1d.padding[0]
    )
    hparams['modules']['encoder_mix'] = hparams["Encoder_mix"]

    if hparams['use_denoiser']:
        hparams['Denoiser'] = Denoiser(hparams)
        hparams['modules']['denoiser'] = hparams['Denoiser']
    else:
        hparams["fc_t1"] = nn.Linear(
            hparams["step_embed"]["diffusion_step_embed_dim_in"],
            hparams["step_embed"]["diffusion_step_embed_dim_mid"]
        )
        hparams['modules']['fc_t1'] = hparams["fc_t1"]
        hparams["fc_t2"] = nn.Linear(
            hparams["step_embed"]["diffusion_step_embed_dim_mid"],
            hparams["step_embed"]["diffusion_step_embed_dim_out"]
        )
        hparams['modules']['fc_t2'] = hparams["fc_t2"]
        hparams["fc_t"] = nn.Linear(
            hparams["step_embed"]["diffusion_step_embed_dim_out"],
            hparams["Encoder"].conv1d.out_channels
        )
        hparams['modules']['fc_t'] = hparams["fc_t"]

    if hparams['diff_loss']=='mse':
        hparams['diff_loss'] = torch.nn.MSELoss(reduce=True, size_average=True)
    elif hparams['diff_loss']=='mae':
        hparams['diff_loss'] = torch.nn.L1Loss(reduce=True, size_average=True)
    elif hparams['diff_loss']=='pit':
        hparams['diff_loss'] = sb.nnet.losses.get_si_snr_with_pitwrapper
    elif hparams['diff_loss']=='stft_loss':
        hparams['diff_loss'] = MultiResolutionSTFTLoss(
            fft_sizes=hparams['stft_loss_param']['fft_sizes'],
            hop_sizes=hparams['stft_loss_param']['hop_sizes'],
            win_lengths=hparams['stft_loss_param']['win_lengths'],
            window=hparams['stft_loss_param']['window'],
        )
    for module_name in hparams['modules']:
        hparams['checkpointer'].add_recoverable(module_name, hparams['modules'][module_name])

    return hparams


# Define training procedure
class DiffSepformer(sb.Brain):
    def model(self, mix, diffusion_steps=None, input_targets=None):
        # Separation
        if input_targets == None:
            mix_w = self.hparams.Encoder(mix) # 1 10000
            est_mask = self.hparams.MaskNet(mix_w)
            mix_w = torch.stack([mix_w] * self.hparams.num_spks) #2 1 256 2499
            sep_h = mix_w * est_mask #2 1 256 1249

        # Defusing
        elif input_targets != None:
            if self.hparams.use_denoiser:
                mix_w = self.hparams.Encoder(mix)  # 1 10000
                est_mask = self.hparams.MaskNet(mix_w)
                mix_w = torch.stack([mix_w] * self.hparams.num_spks)  # 2 1 256 2499
                sep_h = mix_w * est_mask  # 2 1 256 1249
                input_targets = self.hparams.Encoder_mix(input_targets)
                sep_h = torch.cat(
                    [
                        self.hparams.Denoiser(sep_h[i], diffusion_steps, input_targets[i]).unsqueeze(0)
                        for i in range(self.hparams.num_spks)
                    ],
                    dim=0,
                )
            else:
                input_targets = self.hparams.Encoder_mix(input_targets)

                diffusion_steps = diffusion_steps.view(self.hparams.batch_size, 1)
                diffusion_step_embed = calc_diffusion_step_embedding(
                    diffusion_steps,
                    self.hparams.step_embed["diffusion_step_embed_dim_in"]
                )
                diffusion_step_embed = swish(self.hparams.fc_t1(diffusion_step_embed))
                diffusion_step_embed = swish(self.hparams.fc_t2(diffusion_step_embed))
                part_t = self.hparams.fc_t(diffusion_step_embed)
                part_t = ((part_t.unsqueeze(0)).repeat(self.hparams.num_spks, 1, 1)).unsqueeze(-1)  # 2 1 256 1
                input_targets = input_targets + part_t  # 2 1 256 1249

                mix_w = self.hparams.Encoder(mix)  # 1 10000
                est_mask = self.hparams.MaskNet(mix_w, input_targets)
                mix_w = torch.stack([mix_w] * self.hparams.num_spks)  # 2 1 256 2499
                sep_h = mix_w * est_mask  # 2 1 256 1249

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        ) #1 20000 2

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                    if self.hparams.use_wham_noise:
                        noise = noise.to(self.device)
                        len_noise = noise.shape[1]
                        len_mix = mix.shape[1]
                        min_len = min(len_noise, len_mix)

                        # add the noise
                        mix = mix[:, :min_len] + noise[:, :min_len]

                        # fix the length of targets also
                        targets = targets[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Gauss Noise
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                predictions = self.model(mix)
                self.compute_objectives(targets, predictions)
                input_targets=self.reverb(targets)

            Gauss = std_normal((self.hparams.batch_size, mix.shape[1], 1), self.hparams.num_spks)
            diffusion_steps = torch.randint(self.hparams.diffusion_hyperparams["T"], size=(self.hparams.batch_size, 1, 1))
            input_targets = transformer_source(diffusion_steps, self.hparams.diffusion_hyperparams, input_targets, z=Gauss)
            est_source = self.model(mix, diffusion_steps.view(self.hparams.batch_size), input_targets)

            self.diff_loss = torch.tensor(0.0).to(self.device)
            if self.hparams.diff_mode == 'multiple':
                T = self.hparams.diffusion_hyperparams["T"]
                for t in range(T - 1, -1, -1):
                    diffusion_steps = (t * torch.ones((self.hparams.batch_size, 1, 1)).type(torch.long))
                    transformer_predictions = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, est_source, z=Gauss)
                    transformer_targets = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, targets, z=Gauss)
                    self.diff_loss += self.compute_diff_loss(transformer_predictions, transformer_targets)
                self.diff_loss = self.diff_loss / T * self.hparams.diff_loss_coefficient

            elif self.hparams.diff_mode == 'single':
                transformer_predictions = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, est_source, z=Gauss)
                transformer_targets = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, targets, z=Gauss)
                self.diff_loss = self.compute_diff_loss(transformer_predictions, transformer_targets)

            else:
                raise ValueError("diff_mode is None")

        else:
            T = self.hparams.diffusion_hyperparams["T"]
            Gauss = std_normal((self.hparams.batch_size, mix.shape[1], 1), self.hparams.num_spks)
            input_targets = Gauss
            for t in range(T - 1, -1, -1):
                diffusion_steps = (t * torch.ones((self.hparams.batch_size, 1, 1)).type(torch.long))
                est_source = self.model(mix, diffusion_steps.view(self.hparams.batch_size), input_targets)
                # Gauss Noise
                input_targets = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, est_source, z=Gauss)

        return est_source, targets

    def reverb(self, predictions):
        reb = predictions
        for i in range(self.hparams.batch_size):
            reb[i, :, :] = reb[i, :, self.perms[i]]
        return reb

    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        loss, perms= self.hparams.loss(targets, predictions)
        self.perms = perms
        return loss

    def compute_diff_loss(self, predictions, targets):
        return self.hparams.diff_loss(targets, self.reverb(predictions))

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.auto_mix_prec:
            with autocast():
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, noise
                )
                loss = self.compute_objectives(predictions, targets)
                loss += self.diff_loss

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise
            )
            loss = self.compute_objectives(predictions, targets)
            loss += self.diff_loss

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)
            loss = loss.mean()


        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = (sisnr - sisnr_baseline).mean()
                    sisnr = sisnr.mean()

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )