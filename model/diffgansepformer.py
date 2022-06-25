import csv
import logging
import torch.nn as nn
import os
import sys
import torch
import time
import torch.nn.functional as F
import tempfile
import pathlib
import torchaudio
import numpy as np
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from tqdm import tqdm
from torch.nn import SyncBatchNorm
from torch.cuda.amp import autocast
from types import SimpleNamespace
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main
from speechbrain.core import Stage
from torch.utils.data import DataLoader
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.tools import transformer_source, std_normal, calc_diffusion_step_embedding
from model.JCUD import JCUDiscriminator

# Logger info
logger = logging.getLogger(__name__)
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 7


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


def DiffGanNet(hparams):

    hparams["fc_t1"] = nn.Linear(
        hparams["step_embed"]["diffusion_step_embed_dim_in"],
        hparams["step_embed"]["diffusion_step_embed_dim_mid"]
    )
    hparams['generator_modules']['fc_t1'] = hparams["fc_t1"]
    hparams["fc_t2"] = nn.Linear(
        hparams["step_embed"]["diffusion_step_embed_dim_mid"],
        hparams["step_embed"]["diffusion_step_embed_dim_out"]
    )
    hparams['generator_modules']['fc_t2'] = hparams["fc_t2"]
    hparams["fc_t"] = nn.Linear(
        hparams["step_embed"]["diffusion_step_embed_dim_out"],
        hparams["Encoder"].conv1d.out_channels * hparams["num_spks"]
    )
    hparams['generator_modules']['fc_t'] = hparams["fc_t"]
    hparams["Encoder_mix"] = Encoder_mix(
        in_channels = hparams["Encoder"].conv1d.in_channels,
        out_channels = hparams["Encoder"].conv1d.out_channels,
        kernel_size = hparams["Encoder"].conv1d.kernel_size[0],
        stride = hparams["Encoder"].conv1d.stride[0],
        padding = hparams["Encoder"].conv1d.padding[0]
    )
    hparams['generator_modules']['encoder_mix'] = hparams["Encoder_mix"]
    hparams["discriminator_model"] = JCUDiscriminator(hparams)
    hparams['discriminator_modules'] = {'discriminator': hparams["discriminator_model"]}

    return hparams


# Define training procedure
class DiffGanSepformer(sb.Brain):
    def __init__(  # noqa: C901
        self,
        gen_modules=None,
        disc_modules=None,
        gen_opt_class=None,
        disc_opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        self.gen_opt_class = gen_opt_class
        self.disc_opt_class = disc_opt_class
        self.checkpointer = checkpointer

        # Arguments passed via the run opts dictionary
        run_opt_defaults = {
            "debug": False,
            "debug_batches": 2,
            "debug_epochs": 2,
            "device": "cpu",
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "find_unused_parameters": False,
            "jit_module_keys": None,
            "auto_mix_prec": False,
            "max_grad_norm": 5.0,
            "nonfinite_patience": 3,
            "noprogressbar": False,
            "ckpt_interval_minutes": 0,
        }

        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: "
                        + arg
                        + " arg overridden by command line input to: "
                        + str(run_opts[arg])
                    )
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: " + arg + " arg from hparam file is used"
                    )
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            logger.warn(
                "Detected Python "
                + str(sys.version_info.major)
                + "."
                + str(sys.version_info.minor)
                + ". We suggest using SpeechBrain with Python >="
                + str(PYTHON_VERSION_MAJOR)
                + "."
                + str(PYTHON_VERSION_MINOR)
            )

        if self.data_parallel_backend and self.distributed_launch:
            sys.exit(
                "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True"
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.lunch [args]\n"
                "experiment.py hyperparams.yaml --distributed_launch=True "
                "--distributed_backend=nccl"
            )

        # Switch to the right context
        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))

        # Put modules on the right device, accessible with dot notation
        self.gen_modules = torch.nn.ModuleDict(gen_modules).to(self.device)
        self.disc_modules = torch.nn.ModuleDict(disc_modules).to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # Checkpointer should point at a temporary directory in debug mode
        if (
            self.debug
            and self.checkpointer is not None
            and hasattr(self.checkpointer, "checkpoints_dir")
        ):
            tempdir = tempfile.TemporaryDirectory()
            logger.info(
                "Since debug mode is active, switching checkpointer "
                f"output to temporary directory: {tempdir.name}"
            )
            self.checkpointer.checkpoints_dir = pathlib.Path(tempdir.name)

            # Keep reference to tempdir as long as checkpointer exists
            self.checkpointer.tempdir = tempdir

        # Sampler should be handled by `make_dataloader`
        # or if you provide a DataLoader directly, you can set
        # this.train_sampler = your_sampler
        # to have your_sampler.set_epoch() called on each epoch.
        self.train_sampler = None

        # Automatic mixed precision init
        if self.auto_mix_prec:
            self.scaler = torch.cuda.amp.GradScaler()

        # List parameter count for the user
        total_params = sum(
            p.numel() for p in self.gen_modules.parameters() if p.requires_grad
        )
        total_params += sum(
            p.numel() for p in self.disc_modules.parameters() if p.requires_grad
        )
        if total_params > 0:
            clsname = self.__class__.__name__
            fmt_num = sb.utils.logger.format_order_of_magnitude(total_params)
            logger.info(f"{fmt_num} trainable parameters in {clsname}")

        if self.distributed_launch:
            self.rank = int(os.environ["RANK"])
            if not torch.distributed.is_initialized():
                if self.rank > 0:
                    sys.exit(
                        " ================ WARNING ==============="
                        "Please add sb.ddp_init_group() into your exp.py"
                        "To use DDP backend, start your script with:\n\t"
                        "python -m torch.distributed.launch [args]\n\t"
                        "experiment.py hyperparams.yaml "
                        "--distributed_launch=True --distributed_backend=nccl"
                    )
                else:
                    logger.warn(
                        "To use DDP, please add "
                        "sb.utils.distributed.ddp_init_group() into your exp.py"
                    )
                    logger.info(
                        "Only the main process is alive, "
                        "all other subprocess were killed."
                    )

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)


    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).

        The default implementation of this method depends on an optimizer
        class being passed at initialization that takes only a list
        of parameters (e.g., a lambda or a partial function definition).
        This creates a single optimizer that optimizes all trainable params.

        Override this class if there are multiple optimizers.
        """
        if (self.gen_opt_class is not None) and (self.disc_opt_class is not None):
            self.gen_optimizer = self.gen_opt_class(self.gen_modules.parameters())
            self.disc_optimizer = self.gen_opt_class(self.disc_modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("gen_optimizer", self.gen_optimizer)
                self.checkpointer.add_recoverable("disc_optimizer", self.disc_optimizer)

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.gen_modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(
                        module,
                        device_ids=[self.device],
                        find_unused_parameters=self.find_unused_parameters,
                    )
                    self.gen_modules[name] = module
        else:
            # data_parallel_backend
            for name, module in self.gen_modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = DP(module)
                    self.gen_modules[name] = module

        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.disc_modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(
                        module,
                        device_ids=[self.device],
                        find_unused_parameters=self.find_unused_parameters,
                    )
                    self.disc_modules[name] = module
        else:
            # data_parallel_backend
            for name, module in self.disc_modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = DP(module)
                    self.disc_modules[name] = module

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        disc_avg_train_loss = 0.0
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.gen_modules.train()
            self.disc_modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss[0], self.avg_train_loss
                    )
                    disc_avg_train_loss = self.update_average(
                        loss[1], disc_avg_train_loss
                    )
                    t.set_postfix(train_gen_loss=self.avg_train_loss, train_dsic_loss=disc_avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, (self.avg_train_loss, disc_avg_train_loss), epoch)
            self.avg_train_loss = 0.0
            self.disc_avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.gen_modules.eval()
                self.disc_modules.eval()
                gen_avg_valid_loss = 0.0
                disc_avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        gen_avg_valid_loss = self.update_average(
                            loss[0], gen_avg_valid_loss
                        )
                        disc_avg_valid_loss = self.update_average(
                            loss[0], disc_avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[Stage.VALID, (gen_avg_valid_loss, disc_avg_valid_loss), epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break

    def generator_model(self, diffusion_steps, mix, input_targets):
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps,
            self.hparams.step_embed["diffusion_step_embed_dim_in"]
        )
        diffusion_step_embed = swish(self.hparams.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.hparams.fc_t2(diffusion_step_embed))
        part_t = self.hparams.fc_t(diffusion_step_embed)
        part_t = part_t.view(
            [self.hparams.num_spks, self.hparams.batch_size, self.hparams.Encoder.conv1d.out_channels, 1])
        input_targets = self.hparams.Encoder_mix(input_targets)
        input_targets = input_targets + part_t

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks) #2 1 256 2499
        sep_h = mix_w * est_mask

        # Defusing
        sep_h = sep_h + input_targets

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
        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            diffusion_steps = torch.randint(self.hparams.diffusion_hyperparams["T"], size=(self.hparams.batch_size, 1, 1))
            Gauss = std_normal((self.hparams.batch_size, mix.shape[1], self.hparams.num_spks))
            input_targets = transformer_source(diffusion_steps, self.hparams.diffusion_hyperparams, targets, z=Gauss)
            self.input_taregets = input_targets
            self.diffusion_steps = diffusion_steps.view(self.hparams.batch_size, 1)

            est_source = self.generator_model(diffusion_steps.view(self.hparams.batch_size, 1), mix, input_targets)
            # Gauss Noise
            est_source = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, est_source, z=Gauss)
            targets = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, targets, z=Gauss)

        else:
            T = self.hparams.diffusion_hyperparams["T"]
            Gauss = std_normal((self.hparams.batch_size, mix.shape[1], self.hparams.num_spks))
            input_targets = Gauss
            for t in range(T - 1, -1, -1):
                diffusion_steps = (t * torch.ones((self.hparams.batch_size, 1, 1)).type(torch.long))
                est_source = self.generator_model(diffusion_steps.view(self.hparams.batch_size, 1), mix, input_targets)
                # Gauss Noise
                input_targets = transformer_source(diffusion_steps - 1, self.hparams.diffusion_hyperparams, est_source, z=Gauss)

        return est_source, targets

    def generator_compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        loss, perms = self.hparams.generator_loss(targets, predictions)
        self.perms = perms
        return loss

    def jcu_loss_fn(self, logit_cond, logit_uncond, label_fn, mask=None):
        cond_loss = F.mse_loss(logit_cond, label_fn(logit_cond), reduction="none" if mask is not None else "mean")
        cond_loss = (cond_loss * mask).sum() / mask.sum() if mask is not None else cond_loss
        uncond_loss = F.mse_loss(logit_uncond, label_fn(logit_uncond), reduction="none" if mask is not None else "mean")
        uncond_loss = (uncond_loss * mask).sum() / mask.sum() if mask is not None else uncond_loss
        return 0.5 * (cond_loss + uncond_loss)

    def d_loss_fn(self, r_logit_cond, r_logit_uncond, f_logit_cond, f_logit_uncond, mask=None):
        r_loss = self.jcu_loss_fn(r_logit_cond, r_logit_uncond, torch.ones_like, mask)
        f_loss = self.jcu_loss_fn(f_logit_cond, f_logit_uncond, torch.zeros_like, mask)
        return r_loss, f_loss

    def g_loss_fn(self, f_logit_cond, f_logit_uncond, mask=None):
        f_loss = self.jcu_loss_fn(f_logit_cond, f_logit_uncond, torch.ones_like, mask)
        return f_loss

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

        # generator
        if self.auto_mix_prec:
            with autocast():
                predictions, targets_out = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, noise
                )
                gen_loss = self.generator_compute_objectives(predictions, targets_out)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = gen_loss[gen_loss > th]
                    if loss_to_keep.nelement() > 0:
                        gen_loss = loss_to_keep.mean()
                else:
                    gen_loss = gen_loss.mean()

                if self.hparams.epoch_counter.current > self.hparams.discriminator_train_start_epoch:
                    gen_loss *= self.hparams.lambda_aux_after_introduce_adv_loss
                    for i in range(self.hparams.batch_size):
                        predictions[i, :, :] = predictions[i, :, self.perms[i]]
                    x_t_pre = torch.cat(predictions.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
                    x_t = torch.cat(self.input_taregets.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)

                    D_fake_cond, D_fake_uncond = self.hparams.discriminator_model(x_t_pre, x_t, self.diffusion_steps)
                    adv_loss = self.g_loss_fn(D_fake_cond[-1], D_fake_uncond[-1])
                    gen_loss -= self.hparams.lambda_adv * adv_loss

            if (
                gen_loss < self.hparams.loss_upper_lim and gen_loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(gen_loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.gen_modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.gen_optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                gen_loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets_out = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise
            )
            gen_loss = self.generator_compute_objectives(predictions, targets_out)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = gen_loss[gen_loss > th]
                if loss_to_keep.nelement() > 0:
                    gen_loss = loss_to_keep.mean()
            else:
                gen_loss = gen_loss.mean()

            if self.hparams.epoch_counter.current > self.hparams.discriminator_train_start_epoch:
                gen_loss *= self.hparams.lambda_aux_after_introduce_adv_loss
                for i in range(self.hparams.batch_size):
                    predictions[i, :, :] = predictions[i, :, self.perms[i]]
                x_t_pre = torch.cat(predictions.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
                x_t = torch.cat(self.input_taregets.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)

                D_fake_cond, D_fake_uncond = self.hparams.discriminator_model(x_t_pre, x_t, self.diffusion_steps)
                adv_loss = self.g_loss_fn(D_fake_cond[-1], D_fake_uncond[-1])
                gen_loss -= self.hparams.lambda_adv * adv_loss

            if (
                gen_loss < self.hparams.loss_upper_lim and gen_loss.nelement() > 0
            ):  # the fix for computational problems
                gen_loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.gen_modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.gen_optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                gen_loss.data = torch.tensor(0).to(self.device)
        self.gen_optimizer.zero_grad()

        disc_loss = torch.tensor(0).to(self.device)
        if self.hparams.epoch_counter.current > self.hparams.discriminator_train_start_epoch:
            if self.auto_mix_prec:
                with autocast():
                    with torch.no_grad():
                        predictions, targets_out = self.compute_forward(
                            mixture, targets, sb.Stage.TRAIN, noise
                        )
                    for i in range(self.hparams.batch_size):
                        predictions[i, :, :] = predictions[i, :, self.perms[i]]
                    x_t_pre = torch.cat(predictions.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
                    x_t_pre_ = torch.cat(targets_out.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
                    x_t = torch.cat(self.input_taregets.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)

                    D_fake_cond, D_fake_uncond = self.hparams.discriminator_model(x_t_pre, x_t, self.diffusion_steps)
                    D_real_cond, D_real_uncond = self.hparams.discriminator_model(x_t_pre_, x_t, self.diffusion_steps)

                    D_loss_real, D_loss_fake = self.d_loss_fn(D_real_cond[-1], D_real_uncond[-1], D_fake_cond[-1], D_fake_uncond[-1])

                    disc_loss = D_loss_real + D_loss_fake
                    if (
                            disc_loss < self.hparams.loss_upper_lim and disc_loss.nelement() > 0
                    ):  # the fix for computational problems
                        self.scaler.scale(disc_loss).backward()
                        if self.hparams.clip_grad_norm >= 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.disc_modules.parameters(), self.hparams.clip_grad_norm
                            )
                        self.disc_optimizer.step()
                    else:
                        self.nonfinite_count += 1
                        logger.info(
                            "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                                self.nonfinite_count
                            )
                        )
                        disc_loss.data = torch.tensor(0).to(self.device)
            else:
                with torch.no_grad():
                    predictions, targets_out = self.compute_forward(
                        mixture, targets, sb.Stage.TRAIN, noise
                    )
                for i in range(self.hparams.batch_size):
                    predictions[i, :, :] = predictions[i, :, self.perms[i]]
                x_t_pre = torch.cat(predictions.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
                x_t_pre_ = torch.cat(targets_out.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
                x_t = torch.cat(self.input_taregets.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)

                D_fake_cond, D_fake_uncond = self.hparams.discriminator_model(x_t_pre, x_t, self.diffusion_steps)
                D_real_cond, D_real_uncond = self.hparams.discriminator_model(x_t_pre_, x_t, self.diffusion_steps)

                D_loss_real, D_loss_fake = self.d_loss_fn(D_real_cond[-1], D_real_uncond[-1], D_fake_cond[-1],
                                                          D_fake_uncond[-1])
                disc_loss = D_loss_real + D_loss_fake

                if (
                        disc_loss < self.hparams.loss_upper_lim and disc_loss.nelement() > 0
                ):  # the fix for computational problems
                    disc_loss.backward()
                    if self.hparams.clip_grad_norm >= 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.disc_modules.parameters(), self.hparams.clip_grad_norm
                        )
                    self.disc_optimizer.step()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    disc_loss.data = torch.tensor(0).to(self.device)
        self.disc_optimizer.zero_grad()

        return (gen_loss.detach().cpu(), disc_loss.detach().cpu())

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets_out = self.compute_forward(mixture, targets, stage)
            gen_loss = self.generator_compute_objectives(predictions, targets_out)
            gen_loss = gen_loss.mean()


        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets_out, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets_out, predictions)

        disc_loss=torch.tensor(0).to(self.device)
        if stage == sb.Stage.VALID:
            for i in range(self.hparams.batch_size):
                predictions[i, :, :] = predictions[i, :, self.perms[i]]
            x_t_pre = torch.cat(predictions.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
            x_t_pre_ = torch.cat(targets_out.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
            x_t = torch.cat(self.input_taregets.chunk(self.hparams.num_spks, dim=2), dim=1).squeeze(dim=2)
            with torch.no_grad():
                D_fake_cond, D_fake_uncond = self.hparams.discriminator_model(x_t_pre, x_t, self.diffusion_steps)
                D_real_cond, D_real_uncond = self.hparams.discriminator_model(x_t_pre_, x_t, self.diffusion_steps)

            D_loss_real, D_loss_fake = self.d_loss_fn(D_real_cond[-1], D_real_uncond[-1], D_fake_cond[-1], D_fake_uncond[-1])
            disc_loss = D_loss_real + D_loss_fake

        return (gen_loss.detach(), disc_loss.detach())

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.gen_modules.eval()
        self.disc_modules.eval()
        gen_avg_test_loss = 0.0
        disc_avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                gen_avg_test_loss = self.update_average(loss[0], gen_avg_test_loss)
                disc_avg_test_loss = self.update_average(loss[1], disc_avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, (gen_avg_test_loss, disc_avg_test_loss), None]
            )
        self.step = 0

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        gen_stage_stats = {"si-snr": stage_loss[0]}
        disc_stage_stats = {"loss": stage_loss[1]}
        if stage == sb.Stage.TRAIN:
            self.train_gen_stats = gen_stage_stats
            self.train_disc_stats = disc_stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.generator_lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                gen_current_lr, gen_next_lr = self.hparams.generator_lr_scheduler(
                    [self.gen_optimizer], epoch, stage_loss[0]
                )
                schedulers.update_learning_rate(self.gen_optimizer, gen_next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                gen_current_lr = self.hparams.generator_optimizer.optim.param_groups[0]["lr"]


            # Learning rate annealing
            if isinstance(
                self.hparams.discriminator_lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                disc_current_lr, disc_next_lr = self.hparams.discriminator_lr_scheduler(
                    [self.disc_optimizer], epoch, stage_loss[1]
                )
                schedulers.update_learning_rate(self.disc_optimizer, disc_next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                disc_current_lr = self.hparams.discriminator_optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "gen_lr": gen_current_lr, "disc_lr": disc_current_lr},
                train_gen_stats=self.train_gen_stats,
                train_disc_stats=self.train_disc_stats,
                valid_gen_stats=gen_stage_stats,
                valid_disc_stats=disc_stage_stats
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": gen_stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_gen_stats=gen_stage_stats,
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
                    sisnr = self.generator_compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.generator_compute_objectives(
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