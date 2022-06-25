from model.sepformer import Sepformer
from model.diffsepformer import DiffSepformer, DiffNet
from model.diffsepformer_1 import DiffSepformer_1, DiffNet_1
from model.auxsepformer import AuxSepformer, AuxNet
from model.diffgansepformer import DiffGanSepformer, DiffGanNet
from utils.tools import calc_diffusion_hyperparams

def get_model(hparams, run_opts):
    if hparams["mode"] == "sepformer":
        separator = Sepformer(
            modules=hparams["modules"],
            opt_class=hparams["optimizer"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
    elif hparams["mode"] == "diffsepformer":
        hparams["diffusion_hyperparams"] = calc_diffusion_hyperparams(
            hparams["diffusion_config"]["noise_schedule_naive"],
            hparams["diffusion_config"]["T"],
            hparams["diffusion_config"]["beta_0"],
            hparams["diffusion_config"]["beta_T"],
            hparams["diffusion_config"]["s"],
        )
        hparams = DiffNet(hparams)
        separator = DiffSepformer(
            modules=hparams["modules"],
            opt_class=hparams["optimizer"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
    elif hparams["mode"] == "diffsepformer_1":
        hparams["diffusion_hyperparams"] = calc_diffusion_hyperparams(
            hparams["diffusion_config"]["noise_schedule_naive"],
            hparams["diffusion_config"]["T"],
            hparams["diffusion_config"]["beta_0"],
            hparams["diffusion_config"]["beta_T"],
            hparams["diffusion_config"]["s"],
        )
        hparams = DiffNet_1(hparams)
        separator = DiffSepformer_1(
            modules=hparams["modules"],
            opt_class=hparams["optimizer"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
    elif hparams["mode"] == "auxsepformer":
        hparams["diffusion_hyperparams"] = calc_diffusion_hyperparams(
            hparams["diffusion_config"]["noise_schedule_naive"],
            hparams["diffusion_config"]["T"],
            hparams["diffusion_config"]["beta_0"],
            hparams["diffusion_config"]["beta_T"],
            hparams["diffusion_config"]["s"],
        )
        hparams = AuxNet(hparams)
        separator = AuxSepformer(
            modules=hparams["modules"],
            opt_class=hparams["optimizer"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
    elif hparams["mode"] == "diffgansepformer":
        hparams["diffusion_hyperparams"] = calc_diffusion_hyperparams(
            hparams["diffusion_config"]["noise_schedule_naive"],
            hparams["diffusion_config"]["T"],
            hparams["diffusion_config"]["beta_0"],
            hparams["diffusion_config"]["beta_T"],
            hparams["diffusion_config"]["s"],
        )
        hparams = DiffGanNet(hparams)
        separator = DiffGanSepformer(
            gen_modules=hparams["generator_modules"],
            disc_modules=hparams["discriminator_modules"],
            gen_opt_class=hparams["generator_optimizer"],
            disc_opt_class=hparams["discriminator_optimizer"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
    else:
        raise ValueError("mode is None")

    return separator
