import hydra
from omegaconf import DictConfig, OmegaConf
import os

from src.constants import PretrainScript

@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def app(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.eval_only:
        cfg_file = os.path.join(cfg.output_dir, "eval.yaml")
    else:
        cfg_file = os.path.join(cfg.output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    if cfg.exec == PretrainScript.PROGRAM_IO:
        from src.pretrain_program_io import pretrain_program_io
        pretrain_program_io(cfg)

if __name__ == "__main__":
    app()
