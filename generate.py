import hydra
from omegaconf import DictConfig, OmegaConf
import os

from string_trans.generator import PBEGenerator, EditDistanceGenerator, HardProgramWithReleventIOGenerator

@hydra.main(version_base=None, config_path="conf", config_name="generate")
def app(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg_file = os.path.join(cfg.output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    if cfg.exec == "PBE":
        generator = PBEGenerator(cfg)
        generator.generate()
    elif cfg.exec == "EditDistance":
        generator = EditDistanceGenerator(cfg)
        generator.generate()
    elif cfg.exec == "HardProgram":
        generator = HardProgramWithReleventIOGenerator(cfg)
        generator.generate()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    app()
