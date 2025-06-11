import hydra
from omegaconf import DictConfig, OmegaConf
import os

from src.constants import VisualizationScripts

@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def app(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg_file = os.path.join(cfg.output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    # import ipdb
    # ipdb.set_trace()
    if cfg.exec == VisualizationScripts.CLIP_EMBED:
        from src.visualization.clip_embedding import clip_embedding
        clip_embedding(cfg)

if __name__ == "__main__":
    app()
