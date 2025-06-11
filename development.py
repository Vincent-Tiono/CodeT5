import hydra
from omegaconf import DictConfig, OmegaConf
import os

from src.constants import DevelopmentScripts

@hydra.main(version_base=None, config_path="conf", config_name="train")
def app(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg_file = os.path.join(cfg.output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    if cfg.exec == DevelopmentScripts.TOP_k:
        from src.development.clip_top_k import clip_top_k
        clip_top_k(cfg)
    elif cfg.exec == DevelopmentScripts.DIFFICULTY:
        from src.development.clip_difficulty import clip_difficulty
        clip_difficulty(cfg)
    elif cfg.exec == DevelopmentScripts.SIMILAR_ACCURACY:
        from src.development.similar_accuracy import similar_accuracy
        similar_accuracy(cfg)
    elif cfg.exec == DevelopmentScripts.SO_FUN:
        from src.development.so_fun import so_fun
        so_fun(cfg)
    elif cfg.exec == DevelopmentScripts.SEQ2SEQ:
        from src.development.seq2seq_ap_nps import seq2seq_nps
        seq2seq_nps(cfg)


if __name__ == "__main__":
    app()
