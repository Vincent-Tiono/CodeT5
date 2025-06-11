import hydra
from omegaconf import DictConfig, OmegaConf
import os

from src.constants import TrainScript
from src.autotuner import AutoTuner

@hydra.main(version_base=None, config_path="conf", config_name="train")
def app(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.eval_only:
        cfg_file = os.path.join(cfg.output_dir, "eval.yaml")
    else:
        cfg_file = os.path.join(cfg.output_dir, "config.yaml")
    OmegaConf.save(cfg, cfg_file)
    if cfg.exec == TrainScript.SEQ2SEQ_NPS:
        from src.seq2seq_nps import seq2seq_nps
        if cfg.autotune:
            autotuner = AutoTuner(seq2seq_nps, cfg)
            autotuner.tune()
        else:
            seq2seq_nps(cfg)
    elif cfg.exec == TrainScript.SEQ2SEQ_NPSE:
        from src.seq2seq_npse import seq2seq_npse
        seq2seq_npse(cfg)
    elif cfg.exec == TrainScript.SEQ2SEQ_NPE:
        from src.seq2seq_npe import seq2seq_npe
        seq2seq_npe(cfg)



if __name__ == "__main__":
    app()
