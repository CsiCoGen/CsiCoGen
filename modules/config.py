import os
import re
from dataclasses import dataclass, field

import yaml


@dataclass
class CFG:
    TRAIN_FILE: str = ""
    TEST_FILE: str = ""
    GT_RAW_FILE: str = ""

    DATASET: str = "COST2100"
    SCENE: str = "indoor"
    SEED: int = 42

    MAT_SIZE: int = 32
    IN_CHANNELS: int = 2

    GEN_MODE: str = "ddpm"  # Gaussian diffusion
    MODEL_NAME: str = "csicogen"  # csicogen or csicogen_lite

    DIFF_EPOCHS: int = 3000
    DIFF_LR: float = 1e-3
    DIFF_BATCH: int = 4096
    DIFF_TRAIN_SAMPLES: int = 0
    DIFF_TRAIN_WORKERS: int = 8
    DIFF_TRAIN_PREFETCH: int = 4
    DIFF_TRAIN_DROP_LAST: bool = True
    DIFF_TIMESTEPS: int = 50
    DIFF_MAX_KEEP: int = 5

    DIFF_EVAL_EVERY: int = 50
    DIFF_EVAL_SAMPLES: int = 20000
    DIFF_EVAL_BATCH: int = 500
    DIFF_INFER_BATCH: int = 256
    DIFF_DENOISER_MICRO_BATCH: int = 0
    DIFF_MAX_SAMPLES: int = 0
    DIFF_INFER_SAMPLER: str = "ddpm"
    DIFF_INFER_STEPS: int = 0
    DIFF_INFER_TIMESTEP_SPACING: str = "uniform"
    DIFF_INFER_FEEDBACK_STEPS: int = 0
    DIFF_FEEDBACK_TIMESTEP_SPACING: str = ""
    DIFF_DDIM_ETA: float = 0.0
    DIFF_STEP_METRIC_TIMESTEPS: str = "none"
    DIFF_STEP_METRIC_RHO: bool = True
    DIFF_STEP_SAVE_DIR: str = ""
    DIFF_STEP_SAVE_TIMESTEPS: str = ""


    DIFF_BETA_START: float = 1e-4
    DIFF_BETA_END: float = 0.02
    DIFF_PRED_MODE: str = "h0"  # h0 or eps
    DIFF_NOISE_SCHED_TRAIN: str = "linear"  # linear or cosine
    DIFF_NOISE_SCHED_INFER: str = "linear"  # linear or cosine

    DIFF_USE_AMP: bool = True
    DIFF_INFER_USE_AMP: bool = True
    DIFF_GRAD_CLIP: float = 1.0
    DIFF_USE_EMA: bool = True
    DIFF_EMA_DECAY: float = 0.999
    DIFF_EVAL_USE_EMA: bool = True

    DIFF_SAVE_EVERY: int = 1
    DIFF_LR_SCHED: str = "multistep"  # multistep or cosine

    DIFF_CKPT_PATH: str = ""
    DIFF_NORM_STATS_PATH: str = ""
    DIFF_RESUME_PATH: str = ""

    DIFF_MODEL_ARCH: str = "resattn"  # Residual-attention denoiser
    DIFF_MODEL_DIM: int = 128
    DIFF_MODEL_BLOCKS: int = 8
    DIFF_MODEL_ATTN_EVERY: int = 2
    DIFF_MODEL_ATTN_HEADS: int = 4
    DIFF_MODEL_ATTN_DOWNSAMPLE: int = 2
    DIFF_MODEL_ATTN_MAX_HW: int = 8
    DIFF_MODEL_EXPANSION: int = 2  # Recorded in metadata only; does not change the network.


    DIFF_CHANNELS_LAST: bool = True
    DIFF_FAST_BENCHMARK: bool = True
    DIFF_ALLOW_TF32: bool = True
    DIFF_MATMUL_PRECISION: str = "high"

    DIFF_T_SAMPLER: str = "uniform"  # uniform or low_bias
    DIFF_T_BIAS_POWER: float = 2.0
    DIFF_LOSS_WEIGHT: str = "none"   # none or low_t
    DIFF_LOSS_T_POWER: float = 1.0

    CODEBOOK_SIZE: int = 256
    CODEBOOK_PATH: str = ""
    CODEBOOK_SHARED_ROOT: str = ""
    CODEBOOK_SEED: int = -1
    CODEBOOK_CLIP: bool = False
    CODEBOOK_CLIP_MIN: float = -1.0
    CODEBOOK_CLIP_MAX: float = 1.0

    DIFF_MACRO_STRIDE: int = 1
    DIFF_MACRO_TAIL_STEPS: int = 4
    DIFF_MACRO_ETA: float = 1.0
    DIFF_MACRO_MULTI_INDEX: bool = True
    DIFF_MACRO_REFRESH_MODE: str = "none"  # none or teacher
    DIFF_MACRO_REFRESH_RATIO: float = 0.5
    DIFF_MACRO_REFRESH_MIN_SPAN: int = 6
    DIFF_MACRO_REFRESH_COUNT: int = 1
    DIFF_MACRO_REFRESH_T_LIST: str = ""


    DIFF_EVAL_PRINT_STEPS: bool = True
    DIFF_PREFIX_LENGTHS: str = "all"  # all, none, full, or comma-separated prefix lengths
    SAMPLE_PRINT_EVERY_STEP: bool = True

    OUTPUT_ROOT: str = "runs"
    RUN_TAG: str = ""

    # Accepted for archived YAML compatibility; these fields have no runtime effect.
    ARTIFACT_ROOT: str = "checkpoints"
    DATA_ROOT: str = "data/COST2100"
    CUDA: str = "0"
    DIFF_INFER_USE_EMA: bool = True
    DIFF_WARMUP_EPOCHS: int = 0
    DIFF_MODEL_PRESET: str = ""

    RUN_DIR: str = field(init=False)
    DIFF_DIR: str = field(init=False)
    DIFF_CKPT_DIR: str = field(init=False)
    DIFF_RESULTS_DIR: str = field(init=False)
    DIFF_INFERENCE_DIR: str = field(init=False)
    GT_DATA_FILE: str = field(init=False)

    def __post_init__(self):
        self._recompute()

    def _recompute(self):
        exp_name = f"csi{self.MAT_SIZE}_cb{self.CODEBOOK_SIZE}_seed{self.SEED}"
        tag = str(self.RUN_TAG).strip()
        if tag:
            exp_name = f"{exp_name}_{tag}"
        self.RUN_DIR = os.path.join(self.OUTPUT_ROOT, self.DATASET, self.SCENE, exp_name)
        self.DIFF_DIR = os.path.join(self.RUN_DIR, "diffusion")
        self.DIFF_CKPT_DIR = os.path.join(self.DIFF_DIR, "ckpt")
        self.DIFF_RESULTS_DIR = os.path.join(self.DIFF_DIR, "training")
        self.DIFF_INFERENCE_DIR = os.path.join(self.DIFF_DIR, "inference")
        self.GT_DATA_FILE = self.TEST_FILE
        if not self.GT_RAW_FILE:
            self.GT_RAW_FILE = self._infer_raw_file_from_test(self.TEST_FILE)

    @staticmethod
    def _infer_raw_file_from_test(test_file: str) -> str:
        if not test_file:
            return ""
        base = os.path.basename(test_file)
        m = re.match(r"DATA_Htest(in|out)\.mat$", base)
        if not m:
            return ""
        scenario = m.group(1)
        return os.path.join(os.path.dirname(test_file), f"DATA_HtestF{scenario}_all.mat")

    def load_yaml(self, yaml_file: str, strict: bool = False):
        with open(yaml_file, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

        unknown = []
        for k, v in cfg_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                unknown.append(k)

        if unknown and strict:
            raise KeyError(f"Unknown keys in yaml: {unknown}")
        self._recompute()

    def ensure_dirs(self):
        for d in [self.DIFF_CKPT_DIR, self.DIFF_RESULTS_DIR, self.DIFF_INFERENCE_DIR]:
            os.makedirs(d, exist_ok=True)
