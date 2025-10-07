from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple

class TargetType(str, Enum):
    PC1 = "pc1"
    PC2 = "pc2"
    CUSTOM_PC = "custom_pc"

class AxisOfInterest(str, Enum):
    COLOR = "color"
    SHAPE = "shape"
    CUSTOM_LINE = "custom_line"  # defined by two stimulus points

class ConstraintType(str, Enum):
    BALL = "ball"   # L2 ball around 1
    BOX = "box"     # box bounds around 1

class WRNormalization(str, Enum):
    NONE = "none"            # dominant PC1 regime
    ROW = "row"              # small PC1 regime (row-sum zero)
    ROW_AND_COL = "row_and_col"

@dataclass
class NetworkConfig:
    N: int = 120
    K: int = 2  # features (shape,color)
    seed: int = 21
    desired_radius: float = 0.9
    p_high: float = 0.2    # within-block connection prob
    p_low: float = 0.2     # cross-block connection prob
    zero_sum: WRNormalization = WRNormalization.ROW
    wr_tuned: bool = False
    weight_scale: float = 0.1
    baseline_equalize: bool = False

@dataclass
class ObjectiveConfig:
    target_type: TargetType = TargetType.CUSTOM_PC
    theta_deg: float = 30.0   # azimuth (deg) for CUSTOM_PC in PC space
    phi_deg: float = 25.0     # elevation from XY plane (deg) for CUSTOM_PC
    axis_of_interest: AxisOfInterest = AxisOfInterest.COLOR
    shape_for_color_line: float = 0.3    # for color-axis
    color_for_shape_line: float = 0.3    # for shape-axis
    custom_stim_line_start: Optional[Tuple[float, float]] = None  # when CUSTOM_LINE
    custom_stim_line_end: Optional[Tuple[float, float]] = None

@dataclass
class ConstraintConfig:
    type: ConstraintType = ConstraintType.BALL
    radius: float = 0.12        # L2 radius as fraction of sqrt(N); actual R = radius*sqrt(N)
    box_half_width: float = 0.2 # g in [1-w, 1+w] if BOX
    hard_norm: bool = True      # enforce ||axis(g)|| == ||axis(1)||
    positive_gains: bool = False

@dataclass
class GridConfig:
    shape_vals: Sequence[float] = (0.0, 0.33, 0.66, 1.0)
    color_vals: Sequence[float] = (0.0, 0.33, 0.66, 1.0)

@dataclass
class ShuffleConfig:
    enabled: bool = True
    num_bins: int = 10
    binning: str = "quantile"      # "quantile" or "equal"
    mode: str = "independent"      # "independent" or "paired"
    seed: Optional[int] = None     # if None, will use network.seed + 101
    repeats: int = 1               # (N repeats for panel_d)

@dataclass
class ExperimentConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    save_dir: str = "outputs"
    tag: str = "default_smallpc1_custom"
    shuffle: ShuffleConfig = field(default_factory=ShuffleConfig)  

@dataclass
class SweepConfig:
    ranges_ball: Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22,0.26,0.30)
    ranges_box:  Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)

@dataclass
class ExperimentConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    save_dir: str = "outputs"
    tag: str = "default_smallpc1_custom"
    shuffle: ShuffleConfig = field(default_factory=ShuffleConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)   # <-- NEW
