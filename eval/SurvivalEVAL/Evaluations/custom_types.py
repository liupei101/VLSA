from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]
