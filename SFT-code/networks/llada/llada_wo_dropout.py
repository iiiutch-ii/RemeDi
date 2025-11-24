import torch
import torch.nn as nn
from networks.llada.modeling_llada import LLaDAModelLM

class LLaDAWODropout(LLaDAModelLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset_dropout()

    def reset_dropout(self):
        for m in self.modules():
            # Only override for layers where behavior changes between train/eval
            if isinstance(m, (
                nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                nn.AlphaDropout,
            )):
                m.p = 0  # Force eval behavior
