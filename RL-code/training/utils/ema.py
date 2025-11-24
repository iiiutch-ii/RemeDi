import torch 
import torch.nn as nn
from copy import deepcopy

import torch_utils.distributed as dist



class ModelEmaV2(nn.Module):
    """
    code from timm
    """
    def __init__(self, model, decay=0.9999, device=None, dtype=torch.float32):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval().requires_grad_(False).to(device, dtype=dtype)
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device == 'cpu':
            self.off_loading = True
        else:
            self.off_loading = False
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        # if self.off_loading:
        #     self.model.to(device=)
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




class EDMEma(nn.Module):
    
    def __init__(self, model, batch_size, ema_halflife_kimg, ema_rampup_ratio: float = .05, device=None):
        super(EDMEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model) # .state_dict()
        self.module.eval().requires_grad_(False)
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        self.batch_size = batch_size * dist.get_world_size() 
        self.ema_halflife_nimg = ema_halflife_kimg * 1000
        self.register_buffer("cur_nimg", torch.tensor(0.))
        self.ema_rampup_ratio = ema_rampup_ratio
    
    @torch.no_grad()
    def update(self, model):
        self.cur_nimg += self.batch_size
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(self.ema_halflife_nimg, self.cur_nimg.item() * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (self.batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.module.parameters(), model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))


class NOEma(nn.Module):


    def __init__(self, model):
        super(ModelEmaV2, self).__init__()

    def update(self, model):
        pass



# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) /
                        (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))
                

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']