import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

params = AttrDict(
        # Training params
        batch_size=32,
        learning_rate=2e-4,
        max_grad_norm=None,

        # Data params
        sample_rate=16000,
      
        # Model params
        residual_layers=36,
        residual_channels=256,
        dilation_cycle_length=11,
        noise_schedule=np.linspace(1e-4, 0.02, 200).tolist(),

        # text params
        total_phonemes = 73,
        max_duration_phoneme = 101, # the meaning is the whole frame is consist of single phoneme.

        # Autoregressive behavior:
        window_length = 8000, #500 ms
        frame_length = 4000, #250 ms
)

