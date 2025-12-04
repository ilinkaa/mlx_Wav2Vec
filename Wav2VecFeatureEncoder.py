import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import tree_flatten

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf



class Wav2Vec2GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id = 0):
        super().__init__()
        #Okk what is going on here, maybe try to get the OG parameters for the test run 
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )

        self.activation = nn.GELU()
        #Not sure if Dims is correct here 
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, dims=self.out_conv_dim, affine=True)
    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# Not sure about the class properties also
class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id = 0):
        super().__init__()
        #Okk what is going on here, maybe try to get the OG parameters for the test run 
        self.in_conv_dim = 512
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )

        self.activation = nn.GELU()
    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        #hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
  
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers
        #self.gradient_checkpointing = False
        #self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            mx.stop_gradient(param)
        #self._requires_grad = False

    def __call__(self, input_values):
        hidden_states = input_values[:, None, :]          # (batch, 1, seq_len)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1))
        #hidden_states = input_values[:, None]

    

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states
    