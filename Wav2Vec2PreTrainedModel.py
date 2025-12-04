import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import tree_flatten
from typing import Union, Optional
from Wav2Vec2PositionalConvEmbeddings import Wav2Vec2PositionalConvEmbedding
from Wav2Vec2FeatureProjection import Wav2Vec2FeatureProjection
class Wav2Vec2PreTrainedModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    
    def _init_weights(self, module):
        """Initialize the weights"""
        # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
      
        # gumbel softmax requires special init
        zero_init = nn.init.constant(0)
        mx.eval(module.parameters())
       
        if isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal(
                module.conv.weight,
                mean=0,
                std=2 * mx.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant(module.conv.bias, 0)
        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = mx.sqrt(1 / module.projection.in_features)
            nn.init.uniform(module.projection.weight, a=-k, b=k)
            nn.init.uniform(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            nn.init.normal(module.weight, mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                nn.init.constant(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            
            zero_init(module.bias)
            zero_init(module.weight)
        elif isinstance(module, nn.Conv1d):
            kaiming = nn.init.he_normal()
            kaiming(module.weight)

            if module.bias is not None:
                k = mx.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[mx.array, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return mx.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: mx.array, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        #output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = mx.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(mx.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
