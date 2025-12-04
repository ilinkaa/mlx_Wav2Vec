#from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder as torch_feature_encoder
import soundfile as sf 
import json
import torch
import glob
from types import SimpleNamespace
from huggingface_hub import snapshot_download
from Wav2Vec2FeatureProjection import Wav2Vec2FeatureProjection as mlx_feature_proj
from transformers import AutoFeatureExtractor
from Wav2Vec2FeatureEncoder import Wav2Vec2FeatureEncoder as mlx_feature_encoder
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureProjection as torch_feature_projection
from Wav2Vec2Model import Wav2Vec2Model
import mlx.core as mx
from pathlib import Path
from Wav2Vec2ForCTC import Wav2Vec2ForCTC
from transformers import Wav2Vec2ForCTC as torch_model
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

torch.manual_seed(0)

mx.random.seed(0)

audio_path ="/Users/IlincaV/osati-asr/corpora/sample-corpus/recordings/wav/bicycle.wav"

audio, sr = sf.read(audio_path)
model = torch_model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
model_config = Wav2Vec2Config("facebook/wav2vec2-base-960h")

w = torch_feature_encoder(config=model_config)
#ft_projection = Wav2Vec2FeatureProjection(config = model_config)
w._requires_grad_ = False
w.training = False
ft_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")




#Wait what if i kept the torch feature extractor and then converted for pass to model ?

extracted = ft_extractor(audio, sampling_rate = sr, return_tensors = "pt")
extracted = extracted["input_values"]
# So pass to feature extractor works
torch_feature_enc = torch_feature_encoder(config= model_config)
res_torch = torch_feature_enc(extracted)
res_torch = res_torch.transpose(1, 2)

torch_feature_proj = torch_feature_projection(config = model_config)
res_torch = torch_feature_proj(res_torch)
print(res_torch[0].shape)


# But its beefing with the feature projection
#BECAUSE IT NEEDS THE TRANSPOSE DUH
#res = res.transpose(1, 2)
# Now has shape [batch, timeframes, features]
#print(res.shape)
#BASICALLY THE QUESTION is: is the output the same after applying the extractor / applying the 
# extractor within the model


mlx_model = Wav2Vec2ForCTC(config = model_config)

def from_pretrained(model : Wav2Vec2ForCTC,
    hf_id_or_path: str,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    cache_dir: str | Path | None = None,
) -> Wav2Vec2ForCTC:
    """Loads model from Hugging Face or local directory"""
    model_path = snapshot_download(hf_id_or_path)
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
        model_config = SimpleNamespace(**config)
    weight = [(k, v  ) 
                for wf in glob.glob(f"{model_path}/*.safetensors") 
                for k, v in mx.load(wf).items()]    
    
    
    weights = []
    for i in weight: 
        if "wav2vec2.encoder.pos_conv_embed.conv.bias."  in i[0]:
            pass
        else:
            weights.append(i)
    for j in weights:
        if j[0] =="wav2vec2.encoder.pos_conv_embed.conv.bias":
            print("here")
    model.load_weights(weights)
    # cast dtype
    #curr_weights = dict(tree_flatten(model.parameters()))
    #curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    #model.update(tree_unflatten(curr_weights))

    return model

mlx_model = from_pretrained(hf_id_or_path="facebook/wav2vec2-base-960h", model = mlx_model)
res = mx.array(extracted)
res_extr = mx.transpose(res[0,2,1])
