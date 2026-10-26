"""mlx-vlm wrapper for Olala.

mlx-engine serves text models with continuous batching through mlx-vlm model
classes (BatchedVisionModelKit), so the model is also exposed as
``mlx_vlm.models.olala``: same modules and weights as the mlx-lm model under a
``language_model`` prefix, with mlx-vlm cache classes.
"""

from dataclasses import dataclass

from mlx import nn
from mlx_vlm.models import cache as vlm_cache
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput

from . import model as olala


@dataclass
class ModelConfig(olala.ModelArgs):
    pass


class KVCache(olala.InplaceAppendMixin, vlm_cache.KVCache):
    """mlx-vlm KVCache with in-place appends for the fused decode path."""


class CacheList(olala.AttnCacheMixin, vlm_cache.CacheList):
    pass


class LanguageModel(olala.Model):
    def __call__(
        self, inputs=None, cache=None, inputs_embeds=None, logits_to_keep=None, **kwargs
    ):
        from . import metal

        # continuous batching sees several batch sizes: compile their decode kernels in the
        # background from the first call on (mlx-engine probes the model at load time)
        metal.maybe_warmup(self.model, self.model._head_fn)
        x = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        if isinstance(x, tuple):
            from . import metal

            return LanguageModelOutput(logits=metal.head(self.lm_head, x[0]))
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            x = x[:, -logits_to_keep:]
        return LanguageModelOutput(
            logits=self.lm_head(x.astype(self.model.compute_dtype()))
        )

    def make_cache(self):
        return [
            vlm_cache.ArraysCache(1)
            if layer.layer_type == "M"
            else CacheList(KVCache(), vlm_cache.ArraysCache(2))
            for layer in self.model.layers
        ]


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self, input_ids=None, pixel_values=None, **kwargs
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embedding(input_ids)
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        if kwargs.get("inputs_embeds") is None:
            kwargs["inputs_embeds"] = self.get_input_embeddings(input_ids).inputs_embeds
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        if any(k.startswith("language_model.") for k in weights):
            inner = {
                k[len("language_model.") :]: v
                for k, v in weights.items()
                if k.startswith("language_model.")
            }
        else:
            inner = weights
        return {
            f"language_model.{k}": v
            for k, v in self.language_model.sanitize(inner).items()
        }

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
