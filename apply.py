import torch

from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from .curves import latent_time_weights, resample


def blend_conditioning(positive, action_positive, latent_shapes, curve, strength):
    """Two masked positives whose per-frame weights sum to 1, so the sampler averages the
    base and action predictions frame by frame. Action entries inherit the base entry's
    extra conds (guides, references) and override only what the action encode set."""
    # mask layout is the latent minus its channel dim; multi-stream latents keep channels inside the packed axis
    action_mask = latent_time_weights(latent_shapes, curve, "cpu")[:, 0] * strength
    base_mask = 1.0 - action_mask
    base_extras = positive[0][1]

    out = []
    for cond, extras, mask in [(c, d, base_mask) for c, d in positive] + [(c, {**base_extras, **d}, action_mask) for c, d in action_positive]:
        extras = extras.copy()
        extras["mask"] = mask
        extras["mask_strength"] = 1.0
        extras["set_area_to_bounds"] = False
        out.append([cond, extras])
    return out


class CurveCFG:
    """sampler_cfg_function: CFG scale follows the curve on each latent frame, 1.0 at curve 0 with full strength."""

    def __init__(self, curve, strength):
        self.curve = curve
        self.strength = strength

    def __call__(self, args):
        cond, uncond = args["cond"], args["uncond"]
        w = latent_time_weights(args["model"].latent_shapes, self.curve, cond.device).to(cond.dtype)
        scale = 1.0 + (args["cond_scale"] - 1.0) * (1.0 - self.strength + self.strength * w)
        return uncond + (cond - uncond) * scale


class CurveFunControlPatch(MiniMaxH3FunControlPatch):
    """MiniMax H3 Fun ControlNet whose injection strength follows the curve on each target video frame."""

    def __init__(self, model_patch, vae, control_video, strength, sigma_start, sigma_end, curve):
        super().__init__(model_patch, vae, control_video, None, None, strength, sigma_start, sigma_end)
        self.curve = curve
        self.row_weights = None

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        layout = args["layout"]
        control_index = self.model_patch.model.injection_layers.index(block_index)
        if control_index == 0:
            self.control_latent = self.control_latent.to(out["img"].device)
            self.control_stream = self.model_patch.model.init_stream(
                self.pristine_stream, self.control_latent, layout, args["t_emb"])
            self.pristine_stream = None
            # target video is the last packed segment, frame-major rows
            va, vb, _ = next(s for s in layout.segments if s[2] == "video")
            latent_t = layout.signature[1]
            w = torch.tensor(resample(self.curve, latent_t), device=out["img"].device, dtype=out["img"].dtype)
            self.row_weights = (va, vb, w.repeat_interleave((vb - va) // latent_t).unsqueeze(1))
        self.control_stream, skip = self.model_patch.model.step(
            control_index, self.control_stream, args["t_emb"], args["mod_segments"], args["rope_freqs"],
            transformer_options=args["transformer_options"])
        skip[layout.audio_pos.to(skip.device)] = 0
        va, vb, w = self.row_weights
        out["img"][:va].add_(skip[:va], alpha=self.strength)
        out["img"][va:vb].addcmul_(skip[va:vb], w, value=self.strength)
        return out

    def cleanup(self):
        super().cleanup()
        self.row_weights = None
