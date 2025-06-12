from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from copy import deepcopy
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo.prompts
import ddpo.rewards
from ddpo.stat_tracking import PerPromptStatTracker
from ddpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
from tqdm import tqdm
import tempfile
from PIL import Image

class DDPOAgent:
    def __init__(self, pop_size, weights: np.ndarray,config, accelerator: Accelerator) -> None:
        # set policy id and weights of objectives
        self.pop_size = pop_size
        self.weights = weights
        assert weights.shape[0] == pop_size
        # load scheduler, tokenizer and models.
        self.config = config
        self.accelerator = accelerator
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            config.pretrained.model, revision=config.pretrained.revision
        )
        # freeze parameters of models to save more memory
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(not config.use_lora)
        # disable safety checker
        self.pipeline.safety_checker = None
        # make the progress bar nicer
        self.pipeline.set_progress_bar_config(
            position=1,
            disable=not accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        # switch to DDIM scheduler
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        inference_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to inference_dtype
        self.pipeline.vae.to(accelerator.device, dtype=inference_dtype)
        self.pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
        if config.use_lora:
            self.pipeline.unet.to(accelerator.device, dtype=inference_dtype)

        if config.use_lora:
            # Set correct lora layers
            lora_attn_procs = {}
            for name in self.pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.pipeline.unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.pipeline.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            self.pipeline.unet.set_attn_processor(lora_attn_procs)

            unet = self.wrap_unet(self.pipeline.unet.attn_processors)
        else:
            unet = self.pipeline.unet

        # Copy pop_size unet as populations
        self.unets = [deepcopy(unet) for _ in range(pop_size)]

        # set up diffusers-friendly checkpoint saving with Accelerate
        def save_model_hook(models, weights, output_dir):
            assert len(models) == self.pop_size
            for i in range(self.pop_size):
                if config.use_lora and isinstance(models[i], AttnProcsLayers):
                    self.pipeline.unet.save_attn_procs(os.path.join(output_dir, f"agent{i}"))
                elif not config.use_lora and isinstance(models[i], UNet2DConditionModel):
                    models[i].save_pretrained(os.path.join(output_dir, "unet"))
                else:
                    raise ValueError(f"Unknown model type {type(models[0])}")
                weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            if config.use_lora and isinstance(models[0], AttnProcsLayers):
                # pipeline.unet.load_attn_procs(input_dir)
                tmp_unet = UNet2DConditionModel.from_pretrained(
                    config.pretrained.model,
                    revision=config.pretrained.revision,
                    subfolder="unet",
                )
                tmp_unet.load_attn_procs(input_dir)
                models[0].load_state_dict(
                    AttnProcsLayers(tmp_unet.attn_processors).state_dict()
                )
                del tmp_unet
            elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                del load_model
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            models.pop()  # ensures that accelerate doesn't try to handle loading of the model

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        self.optimizers = [torch.optim.AdamW(
            self.unets[i].parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        ) for i in range(self.pop_size)]

        # generate negative prompt embeddings
        self.neg_prompt_embed = self.pipeline.text_encoder(
            self.pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
        self.train_neg_prompt_embeds = self.neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

        # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = contextlib.nullcontext if self.config.use_lora else self.accelerator.autocast
        # autocast = accelerator.autocast

        # Prepare everything with our `accelerator`.
        for i in range(self.pop_size):
            self.unets[i], self.optimizers[i] = self.accelerator.prepare(self.unets[i], self.optimizers[i])

        self.global_steps = [0 for _ in range(self.pop_size)]
        self.info1 = None # for logging
        self.info2 = None # for logging

    def wrap_unet(self, unet_attn_procs):
        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        pipeline = self.pipeline
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)
        return _Wrapper(unet_attn_procs)

    def unwrap_for_copy(self):
        unwarpped_model = self.accelerator.unwrap_model(self.pipeline.unet)
        return unwarpped_model.attn_processors

    def sampling(self, id, prompt_fn, reward_fns, executor, epoch):
        lora_attn_procs = {}
        for lora_id, lora_name in self.unets[id].mapping.items():
            lora_attn_procs[lora_name] = self.unets[id].layers[lora_id]
        self.pipeline.unet.set_attn_processor(lora_attn_procs)
        self.unets[id] = self.wrap_unet(self.pipeline.unet.attn_processors)
        self.optimizers[id] = torch.optim.AdamW(
            self.unets[id].parameters(),
            lr=self.config.train.learning_rate,
            betas=(self.config.train.adam_beta1, self.config.train.adam_beta2),
            weight_decay=self.config.train.adam_weight_decay,
            eps=self.config.train.adam_epsilon,)
        self.unets[id], self.optimizers[id] = self.accelerator.prepare(self.unets[id], self.optimizers[id])
        self.accelerator._models.pop(0)
        self.accelerator._optimizers.pop(0)
        self.pipeline.unet.eval()
        self.samples = []
        self.prompts = []
        self.info1 = defaultdict(list)
        for i in tqdm(
            range(self.config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[
                    prompt_fn(**self.config.prompt_fn_kwargs)
                    for _ in range(self.config.sample.batch_size)
                ]
            )

            # encode prompts
            prompt_ids = self.pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]

            # sample
            with self.autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    self.pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample.num_steps,
                    guidance_scale=self.config.sample.guidance_scale,
                    eta=self.config.sample.eta,
                    output_type="pt",
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.pipeline.scheduler.timesteps.repeat(
                self.config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            #rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            m_rewards = [executor.submit(r, images, prompts, prompt_metadata) for r in reward_fns]
            # yield to to make sure reward computation starts
            time.sleep(0)

            self.samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": m_rewards,
                }
            )

        # wait for all rewards to be computed
        m_rewards = []
        for sample in tqdm(
            self.samples,
            desc="Waiting for rewards",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            rewards = [r.result()[0].cpu().numpy() if isinstance(r.result()[0], torch.Tensor) else r.result()[0] for r in sample["rewards"]]
            rewards = np.stack(rewards, axis=1)
            m_rewards.append(rewards)
            sample["scaled_rewards"] = torch.as_tensor(self.weights @ np.array(rewards).T, device=self.accelerator.device)
            sample["rewards"] = torch.as_tensor(rewards, device=self.accelerator.device)
        
        m_rewards = np.stack(m_rewards, axis=1)
        mean_rewards = np.mean(np.mean(m_rewards, axis=0), axis=0)
        
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        self.samples = {k: torch.cat([s[k] for s in self.samples]) for k in self.samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            # self.info1[f"{id} images"] = [
            #             wandb.Image(
            #                 os.path.join(tmpdir, f"{i}.jpg"),
            #                 caption=f"{prompt:.25} | {reward[0]:.2f} | {reward[1]:.2f}",
            #             )
            #             for i, (prompt, reward) in enumerate(
            #                 zip(prompts, m_rewards.reshape(-1, len(reward_fns)))
            #             )  # only log rewards from process 0
            #         ]
            self.accelerator.log(
                {
                    f"{id} images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward[0]:.2f} | {reward[1]:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, m_rewards.reshape(-1, len(reward_fns)))
                        )  # only log rewards from process 0
                    ],
                },
                step=self.global_steps[id],
            )
            
        # gather scalar rewards across processes
        rewards = self.accelerator.gather(self.samples["scaled_rewards"]).cpu().numpy()

        # log rewards and images
        # self.info1[f"{id} weighted reward_mean"] = rewards.mean()
        # self.info1[f"{id} weighted reward_std"] = rewards.std()
        self.accelerator.log(
            {
                "epoch": epoch,
                f"{id} weighted reward_mean": rewards.mean(),
                f"{id} weighted reward_std": rewards.std(),
            },
            step=self.global_steps[id],
        )

        # per-prompt mean/std tracking
        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(self.samples["prompt_ids"]).cpu().numpy()
            prompts = self.pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        self.samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del self.samples["rewards"]
        del self.samples["prompt_ids"]

        return mean_rewards

    def training(self, id, epoch):
        total_batch_size, num_timesteps = self.samples["timesteps"].shape
        assert (
            total_batch_size
            == self.config.sample.batch_size * self.config.sample.num_batches_per_epoch
        )
        assert num_timesteps == self.config.sample.num_steps
        for inner_epoch in range(self.config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in self.samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=self.accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, self.config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            self.pipeline.unet.train()
            self.info2 = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                if self.config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [self.train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                num_train_timesteps = int(self.config.sample.num_steps * self.config.train.timestep_fraction)
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                ):
                    with self.accelerator.accumulate(self.unets[id]):
                        with self.autocast():
                            if self.config.train.cfg:
                                noise_pred = self.unets[id](
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + self.config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = self.unets[id](
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                self.pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=self.config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -self.config.train.adv_clip_max,
                            self.config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - self.config.train.clip_range,
                            1.0 + self.config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        self.info2[f"{id} approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        self.info2[f"{id} clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > self.config.train.clip_range
                                ).float()
                            )
                        )
                        self.info2[f"{id} loss"].append(loss)

                        # backward pass
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.unets[id].parameters(), self.config.train.max_grad_norm
                            )
                        self.optimizers[id].step()
                        self.optimizers[id].zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % self.config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        self.info2 = {k: torch.mean(torch.stack(v)) for k, v in self.info2.items()}
                        self.info2 = self.accelerator.reduce(self.info2, reduction="mean")
                        self.info2.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        self.accelerator.log(self.info2, step=self.global_steps[id])
                        if id == self.pop_size - 1:
                            for idx in range(self.pop_size):
                                self.global_steps[idx]+=1
                        self.info2 = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert self.accelerator.sync_gradients

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()