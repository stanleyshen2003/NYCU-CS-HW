from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import torch
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
import numpy as np
from functools import partial
import tqdm
from PIL import Image
from copy import deepcopy
import torchvision
from ddpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import wandb.plot
import ddpo.prompts
import ddpo.rewards
from ddpo.ddpo_agent import DDPOAgent

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

def run(_):
    config = FLAGS.config
    config.resume_from = "logs/2024.06.04_16.01.53/checkpoints/evol_195"
    pareto_agents = list(filter(lambda x: "agent" in x, os.listdir(config.resume_from)))
    pareto_agents = [os.path.join(config.resume_from, f) for f in sorted(pareto_agents, key=lambda x: int(x.split('.')[0][5:]))]
    eval_pareto = np.load(os.path.join(config.resume_from, "evaluations.npy"))
    print(eval_pareto)
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        logging_dir=os.path.join(config.logdir, config.run_name, "logs"),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={config.log_type: {"name": config.run_name}},
        )

    agents = DDPOAgent(pop_size=1, config=config, weights=np.array([[1.0, 0]]), accelerator=accelerator)
    prompt = "cat"
    for i, unet_path in enumerate(pareto_agents):
        agents.pipeline.unet.set_attn_processor(torch.load(unet_path, map_location=accelerator.device))
        agents.pipeline.unet.eval()
        image = agents.pipeline(prompt).images[0]
        image.save(f"images/pareto_{i}.png")
        
if __name__ == "__main__":
    app.run(run)