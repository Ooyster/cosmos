# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import gc
import json
import os
import pathlib
import torch
# import sys
# sys.path.append("/workspace/repos/cosmos-transfer1")

import sys, importlib

# # ✅ 1. 确保新路径在搜索路径最前
# sys.path.insert(0, "/workspace/repos/cosmos-transfer1")

# # ✅ 2. 删除旧的 cosmos_transfer1 缓存（关键）
# for name in list(sys.modules.keys()):
#     if name.startswith("cosmos_transfer1"):
#         del sys.modules[name]


from cosmos_transfer1.checkpoints import BASE_7B_CHECKPOINT_AV_SAMPLE_PATH, BASE_7B_CHECKPOINT_PATH
from cosmos_transfer1.diffusion.inference.inference_utils import default_model_names
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.io import save_video

"""
This module wrapper classes required for the transfer pipeline to work with model server/worker classes.
The pipeline wrapper maintains loaded models across multiple inferences for better performance,
unlike the demo function which discards the pipeline after each inference.

Key Components:
    - TransferValidator: Validates and processes inference parameters
    - WorkerPipeline: Base interface for model server/worker
    - TransferPipeline: pipeline wrapper implementation for video transfer
"""

# todo "keypoint" is causing dependency issue
hint_keys = {"vis", "seg", "edge", "depth"}
hint_keys_av = {"hdmap", "lidar"}
default_prompt = "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution."
default_negative_prompt = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."


class TransferValidator:
    """Validates and processes inference paramters.

    This class handles inference parameter validation and validation of controlnet specifications.
    This class allows to use and test the validation independently from the pipeline.

    Args:
        hint_keys (set): Valid hint keys for controlnet specifications

    Attributes:
        valid_keys (set): Set of valid controlnet hint keys

    """

    def __init__(self, hint_keys=hint_keys):
        self.valid_keys = hint_keys

    def extract_params(self, controlnet_specs):
        args_dict = {}
        controlnet_specs_clean = {}
        for key, val in controlnet_specs.items():
            if key in self.valid_keys:
                # 只保留允许的控制键
                controlnet_specs_clean[key] = val
            else:
                # 非控制键下放到 args_dict
                args_dict[key] = val
        return args_dict, controlnet_specs_clean

    def validate_control_spec(self, controlnet_specs_clean):
        for key in controlnet_specs_clean:
            if key not in self.valid_keys:
                raise ValueError(f"Invalid control key: {key}")
        for key, config in controlnet_specs_clean.items():
            if "control_weight" not in config:
                raise ValueError(f"Missing control_weight for {key}")
        return True

 
    def validate_params(
        self,
        controlnet_specs,
        input_video=None,
        prompt=default_prompt,
        negative_prompt=default_negative_prompt,
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
        output_dir: str = "outputs/",
        num_input_frames: int = 1,
        num_video_frames: int | None = None,
        original_input_is_folder: bool = False,
        height: int | None = None,   # <-- added
        width: int | None = None,    # <-- added
    ):
        args_dict = {}
        if input_video:
            args_dict["input_video"] = input_video
        args_dict["prompt"] = prompt
        args_dict["negative_prompt"] = negative_prompt
        args_dict["guidance"] = guidance
        args_dict["num_steps"] = num_steps
        args_dict["seed"] = seed
        args_dict["sigma_max"] = sigma_max
        args_dict["blur_strength"] = blur_strength
        args_dict["canny_threshold"] = canny_threshold
        args_dict["output_dir"] = output_dir
        args_dict["num_input_frames"] = num_input_frames
        if num_video_frames is not None:
            args_dict["num_video_frames"] = int(num_video_frames)
        args_dict["original_input_is_folder"] = bool(original_input_is_folder)

        if height is not None:
            if height % 32 != 0:
                raise ValueError(f"height {height} must be divisible by 32")
            args_dict["height"] = height
        if width is not None:
            if width % 32 != 0:
                raise ValueError(f"width {width} must be divisible by 32")
            args_dict["width"] = width

        self.validate_control_spec(controlnet_specs)
        args_dict["controlnet_specs"] = controlnet_specs
        return args_dict

    def parse_and_validate(self, param_dict: dict):
        # 若顶层包含 controlnet_specs，拆分
        controlnet_specs = param_dict.pop("controlnet_specs", {})
        args_dict, controlnet_specs_clean = self.extract_params(controlnet_specs)
        # 将其余非控制键参数补入 args_dict
        for k, v in param_dict.items():
            if k not in args_dict:
                args_dict[k] = v
        full_dict = self.validate_params(
            controlnet_specs=controlnet_specs_clean,
            **args_dict,
        )
        return full_dict

    def prune_and_validate(self, controlnet_specs, **kwargs):
        """
        Prune the controlnet_specs dictionary to only include valid keys and validate the values.
        """
        _, controlnet_specs_clean = self.extract_params(controlnet_specs)
        full_dict = self.validate_params(
            controlnet_specs=controlnet_specs_clean,
            **kwargs,
        )
        return full_dict


class TransferPipeline:
    """Main transfer pipeline implementation for video-to-video generation.

    This pipeline maintains loaded Cosmos models for efficient video transfer inference.
    Models are kept loaded to avoid repeated initialization overhead.

    The pipeline dynamically updates controlnet configurations and reloads models
    only when necessary, optimizing for inference speed over memory usage.

    Args:
        num_gpus (int): Number of GPUs for distributed inference (default: 1)
        checkpoint_dir (str): Directory containing model checkpoints
        checkpoint_name (str): Specific checkpoint file to load
        hint_keys (set): Valid controlnet hint keys for this pipeline
    """

    def __init__(
        self,
        num_gpus: int = 1,
        checkpoint_dir: str = "/mnt/pvc/cosmos-transfer1",
        checkpoint_name=BASE_7B_CHECKPOINT_PATH,
        hint_keys=hint_keys,
    ):
        self.device_rank = 0
        self.process_group = None

        self.preprocessors = Preprocessors()

        if num_gpus > 1:
            from megatron.core import parallel_state

            from cosmos_transfer1.utils import distributed

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
            self.process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(self.process_group)

        # TODO FIXME: we want to run W/O offloading. therefore we need to give the model at least one control input.
        self.valid_hint_keys = hint_keys
        first_key = next(iter(self.valid_hint_keys))
        self.control_inputs = {
            first_key: {
                "ckpt_path": os.path.join(checkpoint_dir, default_model_names[first_key]),
                "control_weight": 0.5,
            },
        }

        self.checkpoint_dir = checkpoint_dir
        self.video_save_name = "output"

        self.pipeline = DiffusionControl2WorldGenerationPipeline(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            control_inputs=self.control_inputs,
            process_group=self.process_group,
            offload_network=False,
            offload_text_encoder_model=False,
            offload_guardrail_models=False,
            offload_prompt_upsampler=False,
            upsample_prompt=False,
            fps=24,
            num_input_frames=1,
            disable_guardrail=True,
        )

    def update_controlnet_spec(
        self,
        checkpoint_dir: str,
        controlnet_specs: dict,
    ):
        """
        Create the controlnet specification defines which control netwworks are active.
        Note that controlnets are active even if the weights are set to 0."""

        config_changed = False

        for hint_key in self.valid_hint_keys:
            if hint_key in controlnet_specs:
                if hint_key not in self.control_inputs:
                    config_changed = True

                # overwrite old parameters
                self.control_inputs[hint_key] = copy.deepcopy(controlnet_specs[hint_key])
                self.control_inputs[hint_key]["ckpt_path"] = os.path.join(checkpoint_dir, default_model_names[hint_key])
            elif hint_key in self.control_inputs:
                # remove old parameters
                del self.control_inputs[hint_key]
                config_changed = True

        log.info(f"{config_changed=}, control_inputs: {json.dumps(self.control_inputs, indent=4)}")

        return config_changed

    def infer(self, args: dict):
        return self.generate(**args)

    def generate(
        self,
        controlnet_specs,
        input_video=None,
        prompt="",
        negative_prompt="",
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
        num_input_frames: int = 1,
        output_dir: str = "outputs/",
        num_video_frames: int | None = None,
        original_input_is_folder: bool = False,
        height: int | None = None,   # <-- added
        width: int | None = None,    # <-- added
    ):
        if height and width:
            log.info(f"Overriding pipeline resolution to {height}x{width}")
            self.pipeline.height = height
            self.pipeline.width = width
        # 配置帧数
        if num_video_frames is not None:
            self.pipeline.num_video_frames = num_video_frames
            if self.pipeline.num_video_frames == 1:
                self.pipeline.num_input_frames = 0
            else:
                self.pipeline.num_input_frames = num_input_frames

        controlnet_specs = controlnet_specs or {}
        config_changed = self.update_controlnet_spec(
            checkpoint_dir=self.checkpoint_dir,
            controlnet_specs=controlnet_specs,
        )
        if config_changed:
            self.pipeline.reload_model(self.control_inputs)

        current_control_inputs = copy.deepcopy(self.control_inputs)

        log.info("Running preprocessor")
        self.preprocessors(
            input_video=input_video,
            input_prompt=prompt,
            control_inputs=current_control_inputs,
            output_folder=output_dir,
            blur_strength=blur_strength,
            canny_threshold=canny_threshold,
        )

        # 单图情况修正
        if pathlib.Path(input_video).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            self.pipeline.num_video_frames = 1
            self.pipeline.num_input_frames = 0

        # 清空区域提示
        if hasattr(self.pipeline, "regional_prompts"):
            self.pipeline.regional_prompts = []
        if hasattr(self.pipeline, "region_definitions"):
            self.pipeline.region_definitions = None

        self.pipeline.guidance = guidance
        self.pipeline.num_steps = num_steps
        self.pipeline.seed = seed
        self.pipeline.sigma_max = sigma_max
        self.pipeline.blur_strength = blur_strength
        self.pipeline.canny_threshold = canny_threshold

        batch_outputs = self.pipeline.generate(
            prompt=prompt,
            video_path=input_video,
            negative_prompt=negative_prompt,
            control_inputs=current_control_inputs,
            save_folder=output_dir,
            batch_size=1,
        )
        if batch_outputs is None:
            log.critical("Generation blocked by guardrail.")
            return

        if self.device_rank == 0:
            videos, final_prompts = batch_outputs
            for vid, final_p in zip(videos, final_prompts):
                T, H, W, C = vid.shape
                from PIL import Image
                if T == 1:
                    Image.fromarray(vid[0]).save(os.path.join(output_dir, f"{self.video_save_name}.png"))
                else:
                    # 主视频
                    from cosmos_transfer1.utils.io import save_video
                    save_video(
                        video=vid,
                        fps=self.pipeline.fps,
                        H=H,
                        W=W,
                        video_save_quality=5,
                        video_save_path=os.path.join(output_dir, f"{self.video_save_name}.mp4"),
                    )
                    if original_input_is_folder:
                        frames_dir = os.path.join(output_dir, "frames")
                        os.makedirs(frames_dir, exist_ok=True)
                        for i in range(T):
                            Image.fromarray(vid[i]).save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
                with open(os.path.join(output_dir, f"{self.video_save_name}.txt"), "w", encoding="utf-8") as f:
                    f.write(final_p)
#保持 pipeline 单例（避免每次脚本重启导致模型重载）。

    def cleanup(self, cfg):
        """Clean up resources"""
        if cfg.num_gpus > 1:
            import torch.distributed as dist
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def create_transfer_pipeline(cfg, create_model=True):
    """Factory function to create transfer pipeline and validator.

    Args:
        cfg: Configuration object with model settings including checkpoint_dir
        create_model (bool): Whether to actually create the model pipeline (default: True)

    Returns:
        tuple: (pipeline, validator) - TransferPipeline instance and TransferValidator
    """
    log.info(f"Initializing model using factory function {cfg.factory_module}.{cfg.factory_function}")

    pipeline = None
    if create_model:
        pipeline = TransferPipeline(
            num_gpus=int(os.environ.get("WORLD_SIZE", 1)),
            checkpoint_dir=cfg.checkpoint_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()

    validator = TransferValidator(hint_keys=hint_keys)
    return pipeline, validator


def create_transfer_pipeline_AV(cfg, create_model=True):
    """Factory function to create AV-specific transfer pipeline and validator.

    Creates a pipeline configured for autonomous vehicle data with specialized
    hint keys (hdmap, lidar) and AV sample checkpoint.

    Args:
        cfg: Configuration object with model settings including checkpoint_dir
        create_model (bool): Whether to actually create the model pipeline (default: True)

    Returns:
        tuple: (pipeline, validator) - AV-configured TransferPipeline and TransferValidator
    """
    log.info(f"Initializing model using factory function {cfg.factory_module}.{cfg.factory_function}")

    pipeline = None
    if create_model:
        pipeline = TransferPipeline(
            num_gpus=int(os.environ.get("WORLD_SIZE", 1)),
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
            hint_keys=hint_keys_av,
        )
        gc.collect()
        torch.cuda.empty_cache()

    validator = TransferValidator(hint_keys=hint_keys_av)
    return pipeline, validator
