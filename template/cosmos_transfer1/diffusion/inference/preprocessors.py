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

import json
import os
import pathlib

from cosmos_transfer1.auxiliary.depth_anything.model.depth_anything import DepthAnythingModel
from cosmos_transfer1.auxiliary.edge_control.edge_control import EdgeControlModel
from cosmos_transfer1.auxiliary.human_keypoint.human_keypoint import HumanKeypointModel
from cosmos_transfer1.auxiliary.sam2.sam2_model import VideoSegmentationModel
from cosmos_transfer1.auxiliary.vis_control.vis_control import VisControlModel
from cosmos_transfer1.diffusion.inference.inference_utils import valid_hint_keys
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.video_utils import is_valid_video, video_to_tensor


class Preprocessors:
    """Preprocessor class to handle input control generation for various modalities.
    Note that this class will run on each rank, so each file name must be unique per rank to avoid potential file corruption.
    """

    def __init__(self):
        self.depth_model = None
        self.seg_model = None
        self.keypoint_model = None
        self.vis_model = None
        self.edge_model = None

    def __call__(
        self,
        input_video,
        input_prompt,
        control_inputs,
        output_folder,
        regional_prompts=None,
        blur_strength="medium",
        canny_threshold="medium",
    ):
        os.makedirs(output_folder, exist_ok=True)
        rank = int(os.environ.get("LOCAL_RANK", 0))

        for hint_key, cfg in control_inputs.items():
            if hint_key not in valid_hint_keys:
                continue

            # 生成缺失的 input_control
            if hint_key in ["depth", "seg", "keypoint", "vis", "edge"]:
                self.gen_input_control(
                    in_video=input_video,
                    in_prompt=input_prompt,
                    hint_key=hint_key,
                    control_input=cfg,
                    output_folder=output_folder,
                    blur_strength=blur_strength,
                    canny_threshold=canny_threshold,
                )

            # 控制权重 mask 生成 (目前仅 seg)
            if cfg.get("control_weight_prompt") and hint_key == "seg":
                prompt = cfg["control_weight_prompt"]
                log.info(f"{hint_key}: generating control weight tensor with SAM using prompt={prompt}")
                is_image = (
                    os.path.isfile(input_video)
                    and pathlib.Path(input_video).suffix.lower() in [".png", ".jpg", ".jpeg"]
                )
                tensor_path = os.path.join(
                    output_folder, f"{hint_key}_control_weight_{rank}.pt"
                )
                video_or_image_path = os.path.join(
                    output_folder,
                    f"{hint_key}_control_weight_{rank}{'.png' if is_image else '.mp4'}",
                )
                weight_scaler = cfg["control_weight"] if isinstance(cfg.get("control_weight"), float) else 1.0
                self.segmentation(
                    in_video=input_video,
                    out_tensor=tensor_path,
                    out_video=video_or_image_path,
                    prompt=prompt,
                    weight_scaler=weight_scaler,
                    binarize_video=True,
                )
                cfg["control_weight"] = tensor_path
        if regional_prompts and len(regional_prompts):
            log.info(f"processing regional prompts: {regional_prompts}")
            for i, regional_prompt in enumerate(regional_prompts):
                log.info(f"generating regional context for {regional_prompt}")
                out_tensor = os.path.join(
                    output_folder, f"regional_context_r{int(os.environ.get('LOCAL_RANK', 0))}_{i}.pt"
                )
                if "mask_prompt" in regional_prompt:
                    prompt = regional_prompt["mask_prompt"]
                    out_video = os.path.join(
                        output_folder, f"regional_context_r{int(os.environ.get('LOCAL_RANK', 0))}_{i}.mp4"
                    )
                    self.segmentation(
                        in_video=input_video,
                        out_tensor=out_tensor,
                        out_video=out_video,
                        prompt=prompt,
                        weight_scaler=1.0,
                        legacy_mask=True,
                    )
                    if os.path.exists(out_tensor):
                        regional_prompt["region_definitions_path"] = out_tensor
                elif "region_definitions_path" in regional_prompt and isinstance(
                    regional_prompt["region_definitions_path"], str
                ):
                    if is_valid_video(regional_prompt["region_definitions_path"]) or \
                       pathlib.Path(regional_prompt["region_definitions_path"]).suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        video_to_tensor(regional_prompt["region_definitions_path"], out_tensor)
                        regional_prompt["region_definitions_path"] = out_tensor
                    else:
                        raise ValueError(f"Invalid video file: {regional_prompt['region_definitions_path']}")
                else:
                    log.info("do nothing!")

        return control_inputs

    # def gen_input_control(
    #     self,
    #     in_video,
    #     in_prompt,
    #     hint_key,
    #     control_input,
    #     output_folder,
    #     blur_strength="medium",
    #     canny_threshold="medium",
    # ):
    #     # if input control isn't provided we need to run preprocessor to create input control tensor
    #     # for depth no special params, for SAM we need to run with prompt
    #     if control_input.get("input_control", None) is None:
    #         out_video = os.path.join(
    #             output_folder, f"{hint_key}_input_control_{int(os.environ.get('LOCAL_RANK', 0))}.mp4"
    #         )
    #         control_input["input_control"] = out_video
    #         if hint_key == "seg":
    #             prompt = control_input.get("input_control_prompt", in_prompt)
    #             prompt = " ".join(prompt.split()[:128])
    #             log.info(
    #                 f"no input_control provided for {hint_key}. generating input control video with SAM using {prompt=}"
    #             )
    #             self.segmentation(
    #                 in_video=in_video,
    #                 out_video=out_video,
    #                 prompt=prompt,
    #             )
    #         elif hint_key == "depth":
    #             log.info(
    #                 f"no input_control provided for {hint_key}. generating input control video with DepthAnythingModel"
    #             )
    #             self.depth(
    #                 in_video=in_video,
    #                 out_video=out_video,
    #             )
    #         elif hint_key == "vis":
    #             log.info(
    #                 f"no input_control provided for {hint_key}. generating input control video with VisControlModel"
    #             )
    #             self.vis(
    #                 in_video=in_video,
    #                 out_video=out_video,
    #                 blur_strength=blur_strength,
    #             )
    #         elif hint_key == "edge":
    #             log.info(
    #                 f"no input_control provided for {hint_key}. generating input control video with EdgeControlModel"
    #             )
    #             self.edge(
    #                 in_video=in_video,
    #                 out_video=out_video,
    #                 canny_threshold=canny_threshold,
    #             )
    #         else:
    #             log.info(f"no input_control provided for {hint_key}. generating input control video with Openpose")
    #             self.keypoint(
    #                 in_video=in_video,
    #                 out_video=out_video,
    #             )
    def gen_input_control(
        self,
        in_video,
        in_prompt,
        hint_key,
        control_input,
        output_folder,
        blur_strength="medium",
        canny_threshold="medium",
    ):
        if control_input.get("input_control"):
            return

        rank = int(os.environ.get("LOCAL_RANK", 0))
        is_image = (
            os.path.isfile(in_video)
            and pathlib.Path(in_video).suffix.lower() in [".png", ".jpg", ".jpeg"]
        )
        ext = ".png" if is_image else ".mp4"
        # 保持统一前缀, 避免下游假设失败
        out_path = os.path.join(
            output_folder, f"{hint_key}_input_control_{rank}{ext}"
        )

        if hint_key == "seg":
            seg_prompt = control_input.get("input_control_prompt", in_prompt)
            if seg_prompt:
                seg_prompt = " ".join(seg_prompt.split()[:128])
            log.info(f"generating {hint_key} control with SAM prompt={seg_prompt}")
            self.segmentation(
                in_video=in_video,
                prompt=seg_prompt,
                out_video=out_path,
                binarize_video=False,
                legacy_mask=False,
            )
        elif hint_key == "depth":
            log.info("generating depth control with DepthAnything")
            self.depth(in_video=in_video, out_video=out_path)
        elif hint_key == "vis":
            log.info("generating vis control")
            self.vis(in_video=in_video, out_video=out_path, blur_strength=blur_strength)
        elif hint_key == "edge":
            log.info("generating edge control")
            self.edge(in_video=in_video, out_video=out_path, canny_threshold=canny_threshold)
        elif hint_key == "keypoint":
            log.info("generating keypoint control")
            self.keypoint(in_video=in_video, out_video=out_path)
        else:
            log.info(f"hint_key {hint_key} unsupported for auto generation")
            return

        control_input["input_control"] = out_path
    def vis(self, in_video, out_video, blur_strength="medium"):
        if self.vis_model is None:
            self.vis_model = VisControlModel(blur_strength=blur_strength)

        if pathlib.Path(in_video).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            self.vis_model.process_image(in_video, out_video)
        else:
            self.vis_model(in_video, out_video)

    def edge(self, in_video, out_video, canny_threshold="medium"):
        if self.edge_model is None:
            self.edge_model = EdgeControlModel(canny_threshold=canny_threshold)

        if pathlib.Path(in_video).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            self.edge_model.process_image(in_video, out_video)
        else:
            self.edge_model(in_video, out_video)


    def depth(self, in_video, out_video):
        if self.depth_model is None:
            self.depth_model = DepthAnythingModel()
            
        if pathlib.Path(in_video).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            self.depth_model.process_image(in_video, out_video)
        else:
            self.depth_model(in_video, out_video)

    def keypoint(self, in_video, out_video):
        if self.keypoint_model is None:
            self.keypoint_model = HumanKeypointModel()
        if pathlib.Path(in_video).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            self.keypoint_model.process_image(in_video, out_video)
        else:
            self.keypoint_model(in_video, out_video)

    def segmentation(
        self,
        in_video,
        prompt,
        out_video=None,
        out_tensor=None,
        weight_scaler=None,
        binarize_video=False,
        legacy_mask=False,
    ):
        if self.seg_model is None:
            self.seg_model = VideoSegmentationModel()
        self.seg_model(
            input_video=in_video,
            output_video=out_video,
            output_tensor=out_tensor,
            prompt=prompt,
            weight_scaler=weight_scaler,
            binarize_video=binarize_video,
            legacy_mask=legacy_mask,
        )


if __name__ == "__main__":
    input_image = "example.png"
    control_inputs = {
        k: {"control_weight": 1.0, "input_control": None} for k in ["edge", "depth", "seg", "vis"]
    }
    pre = Preprocessors()
    pre(
        input_video=input_image,
        input_prompt="a cinematic mountain landscape",
        control_inputs=control_inputs,
        output_folder="tmp_controls",
        blur_strength="medium",
        canny_threshold="medium",
    )
    print(json.dumps(control_inputs, indent=4))