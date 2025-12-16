Step1：

Clone the project, set up the environment, and download checkpoints based on the `cosmos-transfer1` official version.
https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/INSTALL.md

Step2:

Customize the checkpoint download path outside the `server` folder to prevent checkpoints from being built into the image, facilitating subsequent mounting.

Step3:

Bulid docker：
According to:`carla0.9.16/PythonAPI/examples/nvidia/cosmos/server/README_SERVER.md"`
--bash make docker.sh
