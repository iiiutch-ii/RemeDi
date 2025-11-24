# RemeDi – Anonymous Code Release

This repository contains the anonymized code used in our RemeDi experiments, including both supervised fine-tuning (SFT) and reinforcement learning (RL) pipelines.

## 📁 SFT Code

The SFT implementation is located in `SFT-code/`.

- **Training script**  
  ```bash
  SFT-code/scripts/lanuch_1machine_local.sh
  ```

- **Main configuration file**  
  ```text
  SFT-code/config/remask_sft/mdm_remask_sft_bitower_3stage_s2_4machine_binary_v2_1.yaml
  ```
  This config specifies:
  - The loss function implementation file
  - The model definition file

Please refer to this YAML file to locate the loss function and model architecture.


## 📁 RL Code

The RL implementation is located in `RL-code/`.

- **Training script**  
  ```bash
  RL-code/scripts/run.sh
  ```

- **Main configuration file**  
  ```text
  RL-code/config/remedi.yaml
  ```
  This config specifies:
  - The loss function implementation file  
  - The model definition file  

As with the SFT pipeline, please refer to this file to locate the RL loss and model definitions.

## 📦 Open-Sourced Models

We also provide the trained models used in our experiments:

- [🌟 RemeDi-Instruct (Hugging Face)](https://huggingface.co/iiiutch/RemeDi-Instruct)

- [🚀 RemeDi-RL (Hugging Face)](https://huggingface.co/iiiutch/RemeDi-RL)

  Inference code is provided in `inferece.py`
