# Training Diffusion Models using Reinforcement Learning

This repo contains code for 2024 Spring CS234 final project: Training Diffusion Models using Reinforcement Learning. In this project, I implement Denoising Diffusion Policy Optimization paper [1](https://arxiv.org/abs/2305.13301) to finetune Stable Diffusion model to generate more prompt aligned and aesthetically appealing images.

The repo is organized as follows:

1. "train_text_to_image_lora.py" : This file is the main training script and contains code for following aspects:
    - Implements the training loop and the denoising loop
    - Invokes the reward model and normalizes the reward
    - Prepares the dataset and batches
    - Deals with checkpointing and validation

    **The code in this file was borrowed from hugging face for checkpointing and validation but I modified the file and implemented the DDPO algorithm, training loop, reward model invocation, reward normalization myself.**

2. "scheduling_ddpm.py" : The relevant function in this file is "step()". This function implements a denoising step of the pipeline given the noise     prediction from unet. **I modified this function to additionally return the logprob of the predicted sample which is used for calculating training loss.**

3. "pipeline_stable_diffusion.py" : The main function in this file is ""__call__()". This function implements the inference loop of Stable Diffusion. **The only change I made to this file was to disable the nsfw checker as it was (in most cases) erroneously detecting nsfw content and output a black image, thereby hampering proper evaluation.**

4. "reward_inference.py" : Calculates the reward obtained by the generated image for the given prompt using pick-a-pic reward model. File was created by me.

5. "avg_reward_calculation.py" : Script to calculate mean and variance of the reward on the training dataset. File was created by me.

6. "inference_script.py" : Script to run inference on the DDPO finetuned model. File was created by me.

**Code changes I made in the Hugging Face files, can be easily determined by searching for "cs234" string in the file.**


To run training, follow these steps:

1. Clone the diffusers repo from Hugging Face:
   ```
   git clone https://github.com/huggingface/diffusers
   cd diffusers
   pip install -e .
   ```

2. Replace files in diffusers with files in this repo:
   ```
   cd diffusers
   cp train_text_to_image_lora.py examples/text_to_image/train_text_to_image_lora.py
   cp scheduling_ddpm.py src/diffusers/schedulers/scheduling_ddpm.py
   cp pipeline_stable_diffusion.py src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
   ```

3. Run the following cmd to launch training:

```
cd diffusers/examples/text_to_image/

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --dataset_name="yuvalkirstain/pickapic_v1" --dataloader_num_workers=8 --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=1 --max_train_steps=1000 --learning_rate=1e-04 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir="/home/aniruddh_ramrakhyani/cs234/output_dir" --report_to=wandb --checkpointing_steps=500 --validation_prompt="Red pokemon with blue eyes." --seed=13 --caption_column="caption" --rank=4 --ddpo_num_inference_steps=12
```

To run inference using the trained weights:

1. Modify "inference_script.py" to the correct path containing LoRA finetuned weights.

2. Run the following cmd:
```
cd diffusers
cp inference_script.py .
python inference_script.py

```