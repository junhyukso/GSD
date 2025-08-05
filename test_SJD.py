import argparse
import os
import sys

import os
PATH = '/SSD'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
print(sys.path)

import gc

from lumina_mgpt.inference_solver import FlexARInferenceSolver
from PIL import Image
import torch
import time

import random
import numpy as np

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


cache_dir = PATH

# model_path = "Alpha-VLLM/Lumina-mGPT-7B-512"
# target_size = 512

model_path = "Alpha-VLLM/Lumina-mGPT-7B-768"
target_size = 768
target_size_h, target_size_w = 768, 768

# model_path = "Alpha-VLLM/Lumina-mGPT-7B-1024"
# target_size = 1024

# model_path = "Alpha-VLLM/Lumina-mGPT-34B-512"
# target_size = 512

device = "cuda:0"

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolver(
    model_path=model_path,
    precision="bf16",
    target_size=target_size,
    cache_dir=cache_dir,
    device = device,
)

seeds = [None, ] #[_ for _ in range(124, 200) ]
max_num_new_tokens = 16
multi_token_init_scheme = 'random' # 'repeat_horizon'
image_top_k = 2000 
text_top_k = 10
guidance_scale = 3.0
prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'


### Prompt for appendix - more qualitative results
q_image_content_conditions = [ # Realistic, HumanFAce, Anime, GSD
    
    'close-up two birds on a tree branch, background of blue sky with cumulus clouds and rising sun, 4k, realistic',
    'Three penguins in yellow construction helmets, building a sandcastle on a tropical beach, one holding a blueprint, the ocean behind them glowing in soft blue hues under the setting sun, hyperrealistic textures, playful and cinematic',
    'Deep in the jungle where a rusty robot is abndoned , 4k ,realistic, photography',
    'animation art work, A cheese burger on the sky with birds, bright, detailed',
    'Apple castle on the grass, realistic, 4k, detailed, photography',
    'A mischievous hippo playing soccer, realistic, 4k, detailed, photography',
    'Truck full of vegetables, afternoon, 4k, photography, bright color,',
    'Masterpiece, 4k, photography, bright background, market selling fresh fruits',
    'photo, photography, realistic, very detailed, Amsterdam, center fancy sports car, afternoon, realistic. sharp, bright, film grain, high contrast',

    'dystopic civilization beautiful landscape, morning, woman, very intricate, very detailed, sharp, bright, colorful',
    'A single coffee on a dinner plate on a table, 4k, detailed, photography',
    'A cat in a lab coat, standing in front of a chalkboard full of complex equations, realistic, 4k',
    'Pixel art, A mushroom kingdom, glowing, masterpiece',
    'Japanese woman in a floral-pattern summer dress sitting on an old boat beached on a tropical island, overlooking a majestic azure blue ocean with gentle waves, landscape, sunset. Impressionistic',
    'a_skynet_cyberdyne_craft, the image is featuring a futuristic, highly advanced jet fighter drone flying rapidly at altitude thporugh stormclouds, silhouetted, chiascuro, sunset., realistic, 4k',

    'abstract oil painting, gradient vibrant neon colour, rough, textural, broad brush strokes, a sleek spaceship traversing interstellar space, detailed night sky with stars and nebulas',
    'photo, photography, Fujifilm XT-4 Viltrox, Budapest, Hungary landscape, sunset, very intricate, very detailed, realistic. sharp, bright, colorful, film grain, high contrast',
    'A stylized clay cartoon character, a small, adorable humanoid figure with a skull head, riding a miniature motorcycle., detailed',
    'animation art work, cute cat boxing with silly dog, bright',
    'Pumpkin carraige on the road, 4k, realistic, photography',
    'photography, photo of a war pilot walking to his war plane on sunset, taken from behind, 4k, realistic',

    'animation art work, huge sand castle made by dwarfs, 4k, realistic',
    '4k, realistic, photography, Giant Tree on the hill, afternoon',
]



template_condition_sentences = [
    f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n",
] * len(q_image_content_conditions)

from scheduler.jacobi_iteration_lumina_mgpt_sjd import renew_pipeline_sampler
print(inference_solver.__class__)
inference_solver = renew_pipeline_sampler(
    inference_solver,
    jacobi_loop_interval_l = 3,
    jacobi_loop_interval_r = (target_size // 16)**2 + target_size // 16 - 10,
    max_num_new_tokens = max_num_new_tokens,
    guidance_scale = guidance_scale,
    seed = seeds[0],
    multi_token_init_scheme = multi_token_init_scheme,
    do_cfg=  True,
    image_top_k=image_top_k, 
    text_top_k=text_top_k,
    prefix_token_sampler_scheme = prefix_token_sampler_scheme,
)

import os
from datetime import datetime

# Get the current time in the desired format
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')

# Define the folder name
folder_name = f"{current_time}"

os.makedirs(f"./SJD/"+folder_name)
for seed in seeds:
    inference_solver.model.seed = seed
    for i, q_image_content_condition in enumerate(q_image_content_conditions):
        q1 = template_condition_sentences[i] + q_image_content_condition

        output_file_name = model_path.split("/")[-1] + "-" + q_image_content_condition[:30] + '-' + str(max_num_new_tokens) + '-init-' + multi_token_init_scheme[:6] + '-seed' + str(seed) + '-img_topk' + str(image_top_k) + ".png"

        time_start = time.time()
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t1.record()

        generated = inference_solver.generate(
            images=[],
            qas=[[q1, None]],
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k),
        )
        t2.record()
        torch.cuda.synchronize()

        t = t1.elapsed_time(t2) / 1000
        time_end = time.time()
        print("Time elapsed: ", t, time_end - time_start)

        a1, new_image = generated[0], generated[1][0]
        result_image = inference_solver.create_image_grid([new_image], 1, 1)



        result_image.save(f"./SJD/{folder_name}/" + output_file_name)
        print(a1, 'saved', output_file_name) # <|image|>


del inference_solver
gc.collect()
