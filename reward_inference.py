# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os

# load model
device = "cpu"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
    
    return scores.cpu().tolist()


def main():
    """Main function for calculating reward scores for given prompt and image."""

    # lora_rank = 8
    # lora_type = 'all'
    # img_base_path = '/home/aniruddh_ramrakhyani/cs234_proj/cs234_project/all_lora_rank8'

    # lora_rank = 8
    # lora_type = 'unet'
    # img_base_path = '/home/aniruddh_ramrakhyani/cs234_proj/cs234_project/lora_unet_rank8'

    lora_rank = 4
    lora_type = 'unet'
    img_base_path = '/home/aniruddh_ramrakhyani/cs234_proj/cs234_project'
    
    lora_scales = [0.0]

    pil_images = []
    prompts = []

    num_images_considered = 0
    num_images_not_found = 0

    prompts_filename = {
        'pokemon_with_blue_eyes.png': 'pokemon with blue eyes',
        'dragon_pokemon_red_coat.png': 'A dragon Pokemon wearing red coat',
        'attacking_pikachu.png': 'an attacking pikachu',
        'pikachu_holding_two_green_balls.png': 'pikachu holding two green balls',
        'pink_and_blue_bird_pokemon.png': 'pink and blue bird pokemon',
        'red_pokemon_with_blue_eyes.png': 'Red pokemon with blue eyes',
        'pikachu_bowing_dragon.png': 'pikachu bowing down to a dragon pokemon in front of a mountain and lake',
        'dragon_sword_mountain.png': 'dragon pokemon polishing his sword on top of snow covered mountain',
        'half_bird_half_dragon.png': 'half bird half dragon pokemon flapping his wings in front of a lake',
        'rain_dancing.png': 'pikachu dancing with a red and black dragon in rain on a green grass field',
        'bird_sunset.png': 'yellow and black bird pokemon sitting on a tree and looking at the sunset',
        'dragon_pokemon_eiffel_tower.png': 'dragon pokemon in front of eiffel tower',
        'pikachu_eiffel_tower.png': 'pikachu in front of eiffel tower',
        'dragon_space.png': 'dragon pokemon in space spewing fire on earth',
        'dragon_attacking_dragon.png': 'dragon pokemon attacking another dragon pokemon with fire',
        'meditation.png': 'pokemon meditating under a banyan tree',
        'red_apple.png': 'pokemon and dragon eating red apple in park',
        'pikachu_boat.png': 'pikachu in a boat in a ocean with whales',
        'pikachu_dragon_sheild.png': 'pikachu sheilding from fire attack from dragon using a wooden plank',
        'green_flower.png': 'green pokemon with a red flower around the neck and big blue eyes',
        'pikachu_sunglasses.png': 'pikachu wearing sunglasses on a beach',
        'attacking_building.png': 'blue pokemon attacking a tall orange building',
    }

    
    for filename, prompt in prompts_filename.items():
        for scale in lora_scales:
            full_filename = 'lora_rank' + str(lora_rank) + '_' + lora_type + '_scale_' + str(scale) + '_' + filename
            img_path = os.path.join(img_base_path, full_filename)
            print(img_path)

            if os.path.isfile(img_path):
                num_images_considered += 1
                img = Image.open(img_path)
                pil_images.append(img)
                prompts.append(prompt)
            else:
                num_images_not_found += 1

    if num_images_considered > 0:
        rewards = calc_probs(prompt, pil_images)
        avg_reward = sum(rewards) / len(rewards)
        print('avg. reward for ', lora_type, ' rank: ', str(lora_rank), ', scale: ',  str(lora_scales[0]), ' is: ', avg_reward)
        
    print('num_images_considered: ', num_images_considered)
    print('num_images_not_found: ', num_images_not_found)


if __name__ == "__main__":
    main()
