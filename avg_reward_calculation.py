from datasets import load_dataset
from transformers import AutoProcessor, AutoModel

from PIL import Image
import io
import torch

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

        # calculate emperical mean
        reward_sum = torch.sum(scores)
        num_examples = scores.shape[0]
        emperical_mean = reward_sum / num_examples
        print("emperical_mean: ", emperical_mean)

        # calculate variance
        emperical_variance = torch.sum(torch.square(scores - emperical_mean) / num_examples)
        emperical_std_dev = torch.sqrt(emperical_variance)
        print("emperical_std_dev: ", emperical_std_dev)
    
    return reward_sum, num_examples


def main():

    dataset = load_dataset(
        "yuvalkirstain/pickapic_v1",
        None,
        streaming=True,
        )
    
    caption_column_name = "caption"
    img_col_name = "jpg_0"

    def tokenize_captions(examples, is_train=True):
        captions = []
        images = []

        # print('prev len(examples): ', len(examples))

        for i in range(len(examples)):
            captions.append(examples[caption_column_name][i])

            img = examples[img_col_name][i]
            img_obj = Image.open(io.BytesIO(img))
            images.append(img_obj)

        # print('len(images): ', len(images))
        # print('len(images): ', len(captions))
        return images, captions
    
    def preprocess_train(examples):
        examples[img_col_name], examples[caption_column_name] = tokenize_captions(examples)
        # print("after len(examples): ", len(examples))
        # print("after len(examples['image']): ", len(examples[img_col_name]))
        # print("after len(examples['caption']): ", len(examples[caption_column_name]))
        #print(examples)
        return examples

    column_names = dataset["train"].column_names
    columns_to_keep = [caption_column_name, img_col_name]
    cols_to_remove = [col for col in column_names if col not in columns_to_keep] 
    # print("column_names: ", column_names)
    # print("cols_to_remove: ", cols_to_remove)
    train_dataset = dataset["train"].map(
        preprocess_train,
        batched=True,
        remove_columns=cols_to_remove,
    )

    def collate_fn(examples):
        captions = [example[caption_column_name] for example in examples]
        images = [example[img_col_name] for example in examples]
        # print('hello len(images): ', len(images))
        # print('hello len(captions): ', len(captions))
              
        return {"images": images, "captions": captions}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # shuffle=True,
        collate_fn=collate_fn,
        batch_size=200,
        num_workers=1,
    )

    total_reward = 0.0
    total_num_examples = 0

    # progress_bar = tqdm(
    #     range(0, len(train_dataloader)),
    #     initial=0,
    #     desc="Steps",
    # )

    for step, batch in enumerate(train_dataloader):
        print("step: ", step)
        reward_sum, num_examples = calc_probs(batch["captions"], batch["images"])
        total_reward += reward_sum
        total_num_examples += num_examples
        # progress_bar.update(1)

    final_emperical_mean = total_reward / total_num_examples
    print("final_emperical_mean: ", final_emperical_mean)



if __name__ == "__main__":
    main()
