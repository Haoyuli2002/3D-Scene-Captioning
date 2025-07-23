import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import AutoConfig

model_path = "OpenGVLab/InternVL3-2B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

prompt_prefix = ""
for i in range(5):
    prompt_prefix += f"View {i+1}: <image>\n"

prompt = (
    prompt_prefix +
    '''
    You are provided with 5 multi-view images of a single object.
    This object is surrounded by a green bounding box in the images.
    Please write one short but detailed sentence describing only the main object.  
    The description should focus on the object's distinguishing features: shape, color, texture, structural details, number of holes or buttons, or any properties or attributes.  
    Avoid generic phrases like "a single object" or "rectangular shape" without detail.  
    Now generate the object caption:
    '''
)

prompt_category = (
    prompt_prefix +
    '''
    You are provided with 5 multi-view images of a single object.
    This object is surrounded by a green bounding box in the images.
    Your task is to categorize the object into a single, clear category or class label (for example: "mug", "chair", "bottle").
    The output should be only the object category, without any additional text or description.
    If you cannot identify the object or if it is too ambiguous to categorize confidently, answer with "NaN".
    Now output only the object category:
    '''
)

prompt_surrounding = (
    prompt_prefix +
    '''
    You are provided with multiple views of a specific object which is enclosed in a green bounding box. These views also show its immediate surroundings and neighboring objects.

    Your task is to write a structured description of the surrounding scene context of the object.

    Use the following output format:
    - The first sentence should briefly describe **what is the main object**.
    - Each subsequent sentence should describe **one spatial or functional relationship** between this object and other nearby objects.

    You may mention background elements, relative positions (e.g., "next to", "above", "in front of"), and possible usage or functional relationships.

    For example, your output should look like this:
        This object is a white door with a classic six-panel design.  
        It is positioned next to a dark-colored couch.  
        It is located beneath a wall-mounted world map.  
        It appears to serve as a main entryway within a domestic space.
        ...

    Each sentence must describe exactly one relation or feature.
    Now generate the caption:
    '''
)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_path(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class InternVL_VLM():
    def __init__(self, model_path=model_path, device_map=None):
        self.model = AutoModel.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True,
                                                use_flash_attn=True,
                                                trust_remote_code=True,
                                                device_map="auto").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.input_size = 448

    def image_preprocess(self, images):
        all_pixel_values = []
        num_patches_list = []

        for img in images:
            pixels = load_image(img).to(torch.bfloat16).cuda()
            all_pixel_values.append(pixels)
            num_patches_list.append(pixels.size(0))

        # 拼接
        pixel_values = torch.cat(all_pixel_values, dim=0)
        return pixel_values, num_patches_list

    def generate_caption(self, pixel_values, generation_type = 'object-level', generation_config=None, num_patches_list=None):
        if generation_config is None:
            generation_config = dict(max_new_tokens=256)
        promt_input = None
        if generation_type == 'object-level':
            promt_input = prompt
        elif generation_type == 'object-category':
            promt_input = prompt_category
        elif generation_type == 'object-surrounding':
            promt_input = prompt_surrounding
        else:
            raise ValueError("Invalid generation type. Choose from 'object-level', 'object-category', or 'object-surrounding'.")

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            promt_input,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )
        return response