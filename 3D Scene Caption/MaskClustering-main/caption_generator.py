import torch
from PIL import Image
import images_processing as img_process
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Caption_Generator():
    def __init__(self, model_name="AIDC-AI/Ovis2-1B", summarization_name = "Qwen/Qwen1.5-0.5B-Chat"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True
        ).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.model_id = summarization_name
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype="auto")
        self.summarizer_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def object_level_caption(self, query, images, max_partition=5):
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)

        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype,
                device=self.visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs
            )[0]

        output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output
    
    def image_processing_object_captioning(self, data = None, prompt = None, summary_prompt = None):
        for obj_id in tqdm(data, desc='Generating Object Level Captions:'):
            # Step 1: Get individual frames (images) and mask images of each object
            frame_ids = [frame for (frame, _, _) in data[obj_id]['repre_mask_list']]
            mask_ids = [mask for (_, mask, _) in data[obj_id]['repre_mask_list']]

            image_path = 'data/demo/scene0608_00/color_640'
            images = [Image.open(f'{image_path}/{id}.jpg') for id in frame_ids]

            mask_path = 'data/demo/scene0608_00/output/mask'
            masks = [
                Image.open(os.path.join(mask_path, f'{img}.png')).convert('L')
                for img in frame_ids
            ]

            # Step 2: Identify the Objects from the mask images based on the mask_id
            masked_images_padded = []
            for img, id, mask in zip(images, mask_ids, masks):
                binary_mask = (np.array(mask) == id).astype(np.uint8) * 255
                x_min, y_min, x_max, y_max = img_process.mask_to_bbox(binary_mask)
                bbox = (x_min, y_min, x_max, y_max)
                image = img.copy()
                # Crop the Images to obtain the object only
                cropped_img = img_process.crop_with_padding(image, bbox, padding_ratio=0.1)
                if len(masked_images_padded) < 5:
                    masked_images_padded.append(cropped_img)
            
            images_input = masked_images_padded
            max_partition = len(mask_ids)
            if prompt is None:
                prompt = '''
                You are provided with multiple views of a single object and the views are cropped to only showcase the object.

                Your task is to write **one short paragraph** describing only the main object.

                Focus on the object's physical appearance, attributes or properties. Do not mention background, surrounding objects, or the object's spatial position and the red bounding boxes.

                Now generate the object caption for the following object:
                '''
            query = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images_input))]) + '\n' + prompt

            caption = self.object_level_caption(query=query, images=images_input, max_partition=max_partition)

            if summary_prompt is None:
                summary_prompt = (
                    "You are an assistant tasked with extracting concise object descriptions from detailed scene captions.\n"
                    "Your goal is to output sentences describing only the **object itself**, including its: physical appearance, properties, attributes."
                    f"Caption:\n{caption.strip()}\n\n"
                    "Object Description:"
                )
            result = self.summarizer_pipe(summary_prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
            object_description = result.split("Object Description:")[-1].strip()

            data[obj_id]['object_level_caption'] = caption
            data[obj_id]['object_level_caption_summarized'] = object_description
    
    def image_processing_surrounding_captioning(self, data = None, prompt = None):
            for obj_id in tqdm(data, desc='Generating Object Surrounding Captions:'):
                # Step 1: Get individual frames (images) and mask images of each object
                frame_ids = [frame for (frame, _, _) in data[obj_id]['repre_mask_list']]
                mask_ids = [mask for (_, mask, _) in data[obj_id]['repre_mask_list']]

                image_path = 'data/demo/scene0608_00/color_640'
                images = [Image.open(f'{image_path}/{id}.jpg') for id in frame_ids]

                mask_path = 'data/demo/scene0608_00/output/mask'
                masks = [
                    Image.open(os.path.join(mask_path, f'{img}.png')).convert('L')
                    for img in frame_ids
                ]

                # Step 2: Identify the Objects from the mask images based on the mask_id
                masked_images_padded = []
                for img, id, mask in zip(images, mask_ids, masks):
                    binary_mask = (np.array(mask) == id).astype(np.uint8) * 255
                    x_min, y_min, x_max, y_max = img_process.mask_to_bbox(binary_mask)
                    bbox = (x_min, y_min, x_max, y_max)
                    image = img_process.draw_bbox_on_image(image=img, bbox=bbox)
                    # Crop the Images to obtain the object only
                    cropped_img = img_process.crop_with_padding(image, bbox, padding_ratio=1.5)
                    if len(masked_images_padded) < 5:
                        masked_images_padded.append(cropped_img)
                
                # img_process.img_visualization(masked_images=masked_images_padded)
                
                images_input = masked_images_padded
                max_partition = len(mask_ids)
                if prompt is None:
                    prompt = '''
                        You are provided with multiple views of a specific object which is enclosed in a red bounding box. These views also include its immediate surroundings and neighboring objects.

                        Your task is to write a structural description of the surrounding scene context of the object.

                        Use the following output format:
                        - The first sentence should describe **the main object briefly**.
                        - Each subsequent sentence should describe a **spatial or functional relationship** between this object and other nearby objects.

                        You may mention background elements, relative positions (e.g., "next to", "above", "in front of"), and possible usage relationships.

                        For example, the output should look like this:
                            This object is a white door with a classic six-panel design.  
                            It is positioned next to a dark-colored couch.  
                            It is located beneath a wall-mounted world map.  
                            It appears to serve as a main entryway within a domestic space.
                        Each sentence must describe exactly one relation.
                        Do not include any labels like "Image", "Caption", or notes. Only output the structured paragraph.
                        Now generate the caption:
                        '''

                query = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images_input))]) + '\n' + prompt

                caption = self.object_level_caption(query=query, images=images_input, max_partition=max_partition)
                data[obj_id]['object_surrounding_caption'] = caption

    def single_object_image_processing_object_captioning(self, data = None, obj_id = 0, prompt = None, summary_prompt = None):
        # Step 1: Get individual frames (images) and mask images of each object
            frame_ids = [frame for (frame, _, _) in data[obj_id]['repre_mask_list']]
            mask_ids = [mask for (_, mask, _) in data[obj_id]['repre_mask_list']]

            image_path = 'data/demo/scene0608_00/color_640'
            images = [Image.open(f'{image_path}/{id}.jpg') for id in frame_ids]

            mask_path = 'data/demo/scene0608_00/output/mask'
            masks = [
                Image.open(os.path.join(mask_path, f'{img}.png')).convert('L')
                for img in frame_ids
            ]

            # Step 2: Identify the Objects from the mask images based on the mask_id
            masked_images_padded = []
            for img, id, mask in zip(images, mask_ids, masks):
                binary_mask = (np.array(mask) == id).astype(np.uint8) * 255
                x_min, y_min, x_max, y_max = img_process.mask_to_bbox(binary_mask)
                bbox = (x_min, y_min, x_max, y_max)
                image = img.copy()
                # Crop the Images to obtain the object only
                cropped_img = img_process.crop_with_padding(image, bbox, padding_ratio=0.1)
                if len(masked_images_padded) < 5:
                    masked_images_padded.append(cropped_img)
            
            images_input = masked_images_padded
            max_partition = len(mask_ids)
            if prompt is None:
                prompt = '''
                You are provided with multiple views of a single object and the views are cropped to only showcase the object.

                Your task is to write **one short paragraph** describing only the main object.

                Focus on the object's physical appearance, attributes or properties. Do not mention background, surrounding objects, or the object's spatial position and the red bounding boxes.

                Now generate the object caption for the following object:
                '''
            query = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images_input))]) + '\n' + prompt

            caption = self.object_level_caption(query=query, images=images_input, max_partition=max_partition)

            if summary_prompt is None:
                summary_prompt = (
                    "You are an assistant tasked with extracting concise object descriptions from detailed scene captions.\n"
                    "Your goal is to output sentences describing only the **object itself**, including its: physical appearance, properties, attributes."
                    f"Caption:\n{caption.strip()}\n\n"
                    "Object Description:"
                )
            result = self.summarizer_pipe(summary_prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
            object_description = result.split("Object Description:")[-1].strip()
            return caption, object_description
    
    def single_image_processing_surrounding_captioning(self, data = None, prompt = None, obj_id = None):
            # Step 1: Get individual frames (images) and mask images of each object
            frame_ids = [frame for (frame, _, _) in data[obj_id]['repre_mask_list']]
            mask_ids = [mask for (_, mask, _) in data[obj_id]['repre_mask_list']]

            image_path = 'data/demo/scene0608_00/color_640'
            images = [Image.open(f'{image_path}/{id}.jpg') for id in frame_ids]

            mask_path = 'data/demo/scene0608_00/output/mask'
            masks = [
                Image.open(os.path.join(mask_path, f'{img}.png')).convert('L')
                for img in frame_ids
            ]

            # Step 2: Identify the Objects from the mask images based on the mask_id
            masked_images_padded = []
            for img, id, mask in zip(images, mask_ids, masks):
                binary_mask = (np.array(mask) == id).astype(np.uint8) * 255
                x_min, y_min, x_max, y_max = img_process.mask_to_bbox(binary_mask)
                bbox = (x_min, y_min, x_max, y_max)
                image = img_process.draw_bbox_on_image(image=img, bbox=bbox)
                # Crop the Images to obtain the object only
                cropped_img = img_process.crop_with_padding(image, bbox, padding_ratio=1.5)
                if len(masked_images_padded) < 5:
                    masked_images_padded.append(cropped_img)
            
            img_process.img_visualization(masked_images=masked_images_padded)
            
            images_input = masked_images_padded
            max_partition = len(mask_ids)
            if prompt is None:
                prompt = '''
                    You are provided with multiple views of a specific object which is enclosed in a red bounding box. These views also include its immediate surroundings and neighboring objects.

                    Your task is to write a structural description of the surrounding scene context of the object.

                    Use the following output format:
                    - The first sentence should describe **the main object briefly**.
                    - Each subsequent sentence should describe a **spatial or functional relationship** between this object and other nearby objects.

                    You may mention background elements, relative positions (e.g., "next to", "above", "in front of"), and possible usage relationships.

                    For example, the output should look like this:
                        This object is a white door with a classic six-panel design.  
                        It is positioned next to a dark-colored couch.  
                        It is located beneath a wall-mounted world map.  
                        It appears to serve as a main entryway within a domestic space.
                    Each sentence must describe exactly one relation.
                    Do not include any labels like "Image", "Caption", or notes. Only output the structured paragraph.
                    Now generate the caption:
                    '''

            query = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images_input))]) + '\n' + prompt

            caption = self.object_level_caption(query=query, images=images_input, max_partition=max_partition)

            return caption

          
                