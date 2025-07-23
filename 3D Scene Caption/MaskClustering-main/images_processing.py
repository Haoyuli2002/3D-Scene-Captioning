from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

def mask_to_bbox(mask):
    '''
    Compute the bounding box coordinates from a binary mask.

    Parameters:
        - mask (np.ndarray): A 2D binary mask (values 0 or 1) where the target object is masked with non-zero value.
    
    Returns:
        - tuple: (x_min, y_min, x_max, y_max)
    '''
    # Non-zero values in each row: At least one value that is non-zero.
    rows = np.any(mask, axis=1)
    # Non-zero values in each column
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None  # empty mask
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max, y_max

def draw_bbox_on_image(image, bbox, color='red', width=5):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    return img_copy

def crop_with_padding(image, bbox, padding_ratio=0.1):
    """
    Crop image to bbox with padding. Padding is proportional to bbox size.
    """
    x_min, y_min, x_max, y_max = bbox
    width, height = image.size

    # Compute padding in pixels
    pad_x = int((x_max - x_min) * padding_ratio)
    pad_y = int((y_max - y_min) * padding_ratio)

    # Apply padding and clip to image boundaries
    left = max(x_min - pad_x, 0)
    upper = max(y_min - pad_y, 0)
    right = min(x_max + pad_x, width)
    lower = min(y_max + pad_y, height)
    img_copy = image.copy()

    return img_copy.crop((left, upper, right, lower))

def img_visualization(images=None, masks=None, mask_ids=None, masked_images=None, number_of_images=5):
    if images and masks and mask_ids and len(images) > 0 and len(masks) > 0 and len(mask_ids) > 0:
        fig, axes = plt.subplots(3, number_of_images, figsize=(12, 6))
        split_index = len(images) + len(masks)
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                ax.imshow(images[idx])
                ax.axis('off')
                ax.set_title(f'Image {idx+1}')
            elif idx < split_index:
                ax.imshow(masks[idx - len(images)])
                ax.axis('off')
                ax.set_title(f'Mask {idx+1 - len(images)}')
            else:
                img = np.array(masks[idx - split_index])
                object_id = mask_ids[idx - split_index]
                binary_mask = (img == object_id).astype(np.uint8) * 255
                ax.imshow(binary_mask, cmap='gray')
                ax.axis('off')
                ax.set_title(f'Object_Mask {idx+1 - split_index}')
        plt.tight_layout()
        plt.show()

    elif images is None and masked_images and len(masked_images) > 0:
        num_images = min(number_of_images, len(masked_images))
        fig, axes = plt.subplots(1, num_images, figsize=(10, 5))
        for idx, ax in enumerate(axes):
            ax.imshow(masked_images[idx])
            ax.axis('off')
            ax.set_title(f'Masked Object {idx+1}')
        plt.tight_layout()
        plt.show()

    else:
        print('Not supported visualization.')


def cropped_image_visualization(image_path = 'ScanNet_Data/data/0a7cc12c0e/object_crops_no_padding', object_id = 8):
    current_obj = []
    for f in os.listdir(image_path):
        if f'object_{object_id}_' in f:
            current_obj.append(os.path.join(image_path, f))

    object_img = [Image.open(f) for f in current_obj]

    fig, axes = plt.subplots(5, 8, figsize=(20, 12))
    for idx, ax in enumerate(axes.flat):
        if idx < len(object_img):
            ax.imshow(object_img[idx])
            ax.axis('off')  # Hide axes
            ax.set_title(f'Image {idx+1}')
    plt.tight_layout()
    plt.show()

def top5_view_selection(image_path = 'ScanNet_Data/data/0a7cc12c0e/object_crops_no_padding', object_id = 8, visualization = True):
    object_images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f'object_{object_id}_' in f]
    img_with_area = []
    for f in object_images:
        img = Image.open(f)
        area = img.size[0] * img.size[1]  # width × height
        img_with_area.append((f, area, img))

    # Sort by area descending
    top5_data = sorted(img_with_area, key=lambda x: -x[1])[:5]
    top5_images = [item[2] for item in top5_data]

    if visualization:
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        for i, (img, ax) in enumerate(zip(top5_images, axes)):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Top {i+1}')
        plt.tight_layout()
        plt.show()
    return top5_images