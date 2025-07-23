# 3D-Scene Captioning

We present an **unsupervised framework** for holistic **indoor scene understanding**, which generates accurate, detailed, and structured **scene-level captions** from a 3D scene scan and registered multi-view RGB images.

---

## Overview

Our pipeline consists of the following stages:

1. **Object Segmentation**  
   Each scene is segmented into individual objects using either:
   - Ground truth instance labels from the **ScanNet++** dataset, or
   - The open-vocabulary 3D instance segmentation method **MaskClustering**.

2. **Object-Level Captioning**  
   For each object, a vision-language model generates:
   - An **object category**
   - A **fine-grained caption** describing its appearance and properties
   - A **structured caption** describing its **spatial and functional relations** with nearby objects

3. **Region-Level Captioning**  
   Objects are grouped into **functional regions** based on both:
   - **Spatial proximity** (3D centroid distance)
   - **Semantic similarity** (caption embeddings)  
   Captions for each region are then generated using these clusters.

4. **Scene-Level Captioning**  
   A large language model (LLM) reasons over region-level summaries to produce a **global scene caption** that reflects the functional composition and semantic diversity of the environment.

---

## Results

Our method is evaluated on the **ScanNet++** dataset. It generates **more informative**, **well-organized**, and **annotation-free** captions, demonstrating strong potential for scalable and human-centric scene understanding.

---

## Repository Structure

### 1. `data_download/`  
Experiments using **ground truth object segmentation** from ScanNet++ to extract individual objects and generate captions.

### 2. `MaskClustering-main/`  
Experiments using **MaskClustering**, an open-vocabulary 3D instance segmentation method, to extract object instances for downstream captioning.

All experiments are conducted in the notebook `0. Preprocess_and_MaskClustering.ipynb`, which includes the following steps:

- Preprocessing
- Mask Clustering
- Visualization
- Result Verification

Please follow the steps in the notebook in the given order to ensure correct execution of each module.

---

## Core Modules

### ### `0. Preprocess_and_MaskClustering.ipynb`

Provides a full pipeline from downloading the ScanNet++ data to result verification.

**Key functionalities:**
- `Preprocess`: Download the dataset, extract frames from video, render the extracted frames
- `2D Segmentation`: Generate 2D masks for every ten frames 
- `MaskClustering`: Call the MaskClustering methods
- `Visualization`: Visualize the clustering result
- `Result Verfication`: Sample a object to verify its detail and best views

---

### ### `Image_Processing.py`

Provides a full pipeline for visualizing and processing segmented object instances from 2D images.

**Key functionalities:**
- `mask_to_bbox`: Extract bounding boxes from binary masks  
- `draw_bbox_on_image`: Visualize bounding boxes  
- `crop_with_padding`: Crop images with contextual padding  
- `top5_view_selection`: Select top-N views based on projected area  
- `img_visualization`, `cropped_image_visualization`: Side-by-side comparison of raw images, masks, and crops

---

### ### `internVL.py`

Implements a pipeline using **InternVL-3**, a vision-language model, for object captioning and contextual understanding.

**Supports:**
- **Multi-view object captioning**  
- **Category label prediction**  
- **Scene context description**

The `InternVL_VLM` class handles:
- Preprocessing (resizing, normalization, aspect-ratio tiling)
- Prompt construction and inference
- Multi-GPU inference with HuggingFace's InternVL-3

---

### ### `Caption_Generator.py` *(deprecated; use `internVL.py` instead)*

Defines a `Caption_Generator` class using:
- **Ovis2-1B** multimodal LLM
- **Qwen1.5** for summarization

Supports both batch and per-object inference:
- Object view cropping and padding
- Object captioning with customizable prompts
- Summarization of long captions into concise object descriptions
- Structured descriptions of object surroundings

---

## Summary

Our compositional captioning framework transitions from **low-level 3D geometry** to **high-level language** by leveraging pretrained VLMs and LLMs. This enables rich object-centric understanding and coherent scene narratives—**without requiring labeled data**.

---
