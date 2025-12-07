import torch
#################################### For Image ####################################
from PIL import Image
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model


model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("images\image.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="Segment the mouse in the image")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print("masks", type(masks))


merged = masks.sum(dim=0) > 0           # merge all masks → boolean
merged_np = merged.cpu().numpy()        # to numpy
merged_np = np.squeeze(merged_np)       # remove (1,1,...) dims

# ensure shape is (H, W)
if merged_np.ndim != 2:
    raise ValueError(f"Unexpected merged mask shape: {merged_np.shape}")

# convert to image
mask_img = (merged_np.astype(np.uint8) * 255)
Image.fromarray(mask_img).save("merged_mask_face.png")

print("saved merged_mask.png")