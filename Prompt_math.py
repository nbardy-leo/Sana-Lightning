

# Initialize the pipeline
pipeline = SanaPipeline(config="path_to_config.yaml")
pipeline.from_pretrained("path_to_model_checkpoint.pth")

# Define your prompts
prompt1 = "A beautiful landscape with mountains and a river."
prompt2 = "A futuristic cityscape at night."

# Embed the prompts
embedding1, mask1 = pipeline.embed(prompt1)
embedding2, mask2 = pipeline.embed(prompt2)

# Interpolate between embeddings
alpha = 0.5  # You can vary alpha between 0 and 1 for different interpolations
interpolated_embedding = embedding1 * (1 - alpha) + embedding2 * alpha
interpolated_mask = ((mask1 + mask2) > 0).long()  # Combine masks appropriately

# Generate images
image1 = pipeline.render_prompt(prompt1)
image_interpolated = pipeline.render_embedding(interpolated_embedding, emb_masks=interpolated_mask)
image2 = pipeline.render_prompt(prompt2)

# Save or display images as needed
# For example, to display image_interpolated using matplotlib:
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Convert tensor to image
to_pil = T.ToPILImage()
image = to_pil(image_interpolated.squeeze(0).cpu())

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()
