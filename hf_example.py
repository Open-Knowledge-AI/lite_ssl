from PIL import Image

from transformers import AutoModel, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("OK-AI/dino-vits16-pretrain-in1k")
pil_input = Image.new("RGB", (1024, 1024))
preprocessed = processor(pil_input, return_tensors="pt")

model_input = preprocessed.data["pixel_values"]
print(f"Preprocessed Input Shape: {model_input.shape}")

model = AutoModel.from_pretrained(
    "OK-AI/dino-vits16-pretrain-in1k",
    # revision="ep100-teacher",
    trust_remote_code=True,
)
outputs = model(model_input)
print(f"Output Keys: {outputs.keys()}")
print(f"{outputs["latent"].shape=}, {outputs["patch_latent"].shape=}")
