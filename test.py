from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained(
    "OK-AI/dino-vits16-pretrain-in1k",
    # revision="ep100-teacher",
    trust_remote_code=True,
)

processor = AutoImageProcessor.from_pretrained("OK-AI/dino-vits16-pretrain-in1k")

print(model)

print(processor)
