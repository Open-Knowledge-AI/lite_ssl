import torch

from safetensors.torch import save_file

ckpt = torch.load("last.ckpt")

# prefix = "teacher_model."
prefix = "model."

teacher_sd = {k.replace(prefix, "backbone."): v for k, v in ckpt.items() if k.startswith(prefix)}

torch.save(
    teacher_sd,
    "pytorch_model.bin",
)
save_file(teacher_sd, "model.safetensors")
