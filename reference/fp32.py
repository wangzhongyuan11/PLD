"""Frozen A checkpoint inference reconstructed from its explicit TinyUNet graph."""
import torch
import torch.nn.functional as F


class FloatModel:
    def __init__(self, checkpoint):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.state = data["model_state"]
        self.names = [f"encoders.{i}.{j}" for i in range(4) for j in (0, 2)]
        self.names += [f"decoder_blocks.{i}.{j}" for i in range(3) for j in (0, 2)] + ["head"]
        if set(self.state) != {n+s for n in self.names for s in (".weight", ".bias")}:
            raise ValueError("unexpected checkpoint graph")

    @torch.inference_mode()
    def run(self, image):
        x = torch.from_numpy(image)
        outputs = []
        for i, name in enumerate(self.names):
            if i:
                x = F.relu(outputs[-1])
                if i in (2, 4, 6):
                    x = F.max_pool2d(x, 2)
                if i in (8, 10, 12):
                    skip = F.relu(outputs[{8: 5, 10: 3, 12: 1}[i]])
                    x = torch.cat((skip, F.interpolate(x, scale_factor=2, mode="nearest")), dim=1)
            x = F.conv2d(x, self.state[name+".weight"], self.state[name+".bias"], padding=0 if i == 14 else 1)
            outputs.append(x)
        return outputs[-1].numpy()
