import argparse

def convert_model(args):
    import torch
    from torch import nn

    import yaml

    import os
    from saicinpainting.training.trainers import load_checkpoint
    from omegaconf import OmegaConf

    class JITWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image, mask):
            batch = {
                "image": image,
                "mask": mask
            }
            out = self.model(batch)
            return out["inpainted"]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lama_model_path = args.model_path

    train_config_path = os.path.join(lama_model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(lama_model_path,'models',args.epoch)
                                    
    print(f"loading checkpoint { checkpoint_path}")
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
    model.eval()
    model.to(device)
    jit_model_wrapper = JITWrapper(model).to(device)
    
    image = torch.rand(1, 3, args.input_height, args.input_width).to(device)
    mask = torch.rand(1, 1, args.input_height, args.input_width).to(device)
    output = jit_model_wrapper(image, mask)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    image = image.to(device)
    mask = mask.to(device)
    traced_model = torch.jit.trace(jit_model_wrapper, (image, mask), strict=False).to(device)

    print(f"Saving big-lama.pt model to {args.pt_name}")
    traced_model.save(args.pt_name)

    print(f"Checking jit model output...")
    jit_model = torch.jit.load(str(args.pt_name))
    jit_output = jit_model(image, mask)
    diff = (output - jit_output).abs().sum()
    print(f"diff: {diff}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="directory with config.yaml and root to models/.ckpt files")
    parser.add_argument('--epoch', type=str, default="last.ckpt", help="name of .ckpt to load (e.g. last.ckpt)")
    parser.add_argument('--pt_name', type=str, required=True, help="pt file name to save")
    parser.add_argument('--input_width', type=int, required=True)
    parser.add_argument('--input_height', type=int, required=True)

    args = parser.parse_args()
    convert_model(args)
