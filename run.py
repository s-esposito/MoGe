import cv2
import os
import torch
from moge.model.v1 import MoGeModel
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoGe')
    parser.add_argument('--input_video', type=str, default='./assets/example_data/davis/car-turn')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             

# list files in the folder (.png or .jpg)
folder_path = args.input_video
files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(('.png', '.jpg', '.jpeg'))]
# sort files by name
files.sort()
if not files:
    raise ValueError(f"No image files found in the folder: {folder_path}")

# iterate over all files
outputs = []
for i, file_name in enumerate(files):
    print(f"Processing file {i+1}/{len(files)}: {file_name}")

    file_path = os.path.join(folder_path, file_name)
    
    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    input_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)                       
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

    # Infer 
    output = model.infer(input_image)
    # `output` has keys "points", "depth", "mask" and "intrinsics",
    # The maps are in the same size as the input image. 
    # {
    #     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
    #     "depth": (H, W),        # scale-invariant depth map
    #     "mask": (H, W),         # a binary mask for valid pixels. 
    #     "intrinsics": (3, 3),   # normalized camera intrinsics
    # }
    # For more usage details, see the `MoGeModel.infer` docstring.
    outputs.append(output)

video_name = os.path.basename(args.input_video)
# save output as npz
output_path = os.path.join(args.output_dir, f'{video_name}_moge.npz')
os.makedirs(args.output_dir, exist_ok=True)
torch.save(outputs, output_path)