import click 

import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path 

def resize_image(image_path, max_dimension, out_dir):
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to RGB mode
    rgb_img = img.convert("RGB")
    
    # Resize the image while preserving aspect ratio
    width, height = rgb_img.size
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_width = int(width * (max_dimension / height))
        new_height = max_dimension
    
    resized_img = rgb_img.resize((new_width, new_height))
    
    # Save the resized image with the same name in the same directory
    directory, filename = os.path.split(image_path)
    filename, ext = os.path.splitext(filename)
    new_filename = os.path.join(out_dir, f"{filename}{ext}")
    resized_img.save(new_filename)
    # print(f"Resized image saved as: {new_filename}")

@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=False),
    help="Output directory to generate images to"
)
@click.option(
    "--in_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    help="Data directory containing the images."
)
def main(out: str, in_dir: str):
    for path in tqdm(sorted(list(Path(in_dir).rglob("*.png")))):
        resize_image(path, 430, f"{out}/faces_resized")
        