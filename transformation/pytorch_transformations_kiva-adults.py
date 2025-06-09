import os
import argparse
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as F
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images with specific transformations.")

    parser.add_argument('--input_directory', type=str,
                        help='Path to the input directory containing the images.')
    parser.add_argument('--output_directory', type=str,
                        help='Path to the output directory where processed images will be saved.')
    parser.add_argument('--next_index', type=int, default=0,
                        help='The starting index for naming the output files.')
    parser.add_argument('--trials', type=int, default=1,
                        help='The number of trials to run.')
    parser.add_argument('--transformation', type=str,
                        choices=["Colour", "Counting", "Reflect", "Resize", "2DRotation"],
                        help='The transformation to apply to the images.')
    parser.add_argument('--parameter', type=str,
                        help='The parameter for the transformation.')
    parser.add_argument('--shuffle', type=bool, default=False,
                        help='To shuffle the objects dataset.')
    return parser.parse_args()

# ------------------------------------------------------------------------------------------------

args = parse_arguments()

args.output_directory = os.path.join(args.output_directory, f"{args.transformation}{args.parameter}")

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

class Objects(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [file for file in os.listdir(img_dir) if file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

if args.transformation == "Resize":
    base_transform = transforms.Compose([
        transforms.Resize((300, 300), antialias=True)
    ])
else:
    base_transform = transforms.Compose([
        transforms.Resize((600, 600), antialias=True),
    ])

# Initialize dataset
dataset = Objects(img_dir=args.input_directory, transform=base_transform)

# Define trial splits for train and test
total_images = len(dataset)
trials = args.trials if args.trials else total_images // 2
if trials > total_images // 2 or trials <= 0:
    raise ValueError(f"The maximum number of trials possible is {total_images // 2}.")

# Shuffle Objects if decided upon
indices = list(range(total_images))
if args.shuffle:
    random.shuffle(indices)

# Define train and test sets using shuffled indices
train_indices = indices[:trials]
test_indices = indices[trials:2 * trials]

train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# -----------------------------------------------------
# Transformation functions

def apply_color(image, target_color, type, train_color=None):
    color_map = {
        "Red": torch.tensor([255, 0, 0], dtype=torch.float32),     
        "Green": torch.tensor([0, 128, 0], dtype=torch.float32),   
        "Blue": torch.tensor([0, 0, 255], dtype=torch.float32),    
        "Yellow": torch.tensor([255, 255, 50], dtype=torch.float32), # (R + G)
        "Grey": None  # Special case for greyscale
    }

    if target_color not in color_map:
        raise ValueError("Invalid color. Choose from 'Red', 'Yellow', 'Green', 'Blue', or 'Grey'.")

    available_colors = [color for color in color_map if color != target_color and color != train_color]

    initial_color, incorrect_color = random.sample(available_colors, 2) # Sample without overlap

    def color_overlap(img, color):
        has_alpha = img.shape[0] == 4
        alpha_channel = None

        if has_alpha: # Separate alpha channel if present
            alpha_channel = img[3, :, :].clone()
            img = img[:3, :, :]  

        img = img.float() / 255.0 # Normalize to [0, 1]
        height, width = img.shape[1], img.shape[2]

        if color == "Grey":  # Convert to grayscale
            grayscale = img.mean(dim=0, keepdim=True) # Average over RGB channels
            blended_img = grayscale.expand_as(img)  
        else:
            target_color = color_map[color].view(3, 1, 1).repeat(1, height, width) / 255.0
            blended_img = (img + target_color) / 2  # Blend the original image with the target color
            # To create a more distinctive change, use the option below; increasing the color distinction will decrease differentiation in object details
            # blended_img = img * (1 - 0.7) + target_color * 0.7 # Blend the original image (0.3) with target color (0.7)

        blended_img = torch.clamp(blended_img, 0, 1)
        blended_img = (blended_img * 255).byte()  

        if has_alpha: # Reattach alpha channel if it exists
            blended_img = torch.cat((blended_img, alpha_channel.unsqueeze(0)), dim=0)

        return blended_img

    initial_image = color_overlap(image, initial_color)
    correct_image = color_overlap(image, target_color)
    incorrect_image = color_overlap(image, incorrect_color)

    if (type == "train"):
        return initial_image, correct_image, initial_color, target_color
    elif (type == "test"):
        return initial_image, correct_image, incorrect_image, initial_color, target_color, incorrect_color

def generate_grid_image(image, count):
    _, img_height, img_width = image.shape
    
    # Resize the base object image to fit in a grid
    max_items_per_row = int((10 ** 0.5) + 1)
    item_size = min(img_width, img_height) // max_items_per_row
    shrunken_image = F.resize(image, (item_size, item_size), antialias=True)  
    
    if shrunken_image.shape[0] == 3:  # If no alpha channel, add alpha channel with 255 (opaque) for objects
        alpha_channel = torch.ones((1, item_size, item_size), dtype=image.dtype) * 255
        shrunken_image = torch.cat((shrunken_image, alpha_channel), dim=0)
    
    canvas = torch.zeros((4, img_height, img_width), dtype=image.dtype)  # 4 channels (RGBA), 0 alpha for transparency

    # Arrange objects on canvas in grid layout
    for i in range(count):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        canvas[:, y:y + item_size, x:x + item_size] = shrunken_image  # Paste shrunken image onto canvas

    return canvas
    
def apply_counting(image, param, type, initial_count=None):
    if param not in {'+1', '+2', '-1', '-2', 'x2', 'x3', 'd2', 'd3'}:
        raise ValueError("Invalid counting operation. Choose from '+1', '+2', '-1', '-2', 'x2', 'x3', 'd2', or 'd3'.")
    
    starting_options_map = {
        '+': [2, 3, 4, 5],
        '-': [7, 6, 5, 4, 3],
        'x': {2: [2, 3, 4], 3: [1, 2, 3]},
        'd': {2: [8, 6, 4], 3: [9, 6, 3]}
    }

    def calculate_counts(operation, starting_count, param_num):
        if operation == '+':
            return starting_count + param_num, starting_count - 1
        elif operation == '-':
            return starting_count - param_num, starting_count + 1
        elif operation == 'x':
            return starting_count * param_num, starting_count + 1
        elif operation == 'd':
            return starting_count // param_num, starting_count - 1

    operation = param[0]  # +, -, x, d
    param_num = int(param[1:])  # 1, 2, or 3

    starting_options = starting_options_map[operation] if operation in ['+', '-'] else starting_options_map[operation][param_num]
    starting_count = random.choice([num for num in starting_options if num != initial_count] or starting_options)
    correct_count, incorrect_count = calculate_counts(operation, starting_count, param_num)

    initial_image = generate_grid_image(image, starting_count)
    correct_image = generate_grid_image(image, correct_count)
    incorrect_image = generate_grid_image(image, incorrect_count)

    if (type == "train"):
        return initial_image, correct_image, starting_count, correct_count
    elif (type == "test"):
        return initial_image, correct_image, incorrect_image, starting_count, correct_count, incorrect_count

def apply_reflection(image, parameter, type):
    if parameter == "X":
        correct_image = F.vflip(image)
        incorrect_image = F.hflip(image)  # Flip along the Y-axis for incorrect
        incorrect_option = "Y"
    elif parameter == "Y":
        correct_image = F.hflip(image)
        incorrect_image = F.vflip(image)  # Flip along the X-axis for incorrect
        incorrect_option = "X"
    elif parameter == 'XY':
        correct_image = F.hflip(F.vflip(image))  # Reflect across both X- and Y-axes
        incorrect_option = random.choice(["X", "Y"])
        incorrect_image = F.hflip(image) if incorrect_option == "X" else F.vflip(image)
    else:
        raise ValueError("Invalid reflect factor. Choose from 'X', 'Y', or 'XY'.")
    
    if (type == "train"):
        return correct_image, image, 0, parameter
    elif (type == "test"):
        return correct_image, incorrect_image, 0, parameter, incorrect_option

def paste_on_600(img: torch.Tensor, canvas_size: int = 600) -> torch.Tensor:
    _, h, w = img.shape

    # Down-scale very large inputs so the larger edge is 600
    if max(h, w) > canvas_size:
        scale = canvas_size / float(max(h, w))
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = F.resize(img, (new_h, new_w), antialias=True)
        _, h, w = img.shape

    pad_left  = (canvas_size - w) // 2
    pad_right = canvas_size - w - pad_left
    pad_top   = (canvas_size - h) // 2
    pad_bottom= canvas_size - h - pad_top

    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

def apply_resizing(image, factor, type):
    enlarge_first = factor.startswith("0.5")
    base_img = image
    if enlarge_first:
        H, W   = image.shape[1:]
        base_img = F.resize(image, (H * 2, W * 2), antialias=True)
    image = base_img
    
    resize_factors = {
        "0.5XY": ((0.5, 0.5), (2.0, 2.0), "2XY"),
        "2XY": ((2.0, 2.0), (0.5, 0.5), "0.5XY"),
        "0.5X": ((0.5, 1.0), (1.0, 0.5), "0.5Y"),
        "0.5Y": ((1.0, 0.5), (0.5, 1.0), "0.5X"),
        "2X": ((2.0, 1.0), (1.0, 2.0), "2Y"),
        "2Y": ((1.0, 2.0), (2.0, 1.0), "2X"),
    }

    try:
        correct_resize_factors, incorrect_resize_factors, incorrect_option = resize_factors[factor]
    except KeyError:
        raise ValueError("Invalid resize factor. Choose from '0.5XY', '2XY', '0.5X', '0.5Y', '2X', or '2Y'.")

    new_width, new_height = image.shape[2], image.shape[1]

    correct_new_width = int(new_width * correct_resize_factors[0])
    correct_new_height = int(new_height * correct_resize_factors[1])

    incorrect_new_width = int(new_width * incorrect_resize_factors[0])
    incorrect_new_height = int(new_height * incorrect_resize_factors[1])

    correct_image = transforms.Resize((correct_new_height, correct_new_width), antialias=True)(image)
    incorrect_image = transforms.Resize((incorrect_new_height, incorrect_new_width), antialias=True)(image)

    if (type == "train"):
        return correct_image, 0, args.parameter
    elif (type == "test"):
        return correct_image, incorrect_image, 0, args.parameter, incorrect_option

def apply_rotation(image, angle, type, train_angle=None):
    matches = {
                "+45": ["+135", "180"],
                "-45": ["-135", "180"],
                "+90": ["180"],
                "-90": ["180"],
                "+135": ["+45", "180"],
                "-135": ["-45", "180"],
                "180": ["+45", "-45", "+90", "-90"],
            }
    
    if angle in matches and matches[angle]:
        incorrect_angle = random.choice(matches[angle])
    else:
        raise ValueError("Invalid rotation angle. Choose from '+45', '-45', '+90', '-90', '+135', or '-135'.")
    
    initial_rotation = random.choice(
        [angle for angle in ["+45", "-45", "+90", "-90", "+135", "-135"] if angle != train_angle] 
        or ["+45", "-45", "+90", "-90", "+135", "-135"]
    )

    def parse_angle(angle):
        if angle[:1] == "+":
            return -int(angle[1:])
        elif angle[:1] == "-":
            return int(angle[1:])
        elif angle == "180":
            return 180
        
    def combine_angles(a1: str, a2: str) -> str: # Always returns a positive angle
        def s2i(s: str) -> int:              
            if s.startswith('+'):
                return int(s[1:])
            elif s.startswith('-'):
                return -int(s[1:])
            else:                 
                return int(s)

        total = (s2i(a1) + s2i(a2)) % 360    # Wrap into a single turn

        if total == 0:      
            return "0"
        if total == 180:
            return "180"
        return f"{total}"   
 
    final_start_angle = combine_angles(initial_rotation, "0")
    final_correct_angle = combine_angles(initial_rotation, angle)
    final_incorrect_angle = combine_angles(combine_angles(initial_rotation, angle), incorrect_angle)

    original_image = F.rotate(image, parse_angle(initial_rotation))
    correct_image = F.rotate(original_image, parse_angle(angle))
    incorrect_image = F.rotate(original_image, parse_angle(incorrect_angle))

    if (type == "train"):
        return original_image, correct_image, final_start_angle, final_correct_angle
    elif (type == "test"):
        return original_image, correct_image, incorrect_image, final_start_angle, final_correct_angle, final_incorrect_angle

def save_values_to_txt(index, initial_train, output_train, initial_test, correct_test, incorrect_test):
    """
    Args:
        initial_train (str): Initial value for train input.
        output_train (str): Output value for train output.
        initial_test (str): Initial value for test input.
        correct_test (str): Correct value for test output.
        incorrect_test (str): Incorrect value for test output.
    """
    with open(f"{args.output_directory}/output_{args.transformation}{args.parameter}.txt", "a") as file:
        file.write(f"Trial: {index}\n")
        file.write(f"Train_input: {initial_train}\n")
        file.write(f"Train_output: {output_train}\n")
        file.write(f"Test_input: {initial_test}\n")
        file.write(f"Test_correct_output: {correct_test}\n")
        file.write(f"Test_incorrect_output: {incorrect_test}\n")

train_iter = iter(train_loader)
test_iter = iter(test_loader)

i = 0  # General index for file naming

while True:
    try:
        # Process one item from train_loader
        original_image = next(train_iter)[0]
        transformation = args.transformation  

        if transformation == "Colour":
            original_image, correct_image, train_input, train_output = apply_color(original_image, args.parameter, type="train")
        elif transformation == "Counting":
            original_image, correct_image, train_input, train_output = apply_counting(original_image, args.parameter, type="train")
        elif transformation == "Reflect":
            original_image, correct_image, train_input, train_output = apply_reflection(original_image, args.parameter, type="train")
        elif transformation == "Resize":
            correct_image, train_input, train_output = apply_resizing(original_image, args.parameter, type="train")
        elif transformation == "2DRotation":
            original_image, correct_image, train_input, train_output = apply_rotation(original_image, args.parameter, type="train")

        if transformation == "Resize":
            original_image = paste_on_600(original_image, 600)
            correct_image = paste_on_600(correct_image, 600)
        transforms.ToPILImage()(original_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_train_0_input.png"))
        transforms.ToPILImage()(correct_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_train_0_output.png"))

        # Process one item from test_loader
        original_image = next(test_iter)[0]

        if transformation == "Colour":
            original_image, correct_image, incorrect_image, test_input, correct_option, incorrect_option = apply_color(original_image, args.parameter, type="test", train_color=train_input)
        elif transformation == "Counting":
            original_image, correct_image, incorrect_image, test_input, correct_option, incorrect_option = apply_counting(original_image, args.parameter, type="test", initial_count=train_input)
        elif transformation == "Reflect":
            correct_image, incorrect_image, test_input, correct_option, incorrect_option = apply_reflection(original_image, args.parameter, type="test")
        elif transformation == "Resize":
            correct_image, incorrect_image, test_input, correct_option, incorrect_option = apply_resizing(original_image, args.parameter, type="test")
        elif transformation == "2DRotation":
            original_image, correct_image, incorrect_image, test_input, correct_option, incorrect_option = apply_rotation(original_image, args.parameter, type="test", train_angle=train_input)

        if transformation == "Resize":
            original_image = paste_on_600(original_image, 600)
            correct_image = paste_on_600(correct_image, 600)
            incorrect_image = paste_on_600(incorrect_image, 600)
        transforms.ToPILImage()(original_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_test_0_input.png"))
        transforms.ToPILImage()(correct_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_test_mc_0_input.png"))
        transforms.ToPILImage()(incorrect_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_test_mc_1_input.png"))

        save_values_to_txt(i, train_input, train_output, test_input, correct_option, incorrect_option)

        i += 1  

    except StopIteration:
        print(f"✓ Generated {i} trial(s) for {transformation}{args.parameter}.")
        break
