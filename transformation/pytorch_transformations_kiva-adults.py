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

base_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True)
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
    color_channels = {
        "Red": torch.tensor([1.0, 0.0, 0.0]),  
        "Green": torch.tensor([0.0, 1.0, 0.0]),
        "Blue": torch.tensor([0.0, 0.0, 1.0]), 
        "Yellow": torch.tensor([1.0, 1.0, 0.2]), # Red + Green
        "Grey": None  
    }

    if target_color not in color_channels:
        raise ValueError("Invalid color. Choose from 'Red', 'Yellow', 'Green', 'Blue', or 'Grey'.")

    available_colors = [color for color in color_channels if color != target_color and color != train_color]

    initial_color, incorrect_color = random.sample(available_colors, 2) # Sample without overlap

    def apply_color_overlay(img, color):
        # Separate RGB and alpha channels (helps with transparency handling)
        rgb_channels = img[:3, :, :].clone().float() / 255.0 
        alpha_channel = img[3, :, :].clone() if img.shape[0] == 4 else None 

        if color == "Grey": # Simply make greyscale
            grayscale = rgb_channels.mean(dim=0, keepdim=True)
            colored_image = grayscale.expand_as(rgb_channels) 
        else: # Get color tensor and apply overlay to RGB channels
            color_tensor = color_channels[color].view(3, 1, 1)
            colored_image = rgb_channels * color_tensor

        colored_image = (colored_image * 255).byte()

        if alpha_channel is not None: # Add the alpha channel back if it exists
            colored_image = torch.cat((colored_image, alpha_channel.unsqueeze(0)), dim=0)

        return colored_image

    initial_image = apply_color_overlay(image, initial_color)
    correct_image = apply_color_overlay(image, target_color)
    incorrect_image = apply_color_overlay(image, incorrect_color)

    if (type == "train"):
        return initial_image, correct_image, initial_color, target_color
    elif (type == "test"):
        return initial_image, correct_image, incorrect_image, initial_color, incorrect_color


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
            return starting_count / param_num, starting_count - 1

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
        return initial_image, correct_image, incorrect_image, starting_count, incorrect_count

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
        return correct_image, image, parameter, parameter
    elif (type == "test"):
        return correct_image, incorrect_image, 0, incorrect_option

def apply_resizing(image, factor, type):
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
        return correct_image, incorrect_image, 0, incorrect_option

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

    original_image = F.rotate(image, parse_angle(initial_rotation))
    correct_image = F.rotate(original_image, parse_angle(angle))
    incorrect_image = F.rotate(original_image, parse_angle(incorrect_angle))

    if (type == "train"):
        return original_image, correct_image, initial_rotation, angle
    elif (type == "test"):
        return original_image, correct_image, incorrect_image, initial_rotation, incorrect_angle

def save_values_to_txt(initial_train, output_train, initial_test, incorrect_test):
    """
    Args:
        initial_train (int): Initial value for train input.
        output_train (int): Output value for train output.
        initial_test (int): Initial value for test input.
        incorrect_test (int): Incorrect value for test output.
    """
    with open(f"{args.output_directory}/output_{args.transformation}{args.parameter}.txt", "a") as file:
        file.write(f"Train_input: {initial_train}\n")
        file.write(f"Train_output: {output_train}\n")
        file.write(f"Test_input: {initial_test}\n")
        file.write(f"MC: {incorrect_test}\n")

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

        transforms.ToPILImage()(original_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_train_input.png"))
        transforms.ToPILImage()(correct_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_train_output.png"))

        # Process one item from test_loader
        original_image = next(test_iter)[0]

        if transformation == "Colour":
            original_image, correct_image, incorrect_image, test_input, incorrect_option = apply_color(original_image, args.parameter, type="test", train_color=train_input)
        elif transformation == "Counting":
            original_image, correct_image, incorrect_image, test_input, incorrect_option = apply_counting(original_image, args.parameter, type="test", initial_count=train_input)
        elif transformation == "Reflect":
            correct_image, incorrect_image, test_input, incorrect_option = apply_reflection(original_image, args.parameter, type="test")
        elif transformation == "Resize":
            correct_image, incorrect_image, test_input, incorrect_option = apply_resizing(original_image, args.parameter, type="test")
        elif transformation == "2DRotation":
            original_image, correct_image, incorrect_image, test_input, incorrect_option = apply_rotation(original_image, args.parameter, type="test", train_angle=train_input)

        transforms.ToPILImage()(original_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_test_input.png"))
        transforms.ToPILImage()(correct_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_test_correct_output.png"))
        transforms.ToPILImage()(incorrect_image).save(os.path.join(args.output_directory, f"{transformation}{args.parameter}_{i}_test_incorrect_output.png"))

        save_values_to_txt(train_input, train_output, test_input, incorrect_option)

        i += 1  

    except StopIteration:
        print(f"Finished generating transformed stimuli for {transformation}{args.parameter}")
        break
