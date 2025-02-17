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
    parser.add_argument('--transformation_list', type=str,
                        help='Comma-separated list of transformation:parameter pairs (e.g., "Colour:Red,2DRotation:+90")')
    parser.add_argument('--shuffle', type=bool, default=False,
                        help='To shuffle the objects dataset.')
    return parser.parse_args()

# ------------------------------------------------------------------------------------------------

def parse_transformation_list(transformation_list):
    pairs = [pair.strip() for pair in transformation_list.split(',')]
    return [tuple(pair.split(':')) for pair in pairs]

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
    
args = parse_arguments()

args.output_directory = args.output_directory + f"/{args.transformation_list}"

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

base_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True)
])

dataset = Objects(img_dir=args.input_directory, transform=base_transform)

total_images = len(dataset)
trials = args.trials if args.trials else total_images // 2
if trials > total_images // 2 or trials <= 0:
    raise ValueError(f"The maximum number of trials possible is {total_images // 2}.")

indices = list(range(total_images))
if args.shuffle:
    random.shuffle(indices)

train_indices = indices[:trials]
test_indices = indices[trials:2 * trials]

train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

transformations = parse_transformation_list(args.transformation_list)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

transform_id = '_'.join([f"{t}{p}" for t, p in transformations])

if len(transformations) >= 2:
    input_transformation, test_incorrect_transformation = random.sample(transformations, 2)
else:
    input_transformation = test_incorrect_transformation = transformations[0]

# List of incompatible transformations
incompatible_transformations = [[("Reflect", "X"), ("2DRotation", "180")]]

for incompatible_set in incompatible_transformations:
    count = sum(1 for t, p in transformations if (t, p) in incompatible_set)
    if count > 1:
        raise ValueError(f"The selected transformations may cancel each other out: {', '.join([f'{t}:{p}' for t, p in incompatible_set])}")

# Check for duplicates
transformation_counts = {}
for transformation, parameter in transformations:
    if transformation not in transformation_counts:
        transformation_counts[transformation] = set()
    transformation_counts[transformation].add(parameter)

duplicates = {key: values for key, values in transformation_counts.items() if len(values) > 1}

if duplicates:
    raise ValueError(f"Multiple parameter values are identified with more than one transformation type(s) {duplicates}")

num_transformations = len(transformations)
base_count = trials // num_transformations # Each transformation should be last at least this many times
remainder = trials % num_transformations  

# Build a list with equal counts for each transformation (list of "last" transformations for each trial)
last_trans_list = []
for t in transformations:
    last_trans_list.extend([t] * base_count)

if remainder: # For any extra trials, randomly assign one extra occurrence to a concept
    last_trans_list.extend(random.sample(transformations, remainder))

# Shuffle list to randomize the order extra-occurrence concepts appear
random.shuffle(last_trans_list)

#---------------------------------------------------------------
# Transformation functions

def apply_color(image, target_color, type, initial_color=None):
    color_map = {
            "Red": torch.tensor([255, 0, 0], dtype=torch.float32),  # Red RGB
            "Green": torch.tensor([0, 200, 0], dtype=torch.float32),  # Green RGB # 128
            "Blue": torch.tensor([0, 0, 255], dtype=torch.float32),  # Blue RGB
        }
    colors = ["Red", "Green", "Blue"]
    if target_color in colors:
        colors.remove(target_color) 
    else:
        raise ValueError("Invalid color. Choose from 'Red', 'Green', or 'Blue'.")

    initial_color = random.choice(colors) if initial_color is None else initial_color
    incorrect_color = colors[0] if initial_color == colors[1] else colors[1]

    def color_overlap(img, color):
        has_alpha = img.shape[0] == 4
        alpha_channel = None

        if has_alpha: # Separate alpha channel if present
            alpha_channel = img[3, :, :].clone() 
            img = img[:3, :, :]  

        img = img.float() / 255.0  # (3, H, W)
        height, width = img.shape[1], img.shape[2]

        target_color = color_map[color].view(3, 1, 1).repeat(1, height, width) / 255.0

        blended_img = (img + target_color) / 2 # Blend the original image with target color
        # To create a more distinctive change, use the option below; increasing the color distinction will decrease differentiation in object details
        # blended_img = img * (1 - 0.7) + target_color * 0.7 # Blend the original image (0.3) with target color (0.7)

        blended_img = torch.clamp(blended_img, 0, 1)
        blended_img = (blended_img * 255).byte()

        if has_alpha: # Reattach the alpha channel if it was present
            blended_img = torch.cat((blended_img, alpha_channel.unsqueeze(0)), dim=0)

        return blended_img

    initial_image = color_overlap(image, initial_color)
    correct_image = color_overlap(image, target_color)
    incorrect_image = color_overlap(image, incorrect_color)

    if type == "train":
        return initial_image, correct_image, initial_color, target_color
    else:
        return initial_image, correct_image, incorrect_image, initial_color, incorrect_color

def generate_grid_image(image, count):
    _, img_height, img_width = image.shape
    max_items_per_row = int((10 ** 0.5) + 1)
    item_size = min(img_width, img_height) // max_items_per_row
    shrunken_image = F.resize(image, (item_size, item_size), antialias=True)
    
    if shrunken_image.shape[0] == 3:
        alpha_channel = torch.ones((1, item_size, item_size), dtype=image.dtype) * 255
        shrunken_image = torch.cat((shrunken_image, alpha_channel), dim=0)
    
    canvas = torch.zeros((4, img_height, img_width), dtype=image.dtype)

    for i in range(count):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        canvas[:, y:y + item_size, x:x + item_size] = shrunken_image

    return canvas

def apply_counting(image, param, type, initial_count=None):
    operation = param[0]
    num_objects_change = int(param[1:])
    
    if operation == '+':
        starting_options = (2, 3, 4, 5)
        starting_count = initial_count if initial_count is not None else random.choice(starting_options)        
        correct_count = starting_count + num_objects_change
        incorrect_count = starting_count - 1
    elif operation == '-':
        starting_options = (7, 6, 5, 4, 3)
        starting_count = initial_count if initial_count is not None else random.choice(starting_options)        
        correct_count = starting_count - num_objects_change
        incorrect_count = starting_count + 1
    else:
        raise ValueError("Invalid counting operation. Choose from '+1', '+2', '-1', or '-2'.")

    initial_image = generate_grid_image(image, starting_count)
    correct_image = generate_grid_image(image, correct_count)
    incorrect_image = generate_grid_image(image, incorrect_count)

    if type == "train":
        return initial_image, correct_image, starting_count, correct_count
    else:
        return initial_image, correct_image, incorrect_image, starting_count, incorrect_count

def apply_reflection(image, parameter, type):
    if parameter == "X":
        correct_image = F.vflip(image)
        incorrect_image = F.hflip(image)
        incorrect_option = "Y"
    elif parameter == "Y":
        correct_image = F.hflip(image)
        incorrect_image = F.vflip(image)
        incorrect_option = "X"
    else:
        raise ValueError("Invalid reflect factor. Choose from 'X' or 'Y'.")
    
    if type == "train":
        return image.clone(), correct_image, 0, parameter
    else:
        return image.clone(), correct_image, incorrect_image, 0, incorrect_option

def apply_resizing(image, factor, type):
    if factor == "0.5XY":
        correct_resize_factor = 0.5
        incorrect_resize_factor = 2.0
        incorrect_option = "2XY"
    elif factor == "2XY":
        correct_resize_factor = 2.0
        incorrect_resize_factor = 0.5
        incorrect_option = "0.5XY"
    else:
        raise ValueError("Invalid resize factor. Choose from '0.5XY' or '2XY'.")

    # Get the original dimensions
    _, original_height, original_width = image.shape

    # Calculate new dimensions
    correct_new_height = int(original_height * correct_resize_factor)
    correct_new_width = int(original_width * correct_resize_factor)
    
    incorrect_new_height = int(original_height * incorrect_resize_factor)
    incorrect_new_width = int(original_width * incorrect_resize_factor)

    correct_image = transforms.Resize((correct_new_height, correct_new_width), antialias=True)(image)
    incorrect_image = transforms.Resize((incorrect_new_height, incorrect_new_width), antialias=True)(image)

    if type == "train":
        return image.clone(), correct_image, 0, factor
    else:
        return image.clone(), correct_image, incorrect_image, 0, incorrect_option

def apply_rotation(image, angle, type):
    if angle == "+90":
        correct_angle = -90
        incorrect_angle = 180
    elif angle == "-90":
        correct_angle = 90
        incorrect_angle = 180
    elif angle == "180":
        correct_angle = 180
        incorrect_angle = random.choice([90, -90])
    else:
        raise ValueError("Invalid rotation angle. Choose from '+90', '-90', or '180'.")

    correct_image = F.rotate(image, correct_angle)
    incorrect_image = F.rotate(image, incorrect_angle)

    if type == "train":
        return image.clone(), correct_image, 0, angle
    else:
        return image.clone(), correct_image, incorrect_image, 0, incorrect_angle

def save_values_to_txt(initial_train, output_train, initial_test, incorrect_test):
    """
    Args:
        initial_train (int): Initial value for train input.
        output_train (int): Output value for train output.
        initial_test (int): Initial value for test input.
        incorrect_test (int): Incorrect value for test output.
    """
    with open(f"{args.output_directory}/output_{transform_id}.txt", "a") as file:
        file.write(f"Train_input: {initial_train}\n")
        file.write(f"Train_output: {output_train}\n")
        file.write(f"Test_input: {initial_test}\n")
        file.write(f"MC: {incorrect_test}\n")
    
def apply_transformation_chain(image, transformation, parameter, type, initial_value=None):
    if transformation == "Colour":
        if type == "train":
            initial_image, correct_image, input_val, output_val = apply_color(image, parameter, type="train")
            return initial_image, correct_image, input_val, output_val
        else:
            initial_image, correct_image, incorrect_image, input_val, incorrect_val = apply_color(
                image, parameter, type="test", initial_color=initial_value)
            return initial_image, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "Counting":
        if type == "train":
            initial_image, correct_image, input_val, output_val = apply_counting(image, parameter, type="train")
            return initial_image, correct_image, input_val, output_val
        else:
            initial_image, correct_image, incorrect_image, input_val, incorrect_val = apply_counting(
                image, parameter, type="test", initial_count=initial_value)
            return initial_image, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "Reflect":
        if type == "train":
            original, correct_image, input_val, output_val = apply_reflection(image, parameter, type="train")
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_reflection(image, parameter, type="test")
            return original, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "Resize":
        if type == "train":
            original, correct_image, input_val, output_val = apply_resizing(image, parameter, type="train")
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_resizing(image, parameter, type="test")
            return original, correct_image, incorrect_image, input_val, incorrect_val
    elif transformation == "2DRotation":
        if type == "train":
            original, correct_image, input_val, output_val = apply_rotation(image, parameter, type="train")
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_rotation(image, parameter, type="test")
            return original, correct_image, incorrect_image, input_val, incorrect_val

i = 0
while True:
    try:    
        train_image = next(train_iter)[0]
        test_image = next(test_iter)[0]
        
        train_values = {}
        
        # Process training and test images with identical transformations
        current_train_input = train_image.clone()
        current_test_input = test_image.clone()
        
        designated_last = last_trans_list[i] # Pick designated "last" transformation for this trial
        other_transformations = [t for t in transformations if t != designated_last]
        random.shuffle(other_transformations)
        transformations = other_transformations + [designated_last] # Create final order with designated last transformation at end
        
        count_index = None # Ensure Counting transformation happens last

        for transform_idx, (transform_name, parameter) in enumerate(transformations):
            if transform_name == "Counting":
                if transform_idx == 0:
                    current_train_output, current_test_output = current_train_input, current_test_input
                count_param, count_index = parameter, transform_idx
                continue

            # Apply transformation to 'train_input' and 'train_output'
            train_input, _, input_val, output_val = apply_transformation_chain(
                current_train_input, 
                transform_name, parameter, "train")

            _, train_output, _, _ = apply_transformation_chain(
                current_train_input if transform_idx == 0 else current_train_output, 
                transform_name, parameter, "train")
            
            if transform_idx == (len(transformations)-1):
                train_output_value = output_val
        
            # Apply the same transformation to 'test_input'
            test_input, test_output, _, input_val, output_val = result = apply_transformation_chain(
                current_test_input, 
                transform_name, parameter, "test", input_val
            )
            
            # Update inputs and store values for reference
            train_values[transform_idx] = input_val
    
            current_train_output, current_test_output = train_output.clone(), test_output.clone()
            current_train_input, current_test_input = train_input.clone(), test_input.clone()
        
        # Now process test correct and incorrect outputs
        current_test_correct = test_image.clone()
        current_test_incorrect = test_image.clone()
        
        for transform_idx, (transform_name, parameter) in enumerate(transformations):
            if transform_name == "Counting":
                continue

            # Apply correct version to build 'current_test_correct'
            _, test_correct, test_incorrect, input_val, incorrect_value = apply_transformation_chain(
                current_test_correct if transform_idx == 0 else current_test_correct,
                transform_name, parameter, "test", train_values[transform_idx])
            current_test_correct = test_correct.clone()

            if (transform_idx == (len(transformations)-1)):
                current_test_incorrect = test_incorrect.clone()
                

        if count_index is not None:
            # Train input & output
            current_train_input, _, count_val, count_output_val = apply_counting(current_train_input.clone(), count_param, type="train")
            
            _, current_train_output, _, _, _ = apply_counting(current_train_output.clone(), count_param, type="test", initial_count=count_val)

            # Test input, correct output, & incorrect output
            current_test_input, _, _, _, _ = apply_counting(current_test_input.clone(), count_param, type="test", initial_count=count_val)

            if (count_index == (len(transformations)-1)): # If Counting is the random incorrect option
                _, _, current_test_incorrect, _, count_incorrect_val = apply_counting(current_test_correct.clone(), count_param, type="test", initial_count=count_val)
                input_val = count_val
                train_output_value = count_output_val
                incorrect_value = count_incorrect_val
            else:
                _, current_test_incorrect, _, _, count_incorrect_val = apply_counting(current_test_incorrect.clone(), count_param, type="test", initial_count=count_val)

            _, current_test_correct, _, _, _ = apply_counting(current_test_correct.clone(), count_param, type="test", initial_count=count_val)
        
        # Save final transformed images
        transforms.ToPILImage()(current_train_input).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_train_0_input.png"))
        transforms.ToPILImage()(current_train_output).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_train_0_output.png"))
        transforms.ToPILImage()(current_test_input).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_0_input.png"))
        transforms.ToPILImage()(current_test_correct).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_mc_0_input.png"))
        transforms.ToPILImage()(current_test_incorrect).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_mc_1_input.png"))
        
        save_values_to_txt(input_val, train_output_value, input_val, incorrect_value)
        
        i += 1
        
    except StopIteration:
        break
