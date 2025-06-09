import os
import argparse
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as F
import torch

'''
Only 2 transformation pairs are supported.

Test Options:
1. Correct
2. Correct in Domain A, Incorrect in Domain B
3. Incorrect in Domain A, Correct in Domain B
4. No change
5. Doesn't apply
'''

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
                        help='Comma-separated list of 2 transformation:parameter pairs (e.g., "Counting:+1,2DRotation:+90")')
    parser.add_argument('--shuffle', type=bool, default=False,
                        help='To shuffle the objects dataset.')
    return parser.parse_args()

# ------------------------------------------------------------------------------------------------

def parse_transformation_list(transformation_list):
    pairs = [pair.strip() for pair in transformation_list.split(',')]
    return [tuple(pair.split(':')) for pair in pairs]

def order_counting_last(transformation_list):
    """Ensures that if 'Counting' is in the transformation list, it appears last."""
    if len(transformation_list) == 2 and transformation_list[0][0] == "Counting":
        transformation_list[0], transformation_list[1] = transformation_list[1], transformation_list[0]
    return transformation_list  

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
transformations = order_counting_last(parse_transformation_list(args.transformation_list))

# Create output directory
args.output_directory = args.output_directory + f"/{args.transformation_list}"

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

# Set up images
if "Resize" not in [t[0] for t in transformations]:
    # If Resize is not in the transformations, we will enlarge the images to 600x600
    base_transform = transforms.Compose([
        transforms.Resize((600, 600), antialias=True)
    ])
else:
    base_transform = transforms.Compose([
        transforms.Resize((300, 300), antialias=True)
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

train_iter = iter(train_loader)
test_iter = iter(test_loader)

transform_id = '_'.join([f"{t}{p}" for t, p in transformations])

# Incompatibilities: 1) Support only 2 transformation pairs, 2) Reflect & 2DRotation are incompatible
if len(transformations) > 2:
    raise ValueError(f"We currently only support 2 transformation pairs. Please limit your input to 2 transformation pairs.")

# Check for Reflect and 2DRotation combination
transformation_types = [t[0] for t in transformations]
if "Reflect" in transformation_types and "2DRotation" in transformation_types:
    raise ValueError("Reflect and 2DRotation transformations cannot be used together.")

# Check for duplicates
transformation_counts = {}
for transformation, parameter in transformations:
    if transformation not in transformation_counts:
        transformation_counts[transformation] = set()
    transformation_counts[transformation].add(parameter)

duplicates = {key: values for key, values in transformation_counts.items() if len(values) > 1}

if duplicates:
    raise ValueError(f"Multiple parameter values are identified with one transformation type: {duplicates}.")


#---------------------------------------------------------------
# Transformation functions

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

def apply_counting(image, param, type, keep_initial_value=None, avoid_initial_value=None):
    if param not in {'+1', '+2', '-1', '-2', 'x2', 'x3', 'd2', 'd3'}:
        raise ValueError("Invalid counting operation. Choose from '+1', '+2', '-1', '-2', 'x2', 'x3', 'd2', or 'd3'.")
    
    starting_options_map = {
        '+': [2, 3, 4, 5],
        '-': [7, 6, 5, 4], # removed 3 for -2
        'x': {2: [2, 3, 4], 3: [1, 2, 3]},
        'd': {2: [8, 6, 4], 3: [9, 6]} # removed 3 for d3
    }

    def calculate_counts(operation, starting_count, param_num):
        if operation == '+':
            correct_count = starting_count + param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])
        elif operation == '-':
            correct_count = starting_count - param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])
        elif operation == 'x':
            correct_count = starting_count * param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])
        elif operation == 'd':
            correct_count = starting_count // param_num
            return correct_count, random.choice([correct_count - 1, correct_count + 1])

    operation = param[0]  # +, -, x, d
    param_num = int(param[1:])  # 1, 2, or 3

    if keep_initial_value is not None:
        starting_count = int(keep_initial_value)
    else:
        starting_options = starting_options_map[operation] if operation in ['+', '-'] else starting_options_map[operation][param_num]
        starting_count = random.choice([num for num in starting_options if num != avoid_initial_value] or starting_options)
    correct_count, incorrect_count = calculate_counts(operation, starting_count, param_num)

    initial_image = generate_grid_image(image, starting_count)
    correct_image = generate_grid_image(image, correct_count)
    incorrect_image = generate_grid_image(image, incorrect_count)

    if (type == "train"):
        return initial_image, correct_image, starting_count, correct_count
    elif (type == "test"):
        return initial_image, correct_image, incorrect_image, starting_count, [correct_count, incorrect_count]

def apply_reflection(image, parameter, type):
    if parameter == "X":
        correct_image = F.vflip(image)
        incorrect_image = random.choice([F.hflip(image), image])  # Flip along the Y-axis for incorrect
        incorrect_option = "Y"
    elif parameter == "Y":
        correct_image = F.hflip(image)
        incorrect_image = random.choice([F.vflip(image), image])  # Flip along the X-axis for incorrect
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
        return image.clone(), correct_image, incorrect_image, 0, incorrect_option

def paste_on_600(img: torch.Tensor, canvas_size: int = 600) -> torch.Tensor:
    """
    Returns a 600 × 600 tensor.  The input *content* is left untouched:
    - if smaller than 600 it is centred and padded with zeros / transparent alpha
    - if one edge happens to be >600 we first down-scale so the larger edge becomes 600
      (this should not happen in your current pipeline, but keeps things safe)
    """
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
    base_img      = image
    if enlarge_first:
        H, W   = image.shape[1:]
        base_img = F.resize(image, (H * 2, W * 2), antialias=True)
    image = base_img

    def get_opts(factor):
        # parse scale and dim without regex
        if factor.startswith("0.5"):
            scale = 0.5
            dim = factor[3:]   # Skip "0.5"
        else:
            scale = 2.0
            dim = factor[1:]   # Skip "2"

        # Correct (w,h)
        correct = (scale if "X" in dim else 1, scale if "Y" in dim else 1)

        # Pick a random other dim
        dims = ["X", "Y", "XY"]
        other = random.choice([d for d in dims if d != dim])

        # incorrect (w,h) and its key
        incorrect = (scale if "X" in other else 1, scale if "Y" in other else 1)
        incorrect_key = f"{scale}{other}"

        return correct, incorrect, incorrect_key
    try:
        correct_resize_factors, incorrect_resize_factors, incorrect_option = get_opts(factor)   
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
        return paste_on_600(image.clone()), paste_on_600(correct_image), 0, factor
    elif (type == "test"):
        return paste_on_600(image.clone()), paste_on_600(correct_image), paste_on_600(incorrect_image), 0, incorrect_option

def apply_rotation(image, angle, type, keep_initial_value=None, avoid_initial_value=None):
    if keep_initial_value is not None:
        initial_rotation = keep_initial_value
    else:
        initial_rotation = random.choice(
            [angle for angle in ["+45", "-45", "+90", "-90", "+135", "-135"] if angle != avoid_initial_value] 
            or ["+45", "-45", "+90", "-90", "+135", "-135"]
        )

    def parse_angle(angle):
        if angle[:1] == "+":
            return -int(angle[1:])
        elif angle[:1] == "-":
            return int(angle[1:])
        elif angle == "180":
            return 180

    random_incorrect_angle = random.choice(["-90", "+90"])

    original_image = F.rotate(image, parse_angle(initial_rotation))
    correct_image = F.rotate(original_image, parse_angle(angle))
    incorrect_image = F.rotate(correct_image, parse_angle(random_incorrect_angle))

    # Add initial_rotation and angle together, handling + and - signs
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

    final_initial_angle = combine_angles(initial_rotation, "0")  # Initial rotation is always the first angle
    final_correct_angle = combine_angles(initial_rotation, angle)
    final_incorrect_angle = combine_angles(combine_angles(initial_rotation, angle), random_incorrect_angle)

    if (type == "train"):
        return original_image, correct_image, final_initial_angle, combine_angles(initial_rotation, angle)
    elif (type == "test"):
        return original_image, correct_image, incorrect_image, final_initial_angle, [final_correct_angle, final_incorrect_angle]

def save_values_to_txt(index, initial_train, output_train, initial_test, correct_test, incorrect_test0, incorrect_test1):
    """
    Args:
        initial_train (str): Initial value for train input.
        output_train (str): Output value for train output.
        initial_test (str): Initial value for test input.
        correct_test (str): Correct value for test output.
        incorrect_test0 (str): Incorrect value for test output.
        incorrect_test1 (str): Incorrect value for test output.
    """
    with open(f"{args.output_directory}/output_{transform_id}.txt", "a") as file:
        file.write(f"Trial: {index}\n")
        file.write(f"Train_input: {initial_train}\n")
        file.write(f"Train_output: {output_train}\n")
        file.write(f"Test_input: {initial_test}\n")
        file.write(f"Test_correct_output: {correct_test}\n")
        file.write(f"Test_incorrect_output0: {incorrect_test0}\n")
        file.write(f"Test_incorrect_output1: {incorrect_test1}\n")

def apply_transformation_chain(image, transformation, parameter, type, keep_initial_value=None, avoid_initial_value=None):
    if transformation == "Counting":
        if type == "train":
            initial_image, correct_image, input_val, output_val = apply_counting(
                image, parameter, type="train", keep_initial_value=keep_initial_value)
            return initial_image, correct_image, input_val, output_val
        else:
            initial_image, correct_image, incorrect_image, input_val, incorrect_val = apply_counting(
                image, parameter, type="test", keep_initial_value=keep_initial_value if keep_initial_value is not None else None, avoid_initial_value=avoid_initial_value)
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
            original, correct_image, input_val, output_val = apply_rotation(image, parameter, type="train", keep_initial_value=keep_initial_value)
            return original, correct_image, input_val, output_val
        else:
            original, correct_image, incorrect_image, input_val, incorrect_val = apply_rotation(image, parameter, type="test", keep_initial_value=keep_initial_value if keep_initial_value is not None else None, avoid_initial_value=avoid_initial_value)
            return original, correct_image, incorrect_image, input_val, incorrect_val

i = 0
while True:
    try:    
        train_image = next(train_iter)[0]
        test_image = next(test_iter)[0]
        
        # Process training & test inputs with identical transformations
        # For Counting & 2DRotation, we differ in initial (input) count & rotation values
        current_train_input = train_image.clone()
        current_test_input = test_image.clone()
        
        train_input_txt = []
        train_output_txt = []
        test_input_txt = []
        test_correct_output_txt = []
        test_incorrect_output0_txt = []
        test_incorrect_output1_txt = []

        # ——————————————————————————————————————————————————————————————————————————————
        # TRAIN STIMULI: 1) train_input, 2) train_output
        # ——————————————————————————————————————————————————————————————————————————————
        for transform_idx, (transform_name, parameter) in enumerate(transformations):
            if transform_idx == 0:
                train_input, train_output, input_value, output_value = apply_transformation_chain(
                current_train_input, 
                transform_name, parameter, "train")
            # For Counting and 2DRotation, ensure we keep the intial starting count / rotation the same
            # We don't need to do this for Reflect and Resize because they don't have an initial transformation
            elif transform_idx == 1:
                train_input, _, input_value, output_value = apply_transformation_chain(
                    current_train_input, 
                    transform_name, parameter, "train")

                _, train_output, _, _ = apply_transformation_chain(
                    current_train_output, 
                    transform_name, parameter, "train", keep_initial_value=input_value) # Keep initial value to ensure test inputs don't start with same

            if transform_name == "Counting" or transform_name == "2DRotation":
                train_input_txt += [(transform_name, input_value)]
            else:
                train_input_txt += [(transform_name, "0")]

            train_output_txt += [(transform_name, output_value)]

            current_train_input = train_input.clone()
            current_train_output = train_output.clone()

        # ——————————————————————————————————————————————————————————————————————————————
        # TEST STIMULI: 1) test_input, 2) test_correct_output, 3) test_incorrect_output0, 4) test_incorrect_output1
        # ——————————————————————————————————————————————————————————————————————————————
        for transform_idx, (transform_name, parameter) in enumerate(transformations):
            # Apply transformation to 'train_input' and 'train_output'
            if transform_idx == 0:
                test_input, test_output, test_incorrect, input_value, incorrect_value0 = apply_transformation_chain(
                current_test_input, 
                transform_name, parameter, "test", avoid_initial_value=train_input_txt[0][0])
            # For Counting and 2DRotation, ensure we keep the intial starting count / rotation the same
            # We don't need to do this for Reflect and Resize because they don't have an initial transformation
            elif transform_idx == 1:
                test_input, _, _, input_value, _ = apply_transformation_chain(
                    current_test_input, 
                    transform_name, parameter, "test", avoid_initial_value=train_input_txt[1][0]) # Avoid initial value to ensure test inputs don't start with same as train

                _, test_output, test_incorrect_output1, _, incorrect_value1 = apply_transformation_chain(
                    current_test_correct, 
                    transform_name, parameter, "test", keep_initial_value=input_value, avoid_initial_value=train_input_txt[1][0])

                # Make incorrect A and correct B
                _, test_incorrect_output0, _, _, _ = apply_transformation_chain(
                    current_test_incorrect0, 
                    transform_name, parameter, "test", keep_initial_value=input_value, avoid_initial_value=train_input_txt[1][0])

            if not (transform_name == "Counting" or transform_name == "2DRotation"):
                test_correct_output_txt = [(transform_name, parameter)]
                test_input_txt += [(transform_name, "0")]
            else:
                # incorrect_value0 & 1 are lists that contain the correct & incorrect end values
                test_input_txt += [(transform_name, input_value)]
                if transform_idx == 0:
                    test_correct_output_txt += [(transform_name, incorrect_value0[0])]
                    incorrect_value0 = incorrect_value0[1]
                elif transform_idx == 1:
                    test_correct_output_txt += [(transform_name, incorrect_value1[0])]
                    test_incorrect_output1_txt = [test_correct_output_txt[0], (transformations[1][0], incorrect_value1[1])]
                    test_incorrect_output0_txt = [(transformations[0][0], incorrect_value0), test_correct_output_txt[1]]

            current_test_input = test_input.clone()
            current_test_correct = test_output.clone()
            current_test_incorrect0 = test_incorrect.clone()
        
        # Save transformation values
        save_values_to_txt(
            i,
            train_input_txt, train_output_txt, 
            test_input_txt, test_correct_output_txt, 
            test_incorrect_output0_txt, test_incorrect_output1_txt
        )

        os.makedirs(args.output_directory, exist_ok=True)

        # Save final transformed images
        transforms.ToPILImage()(current_train_input).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_train_0_input.png"))
        transforms.ToPILImage()(current_train_output).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_train_0_output.png"))
        transforms.ToPILImage()(current_test_input).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_0_input.png"))
        transforms.ToPILImage()(current_test_correct).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_mc_0_output.png"))
        transforms.ToPILImage()(test_incorrect_output0).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_mc_1_output.png"))
        transforms.ToPILImage()(test_incorrect_output1).save(
            os.path.join(args.output_directory, f"{transform_id}_{i}_test_mc_2_output.png"))
                
        i += 1
        
    except StopIteration:
        print(f"✓ Generated {i} trial(s) for {args.transformation_list}.")
        break
