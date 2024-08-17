from PIL import Image
import random
import os
import re
import operator
import argparse

'''
Input directory: directory with stimuli to be transformed
Output directory: path of existing / new target directory for saving transformed stimuli
Next index: define starting index of new stimuli to standardize naming
Trials: trials to run, leave as None if max trials is preferred

Transformations:
"Counting": Apply a mathematical operation (addition, subtraction, multiplication, division)
    Parameters: 
        "+1","+2","-1","-2"
        
"Resize": Resize horizontally(X), vertically(Y)
    Parameters: 
        "0.5XY", "2XY"
        
"Colour": Colour Change
    Parameters: 
        "Red","Green", "Blue"
        
"Reflect": Reflect along the X or Y axis
    Parameters: 
        "X", "Y"
        
"2DRotation": Rotate clockwise(+) or counterclockwise(-) by 90 degrees or rotate 180 degrees
    Parameters: 
        "+90", "-90", "180"
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
    parser.add_argument('--colour', type=str, default="no change",
                        help='The colour to change the image to. Choices: "Red", "Green", "Blue".')
    parser.add_argument('--rotate', type=str, default="0",
                        help='The angle to rotate the image by. Choices: "+90", "-90", "180".')
    parser.add_argument('--count', type=str, default="0",
                        help='The mathematical operation to apply to the image. Choices: "+1", "+2", "-1", "-2".')
    parser.add_argument('--resize', type=str, default="XY",
                        help='The factor to resize the image by. Choices: "0.5XY", "2XY".')
    parser.add_argument('--reflect', type=str, default="no change",
                        help='The axis to reflect the image on. Choices: "X", "Y".')
    parser.add_argument('--randomize', type=bool, default=False,
                       help='Randomize transformations applied. Choices: True, False.')

    return parser.parse_args()

# ------------------------------------------------------------------------------------------------

# Variables
angles = ("0","+90","-90","180")
factors = ("XY","0.5XY","2XY")
operations = ("0","+1","+2","-1","-2")
colours = ("no change","Red","Green","Blue")
reflections = ("no change","X","Y")
add = sub = (1,2)
mul = div = 2
train_1 = []
train_2 = []
test = []
save_inputs = []
transformations = ("Counting", "Resize", "Colour", "Reflect", "2DRotation")
suffixes = (
    f"train_0_input.png",f"train_0_output.png",f"train_1_input.png",f"train_1_output.png",
    f"test_0_input.png",f"test_mc_0_input.png",f"test_mc_1_input.png"
    )
    
# Create ouput text file to save transformations
def save_txt(train_0_in,train_0_out,train_1_in,train_1_out,test_in,mc):
    with open(f"output_{transformation}{parameter}.txt", "a") as file:
        file.write(f'train_0_input: {train_0_in}\n')
        file.write(f'train_0_output: {train_0_out}\n')
        file.write(f'train_1_input: {train_1_in}\n')
        file.write(f'train_1_output: {train_1_out}\n')
        file.write(f'test_input: {test_in}\n')
        file.write(f'mc: {mc}\n')

# Crop image
def crop(image_input):
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    
    new_size = min(img.size)
    left = (img.width - new_size) / 2
    top = (img.height - new_size) / 2
    right = (img.width + new_size) / 2
    bottom = (img.height + new_size) / 2
    cropped = img.crop((left, top, right, bottom))
    
    return cropped

# Save image to output directory
def save_image(image_input, transformation, param, index, file_suffix, final = False):
    all_i = []
    for image, suffix in zip(image_input,file_suffix):
        cropped = crop(image)
        if final:
            file_path = os.path.join(out_directory, f"{transformation}{param}_{index}_{suffix}") 
            cropped.save(file_path,format="PNG")
        all_i.append(cropped)
    return all_i

# Randomly select num elements from parameters that are not to_exclude
def selector(to_exclude, parameters, num):
    filtered = [parameter for parameter in parameters if parameter != to_exclude]
    selected = random.sample(filtered,num)
    return selected  

# Reflect image on an axis
def reflect_image(img_path, axis):
    if isinstance(img_path, Image.Image):
        image = img_path
    else:
        image = Image.open(img_path)

    if axis == 'Y':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif axis == 'X':
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Invalid Axis")
    
    return flipped_image

# Rotate image by a certain degree 
def rotate_image(img_path, angle):
    if angle not in angles and angle != 0:
        raise ValueError("Invalid Angle")
    
    if angle == "+90":
        angle = -90
    elif angle == "-90":
        angle = 90
    elif angle == "180":
        angle = 180
    else:
        angle = 0
    
    if isinstance(img_path, Image.Image):
        image = img_path
    else:
        image = Image.open(img_path)
    
    rotated_image = image.rotate(angle, expand=False)
    
    return rotated_image

# Resize image by a certain axis
def resize_img(img_path, factor, original=False, mc=False):
    if not mc:
        if factor not in factors:
            raise ValueError("Invalid Factor")
    
    # Regex pattern to extract the numeric factor and axis
    pat = r"([0-9.]+)([XY])"
    matched = re.match(pat, factor)
    resize_factor = float(matched.group(1))
    axis = matched.group(2)
    
    initial_shrink_factor = 0.8
    factor_x = factor_y = initial_shrink_factor
    
    if not original:
        if axis == "X":
            factor_x *= resize_factor
        else:
            factor_y *= resize_factor
    
    if isinstance(img_path, Image.Image):
        img = img_path
    else:
        img = Image.open(img_path)

    new_size = (int(img.width * factor_x), int(img.height * factor_y))
    resized_img = img.resize(new_size, Image.LANCZOS)
    
    # Handling transparency and background creation
    if img.mode == 'RGBA' or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGBA', (img.width, img.height), (255, 255, 255, 0))
    else:
        background = Image.new('RGB', (img.width, img.height), (255, 255, 255))
    
    upper_left_x = (background.width - resized_img.width) // 2
    upper_left_y = (background.height - resized_img.height) // 2
    
    # Paste the resized image onto the background
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background.paste(resized_img, (upper_left_x, upper_left_y), resized_img)
    else:
        background.paste(resized_img, (upper_left_x, upper_left_y))
    
    return background

# Change colour of the image
def colour_change(img_path, colour_name):
    colour_map = {
        'Red': (255, 0, 0, 200),  
        'Green': (0, 128, 0, 200), 
        'Blue': (0, 0, 255, 200)
    } 
    colour = colour_map.get(colour_name)
    if colour is None:
        raise ValueError("Invalid Colour")
    
    if isinstance(img_path, Image.Image):
        img = img_path.convert("RGBA")
    else:
        with Image.open(img_path).convert("RGBA") as img:
            pass  

    width, height = img.size

    result_img = Image.new('RGBA', img.size)
    pixels = result_img.load()

    for x in range(width):
        for y in range(height):
            original_pixel = img.getpixel((x, y))
            if original_pixel[3] > 0: 
                new_colour = (
                    int((original_pixel[0] + colour[0]) / 2),
                    int((original_pixel[1] + colour[1]) / 2),
                    int((original_pixel[2] + colour[2]) / 2),
                    original_pixel[3],
                )
                pixels[x,y] = new_colour
            else:
                pixels[x, y] = original_pixel
    return result_img

# Generate math operations 
def count_builder(oper):
    if oper not in operations:
        raise ValueError("Invalid Operation")
    num = int(oper[1:])
    math_operations = {
        '+': operator.add,
        '-': operator.sub
    }
    if oper.startswith('x'):
        starting = (2,4)
        math_op = math_operations['x']
        mc_op = math_operations['d']
        mc_num = num
    elif oper.startswith('+'):
        starting = (1,2,3,4,5)
        math_op = math_operations['+']
        mc_op = math_operations['-']
        mc_num = random.choice([1,2])
    elif oper.startswith('-'):
        starting = (7,6,5,4,3)
        math_op = math_operations['-']
        mc_op = math_operations['+']
        mc_num = random.choice([1,2])
    else:
        math_op = math_operations['d']
        starting = (2,4,6)
        mc_op = math_operations['x']
        mc_num = num
    if oper.startswith('x'):
        train_1, train_2 = random.sample(starting,2)
        test_0 = random.choice([6,8])
    else:
        train_1, train_2, test_0 = random.sample(starting,3)
    train_1_out = math_op(train_1, num)
    train_2_out = math_op(train_2, num)
    test_mc_0 = math_op(test_0, num)
    test_mc_1 = mc_op(test_mc_0,mc_num)
    save_txt(train_1,train_1_out,train_2,train_2_out,test_0, test_mc_1)
    return (train_1,train_1_out,train_2,train_2_out,test_0,test_mc_0,test_mc_1) 

# Build image with num amount of elements    
def count_generator(img_path, num):
    num = int(num)
    img = crop(img_path)
    canvas = Image.new('RGBA', (img.width, img.height), (0, 0, 0, 0))
    max_items_per_row = int((10 ** 0.5) + 1)
    item_size = min(img.width, img.height) // max_items_per_row
    
    for i in range(num):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        shrunken_img = img.resize((item_size, item_size), Image.LANCZOS)
        mask = None
        canvas.paste(shrunken_img, (x, y), mask)
    
    return canvas

# Save images generated from the Counting transformation
def transform_save_count(index, param, next_index, images = None):
    results = count_builder(param)
    if images == None:
        images = (
            train_1[index],train_1[index],train_2[index],train_2[index],
            test[index],test[index],test[index]
        )
    else:
        images = images
    inputs = [None]*7
    for i in range(7):
        inputs[i] = count_generator(images[i],results[i])
    return save_image(inputs, "Counting", param, index + next_index, suffixes)

# Save images generated from the Colour transformation    
def transform_save_colour(index, param, inputs, next_index,  images = None):
    processed = []
    if images == None:
        images = (train_1[index],train_2[index],test[index])
    else:
        images = [images[1], images[3], images[5]]
    for img in images:
        processed.append(colour_change(img,param))
    save_inputs = (
        train_1[index],processed[0],train_2[index],processed[1],test[index],processed[2],colour_change(test[index],inputs)
    )
    save_txt("",param,"",param," ",inputs)
    return save_image(save_inputs,"Colour", param, index+next_index, suffixes)

# Save images generated from the Resize transformation
def transform_save_resize(index, param, inputs, next_index, images = None):
    originals = []
    processed = []
    if images == None:
        images = (
            train_1[index],train_2[index],test[index]
        )
        for img in images:
            originals.append(resize_img(img,param,original=True))
            processed.append(resize_img(img,param))
    else:
        og = [images[0], images[2], images[4]]
        images = [images[1], images[3], images[5]]
        for o in og:
            originals.append(resize_img(o,param,original=True))
        for img in images:
            processed.append(resize_img(img,param))
    save_inputs = (
        originals[0], processed[0], originals[1], processed[1], 
        originals[2], processed[2], resize_img(test[index],inputs, mc=True)
        )
    save_txt("0",param,"0",param,"0",inputs)
    return save_image(save_inputs, "Resize", param, index+next_index, suffixes)

# Save images generated from the Reflect transformation
def transform_save_reflect(index, param, false_axis, next_index, images = None):
    processed_inputs = []
    if images == None:
        images = (
            train_1[index],train_2[index],
            test[index],test[index]
        )
    else:
        images = [images[1], images[3], images[5], images[6]]
    input_params = (
        param,param,param,false_axis
    )
    for img,input_param in zip(images,input_params):
        processed_inputs.append(reflect_image(img,input_param))
    save_inputs = (
        train_1[index], processed_inputs[0], train_2[index], processed_inputs[1], 
        test[index], processed_inputs[2], processed_inputs[3]
        )
    save_txt("0",param,"0",param,"0",false_axis)
    return save_image(save_inputs, "Reflect", param, index+next_index, suffixes)

# Save images generated from the Rotate transformation
def transform_save_rotate(index, param, inputs, next_index, images = None):
    if images == None:
        images = (
            train_1[index],train_1[index],train_2[index],train_2[index],
            test[index],test[index],test[index]
        )
    else:
        images = images
    input_params = (
        0,param,0,param,0,param,inputs
    )
    save_inputs = []
    for img,input_param in zip(images,input_params):
        save_inputs.append(rotate_image(img,input_param))
    save_txt(0,param,0,param,0,inputs)
    return save_image(save_inputs,"2DRotation", param, index+next_index, suffixes)

if __name__ == "__main__":
    args = parse_arguments()

    input_directory = args.input_directory
    output_directory = args.output_directory
    next_index = args.next_index
    trials = args.trials
    colour = args.colour
    rotation = args.rotate
    counting = args.count
    resize = args.resize
    reflection = args.reflect
    randomize = args.randomize

    if randomize:
        colour = random.choice(colours)
        rotation = random.choice(angles)
        counting = random.choice(operations)
        resize = random.choice(factors)
        reflection = random.choice(reflections)

    all_trans = {"Colour":colour, "Reflect":reflection, "Resize":resize, "2DRotation":rotation, "Counting":counting}

    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    print(f"Next Index: {next_index}")
    print(f"Trials: {trials}")
    print(f"Colour: {colour}")
    print(f"Rotation: {rotation}")
    print(f"Counting: {counting}")
    print(f"Resize: {resize}")
    print(f"Reflection: {reflection}")
    print(f"Randomize: {randomize}")

    # Initialise output directory
    out_directory = output_directory
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    # Filter and initialise train and test image paths 
    in_directory = input_directory
    entries = os.listdir(in_directory)
    paths = [os.path.join(in_directory, entry) for entry in entries]
    filtered_paths = [path for path in paths if not path.endswith(".DS_Store")]
    remainder = len(filtered_paths) % 3
    if remainder != 0:
        filtered_paths = filtered_paths[:-(remainder)]
    sublist_len = int(len(filtered_paths)/3)

    # Create sets of train and test image paths
    def create_sets(trials):
        train_1 = filtered_paths[:trials]
        train_2 = filtered_paths[trials : (2*trials)]
        test = filtered_paths[(2*trials) : (3*trials)]
        return train_1,train_2, test

    if trials is None:
        trials = sublist_len
        train_1,train_2,test = create_sets(trials)
    elif trials <= sublist_len and trials > 0:
        sublist_len = trials
        train_1,train_2,test = create_sets(trials)
    else:
        raise ValueError(f"The maximum number of trials possible is {sublist_len}.")

    all_imgs = []
    for i in range(sublist_len):
        all_imgs = []
        cumulative_transformations = []
        cumulative_parameters = []
        count = 0
        for transformation, parameter in all_trans.items():
            cumulative_transformations.append(transformation)
            cumulative_parameters.append(parameter)
            if transformation not in transformations:
                raise ValueError(f"Transformation not found")
            
            elif transformation == "Reflect":
                axis = parameter
                if parameter == "no change":
                    continue
                count += 1
                false_axis = "Y" if axis == "X" else "X"
                r = transform_save_reflect(i, axis, false_axis, next_index, images=all_imgs[count-2] if all_imgs!=[] else None)
                all_imgs.append(r)
                    
            elif transformation == "Colour":
                colour = parameter 
                if parameter == "no change":
                    continue
                count += 1
                tmp = colours[1:]
                available_colours = [c for c in tmp if c != parameter]
                colour_mc_1 = random.choice(available_colours)
                r = transform_save_colour(i, colour, colour_mc_1, next_index, images=all_imgs[count-2] if all_imgs!=[] else None)
                all_imgs.append(r)
                    
            elif transformation == "Resize":
                factor = parameter
                if parameter == "XY":
                    continue
                count += 1
                pat = r"([0-9.]+)([XY])"
                matched = re.match(pat,factor)
                resize_factor = matched.group(1)
                axis = matched.group(2)
                if resize_factor == "0.5":
                    mc_factor = "2"
                else:
                    mc_factor = "0.5"
                factor_mc_1 = mc_factor+axis
                r = transform_save_resize(i, factor, factor_mc_1, next_index, images=all_imgs[count-2] if all_imgs!=[] else None)
                all_imgs.append(r)
                        
            elif transformation == "2DRotation":
                angle = parameter
                if angle == "0":
                    continue  
                count += 1
                if angle == "180":
                    angle_mc_1 = random.choice(angles[:2])
                else:
                    angle_mc_1 = "180"
                r = transform_save_rotate(i, angle, angle_mc_1, next_index, images=all_imgs[count-2] if all_imgs!=[] else None)
                all_imgs.append(r)
                    
            else:
                operation = parameter
                if operation == "0":
                    continue
                count += 1
                r = transform_save_count(i, operation, next_index, images=all_imgs[count-2] if all_imgs!=[] else None)
                all_imgs.append(r)
        
        cumulative_transformation = "_".join(cumulative_transformations)
        cumulative_parameter = "_".join(cumulative_parameters)
        save_image(all_imgs[-1], cumulative_transformation, cumulative_parameter, i + next_index, suffixes, final=True)
