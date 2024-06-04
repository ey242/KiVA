from PIL import Image
import random
import os
import re
import operator
import numpy as np

'''
Input directory: directory with stimuli to be transformed
Output directory: path of existing / new target directory for saving transformed stimuli
Next index: define starting index of new stimuli to standardize naming
Trials: trials to run, leave as None if max trials is preferred

Transformations:
"Counting": Apply a mathematical operation (addition, subtraction)
    Parameters: 
        "+1","+2",
        "-1","-2",
        "x2","x3",
        "d2","d3"
        
"Resize": Resize bigger (2XY), smaller (0.5XY), and variations
    Parameters: 
        "0.5XY", "2XY" 
        
"Colour": Colour Change
    Parameters: 
        "Red", "Yellow", "Green", "Blue", "Grey"
        
"Reflect": Reflect along the X or Y axis
    Parameters: 
        "X", "Y"
        
"2DRotation": Rotate clockwise(+) or counterclockwise(-) by 90 degrees or rotate 180 degrees
    Parameters: 
        "+45", "-45", "+90", "-90", "+135", "-135", "180"
'''
transformation = "2DRotation"
parameter = "180"

input_directory = "/Users/anisamajhi/Desktop/Stimuli100_50_Hard/Objects/Objects100_Achiral"
output_directory = f"/Users/anisamajhi/Desktop/Stimuli100_50_Hard/2DRotation_Objects100"
next_index = 0
trials = None

# ------------------------------------------------------------------------------------------------

# Variables
angles = ("0", "+45", "-45", "+90", "-90", "+135", "-135", "+225", "-225", "+270", "-270", "+315", "180", "360")
factors = ["0.5X", "0.5Y", "0.5XY", "2X", "2Y", "2XY"]
operations = ("+1","+2","-1","-2","x2","x3","d2","d3")
colours = ("Red", "Yellow", "Green", "Blue", "Grey")
reflections = ("X","Y","XY")
add = sub = (1,2)
mul = div = 2
train = [] #1 training only
test = []
save_inputs = []
transformations = ("Counting", "Resize", "Colour", "Reflect", "2DRotation")
suffixes = (
    f"train_input.png",f"train_output.png", # 1 training only
    f"test_input.png",f"test_mc_0_input.png",f"test_mc_1_input.png" # 3 choices
    )
    
# Initialise output directory
out_directory = output_directory
if not os.path.isdir(out_directory):
    os.mkdir(out_directory)

# Filter and initialise train and test image paths 
in_directory = input_directory
entries = os.listdir(in_directory)
paths = [os.path.join(in_directory, entry) for entry in entries]
filtered_paths = [path for path in paths if not path.endswith(".DS_Store")]
remainder = len(filtered_paths) % 2 #divisible by 2: one training, one test
if remainder != 0:
    filtered_paths = filtered_paths[:-(remainder)]
sublist_len = int(len(filtered_paths)/2)
    
# Create sets of train and test image paths
def create_sets(trials):
    train = filtered_paths[:trials]
    test = filtered_paths[trials : (2*trials)]
    return train,test

if trials is None:
    trials = sublist_len
    train,test = create_sets(trials)
elif trials <= sublist_len and trials > 0:
    sublist_len = trials
    train,test = create_sets(trials)
else:
    raise ValueError(f"The maximum number of trials possible is {sublist_len}.")

# Create ouput text file to save transformations
def save_txt(train_in,train_out,test_in,mc):
    with open(f"output_{transformation}{parameter}.txt", "a") as file:
        file.write(f'train_input: {train_in}\n')
        file.write(f'train_output: {train_out}\n')
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

def save_image(image_input, transformation, param, index, file_suffix):
    for image, suffix in zip(image_input,file_suffix):
        cropped = crop(image)
        file_path = os.path.join(out_directory, f"{transformation}{param}_{index}_{suffix}")
        cropped.save(file_path,format="PNG")

# Randomly select num elements from parameters that are not to_exclude
def selector(to_exclude, parameters, num):
    to_exclude = [to_exclude] if not isinstance(to_exclude, list) else to_exclude # Convert to list

    filtered = [parameter for parameter in parameters if parameter not in to_exclude]

    selected = random.sample(filtered, num)
    return selected

# Reflect image on an axis
def reflect_image(img_path, axis):
    with Image.open(img_path) as image:
        if axis == 'Y':
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif axis == "X":
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif axis == "XY":
            flipped_X_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image = flipped_X_image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("Invalid Axis")
        return flipped_image

# Rotate image by a certain degree 
def rotate_image(img_path, angle):
    if angle not in angles and angle != 0:
        print(angle)
        raise ValueError("Invalid Angle")
    
    angle_str = str(angle)
    
    if angle_str[:1] == "+":
        angle = -int(angle_str[1:])
    elif angle_str[:1] == "-":
        angle = int(angle_str[1:]) 
    elif angle_str == "180":
        angle = 180
    else:
        angle = 0

    with Image.open(img_path) as image:
        original_mode = image.mode
        # Convert to 'RGBA' to ensure an alpha channel is present
        image = image.convert('RGBA')
        
        rotated_image = image.rotate(angle, expand=False)
        
        white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        transparent_bg = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        # Calculate position to paste the rotated image
        bg_w, bg_h = white_bg.size
        img_w, img_h = rotated_image.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        
        transparent_bg.paste(rotated_image, offset, rotated_image)
        
        # Composite the transparent background w/rotated image onto white background
        combined = Image.alpha_composite(white_bg, transparent_bg)
        
        if original_mode != 'RGBA': # Convert if necessary
            combined = combined.convert(original_mode)
        
        return combined

# Resize images
def resize_indiv(img_path, factor, original = False):
    if factor not in factors:
        print(factor)
        raise ValueError("Invalid Parameter")
    
    resize_factor = re.findall(r'[\d.]+', factor)[0]
    axis = re.findall(r'[A-Za-z]+', factor)[0]

    if "." in resize_factor:
        resize_factor = 0.5 
        initial_shrink_factor = 0.8
    else:
        resize_factor = 2  
        initial_shrink_factor = 0.5
    
    
    factor_x = factor_y = initial_shrink_factor  
    if not original:
        if axis == "X":
            factor_x *= resize_factor
        elif axis == "Y":
            factor_y *= resize_factor 
        elif axis == "XY": 
            factor_x *= resize_factor
            factor_y *= resize_factor

    
    with Image.open(img_path) as img:
        new_size = (int(img.width * factor_x), int(img.height * factor_y))
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        if img.mode == 'RGBA' or (img.mode == 'P' and 'transparency' in img.info):
            # Create a transparent background instead of white
            background = Image.new('RGBA', (img.width, img.height), (255, 255, 255, 0))
        else:
            # For non-transparent images, create a white background
            background = Image.new('RGB', (img.width, img.height), (255, 255, 255))
        
        upper_left_x = (background.width - resized_img.width) // 2
        upper_left_y = (background.height - resized_img.height) // 2
        
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background.paste(resized_img, (upper_left_x, upper_left_y), resized_img)
        else:
            background.paste(resized_img, (upper_left_x, upper_left_y))
        
        return background

# Change colour of the image
def colour_change(img_path, colour_name):
    colour_map = {
        'Red': (255, 0, 0, 200),  
        'Yellow': (255, 255, 0, 200),
        'Green': (0, 128, 0, 200), 
        'Blue': (0, 0, 255, 200),
    } 
    if colour_name != "Grey":
        colour = colour_map.get(colour_name) 
    else:
        colour = "Grey"           
    if colour is None:
        raise ValueError(f"Color '{colour_name}' is not one of the defined colors.")
    
    with Image.open(img_path).convert("RGBA") as img:
        width, height = img.size

        result_img = Image.new('RGBA', img.size)
        pixels = result_img.load()

        for x in range(width):
            for y in range(height):
                original_pixel = img.getpixel((x, y))
                if original_pixel[3] > 0: 
                    if colour != "Grey":
                        new_colour = (
                            int((original_pixel[0] + colour[0]) / 2),
                            int((original_pixel[1] + colour[1]) / 2),
                            int((original_pixel[2] + colour[2]) / 2),
                            original_pixel[3],
                        )
                    else:
                        luminance = int(original_pixel[0] * 0.299 + original_pixel[1] * 0.587 + original_pixel[2] * 0.114)
                        new_colour = (
                            luminance,
                            luminance,
                            luminance,
                            original_pixel[3],
                        )
                    pixels[x, y] = new_colour
                else:
                    pixels[x, y] = original_pixel
    return result_img

# Add angles incl +/- operations
def add_angles(x, y):
    if x.startswith('+'):
        angle_x = int(x[1:]) 
    elif x.startswith('-'):
        angle_x = -int(x[1:])
    else:
        angle_x = int(x)  
    
    if y.startswith('+'):
        angle_y = int(y[1:]) 
    elif y.startswith('-'):
        angle_y = -int(y[1:])
    else:
        angle_y = int(y)  
    
    result = angle_x + angle_y
    
    if result == 180:
        result_str = str(result)
    elif result == -180:
        result_str = str(result)[1:]
    else:
        result_str = f"+{result}" if result > 0 else f"{result}"
    
    return result_str

# Generate math operations 
def count_builder(oper):
    if oper not in operations:
        raise ValueError("Invalid Operation")
    num = int(oper[1:])
    math_operations = {
        '+': operator.add,
        '-': operator.sub,
        'x': operator.mul,
        'd': operator.truediv
    }
    if oper.startswith('x') and num == 2:
        starting = [2, 3, 4]  # CHANGED
        mc_op = math_operations['+']
        mc_num = 1
        math_op = math_operations['x']
    elif oper.startswith('x') and num == 3:
        starting = [1, 2, 3]  # CHANGED
        mc_op = math_operations['+']
        mc_num = 1
        math_op = math_operations['x']
    elif oper.startswith('+'):
        starting = [2, 3, 4]
        mc_op = math_operations['-']
        mc_num = 1
        math_op = math_operations['+']
    elif oper.startswith('-'):
        starting = [3, 4, 5]
        mc_op = math_operations['+']
        mc_num = 1
        math_op = math_operations['-']
    else:
        mc_op = math_operations['-']
        mc_num = 1
        math_op = math_operations['d']
        if num % 2 == 0:
            starting = [8, 6, 4]
        else:
            starting = [9, 6, 3]  # CHANGED

    train = random.sample(starting, k = 1)[0]
    test = train
    while test == train: # Resample test until it is different from train
        test = random.sample(starting, k=1)[0]
    train_out = math_op(train, num)
    test_mc_0 = math_op(test, num)
    test_mc_1 = mc_op(test,mc_num)
    save_txt(train,train_out,test, test_mc_1)
    return (train,train_out,test,test_mc_0,test_mc_1) 

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

# Build image with num amount of elements  
def count_generator_old(img_path, num):
    num = int(num)
    uncropped = Image.open(img_path)
    img = crop(uncropped)

    max_items_per_row = 2

    num_rows = (num + max_items_per_row - 1) // max_items_per_row

    used_width = img.width * min(num, max_items_per_row)
    used_height = img.height * num_rows

    canvas_width = int(img.width * 3)
    canvas_height = int(img.height * 3)
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

    start_x = (canvas_width - used_width) // 2
    start_y = (canvas_height - used_height) // 2

    for i in range(num):
        x = start_x + (i % max_items_per_row) * img.width
        y = start_y + (i // max_items_per_row) * img.height
        canvas.paste(img, (x, y))

    return canvas

# Save images generated from the Counting transformation
def transform_save_count(index, param, next_index):
    results = count_builder(param)
    images = (
        train[index],train[index],
        test[index],test[index],test[index]
    )
    inputs = [None]*5
    for i in range(5):
        inputs[i] = count_generator(images[i],results[i])
    save_image(inputs, "Counting", param, index + next_index, suffixes)

def transform_save_colour(index, param, inputs, train_input_color, test_input_color, next_index):
    processed = []
    images = (train[index],test[index])
    for img in images:
        processed.append(colour_change(img,param))
    save_inputs = (
        colour_change(train[index], train_input_color),processed[0],colour_change(test[index], test_input_color),processed[1],colour_change(test[index],inputs)
    )
    save_image(save_inputs,"Colour", param, index+next_index, suffixes)
    save_txt(train_input_color, param, test_input_color, inputs)

def transform_save_resize(index, param, inputs, next_index):
    processed_1_in = resize_indiv(train[index], param, original=True)
    processed_1 = resize_indiv(train[index], param)
    processed_test = resize_indiv(test[index], param, original=True)
    processed_mc_0 = resize_indiv(test[index], param)
    processed_mc_1 = resize_indiv(test[index], factor_mc_1)
    save_inputs = (
        processed_1_in, processed_1,  
        processed_test, processed_mc_0, processed_mc_1
        )
    save_image(save_inputs, "Resize", param, index+next_index, suffixes)
    save_txt("0",param,"0",factor_mc_1)

# Save images generated from the Reflect transformation
def transform_save_reflect(index, param, false_axis, next_index):
    processed_inputs = []
    images = (
        train[index],
        test[index],test[index]
    )
    input_params = (
        param, param, false_axis
    )
    for img,input_param in zip(images,input_params):
        processed_inputs.append(reflect_image(img,input_param))
    save_inputs = (
        train[index], processed_inputs[0], 
        test[index], processed_inputs[1], processed_inputs[2]
        )
    save_image(save_inputs, "Reflect", param, index+next_index, suffixes)
    save_txt("0",param,"0",false_axis)

# Save images generated from the Rotate transformation
def transform_save_rotate(index, param, inputs, input_train_angle, input_test_angle, next_index):
    save_inputs = []


    param_train_rotate = add_angles(input_train_angle, param)
    param_test_rotate = add_angles(input_test_angle, param)
    mc_test_rotate = add_angles(input_test_angle, inputs)

    images = [train[index], train[index], test[index], test[index], test[index]]
    input_params = [input_train_angle, param_train_rotate, input_test_angle, param_test_rotate, mc_test_rotate]

    # Apply rotations and store results
    for img, input_param in zip(images, input_params):
        rotated_image = rotate_image(img, input_param)
        save_inputs.append(rotated_image)

    # Save processed images
    save_image(save_inputs, "2DRotation", param, index + next_index, suffixes)
    save_txt(input_train_angle, param_train_rotate, input_test_angle, inputs)

if transformation not in transformation:
    raise ValueError(f"Transformation not found")

elif transformation == "Reflect":
    axis = parameter
    for i in range(sublist_len):
        reflect_angles = ("X", "Y", "XY")
        false_axis = selector(axis, reflect_angles, 1)[0]
        transform_save_reflect(i, axis, false_axis, next_index)
        
elif transformation == "Colour":
    colour = parameter  
    colours = ["".join(color) for color in colours]
    for i in range(sublist_len):
        colour_mc_1 = selector(colour, colours, 1)[0] 
        train_input_colour = selector([colour, colour_mc_1], colours, 1)[0] 
        test_input_colour = selector([colour, colour_mc_1, train_input_colour], colours, 1)[0] 
        transform_save_colour(i, colour, colour_mc_1, train_input_colour, test_input_colour, next_index)

elif transformation == "Resize":
    factor = parameter
    
    for i in range(sublist_len):
        pairs = {
        "0.5X": "0.5Y",
        "0.5Y": "0.5X",
        "2X": "2Y",
        "2Y": "2X",
        "0.5XY": "2XY",
        "2XY": "0.5XY"
        }
        factor_mc_1 = pairs.get(factor, "No matching factor found")
        transform_save_resize(i, factor, factor_mc_1, next_index)
            
elif transformation == "2DRotation":
    angle = parameter
    for i in range(sublist_len):
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
            angle_mc_1 = random.choice(matches[angle])
        input_train_angle = random.choice(["+45", "-45", "+90", "-90", "+135", "-135"])
        input_test_angle = random.choice(["+45", "-45", "+90", "-90", "+135", "-135"])
        transform_save_rotate(i, angle, angle_mc_1, input_train_angle, input_test_angle, next_index)
        
else:
    operation = parameter
    for i in range(sublist_len):
        transform_save_count(i, operation, next_index)
    