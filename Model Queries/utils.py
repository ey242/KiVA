import os
from PIL import Image, ImageOps
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np 
import cv2
import os

def read_image(img_path): 
    png_image = Image.open(img_path)
    
    # Check if the image has an alpha channel
    if png_image.mode in ('RGBA', 'LA') or (png_image.mode == 'P' and 'transparency' in png_image.info):
        # Create a new RGB image with the same size and white background
        rgb_image = Image.new("RGB", png_image.size, (255, 255, 255))
        # Paste the PNG image onto the RGB image, using alpha channel as mask
        rgb_image.paste(png_image, mask=png_image.split()[3]) # 3 is the index of alpha channel in 'RGBA'
    else:
        # If no alpha channel, just convert to RGB
        rgb_image = png_image.convert('RGB')

    rgb_image.thumbnail((300, 300))

    return rgb_image

def draw_text(img, text, position=(10, 0), fontsize = 40): 
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", fontsize)
    # font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", fontsize) 

    draw.text(position,text,(0,0,0), font = font)
    return img

def draw_arrow(img, start, end): 
    img = np.array(img)
    img = cv2.arrowedLine(img, start, end, (0,0,0), 5)
    return Image.fromarray(img)

def stitch_images(image1, image2, add_borders=True):
    border_size = 30

    if add_borders:
        new_width = image1.width + image2.width + border_size
    else: 
        new_width = image1.width + image2.width

    new_height = max(image1.height, image2.height)
    
    # Create a new image with a black background
    new_image = Image.new('RGB', (new_width, new_height), color="black")
    
    # Paste image1 and image2 onto the new image
    new_image.paste(image1, (0, (new_height - image1.height) // 2))

    if add_borders:
        new_image.paste(image2, (image1.width + border_size, (new_height - image2.height) // 2))

         # Create a mask for the white border
        border_mask = Image.new('RGB', (border_size, new_height), color="white")
        new_image.paste(border_mask, (image1.width, 0))
    else:
        new_image.paste(image2, (image1.width, (new_height - image2.height) // 2))

    # Add a black border around the entire stitched image
    new_image_with_border = ImageOps.expand(new_image, border=5, fill="black")
    new_image_with_border = draw_arrow(new_image_with_border, (image1.width - 20, new_height//2), (image1.width + border_size + 20, new_height//2))

    return new_image_with_border

def stitch_images_v(images, add_borders=True, color="white", border_size = 40, tight=True):

    # Read image using pillow 
    if add_borders:
        new_height = sum([image.height for image in images]) + (len(images) - 1) * border_size
    else: 
        new_height = sum([image.height for image in images])

    new_width = max([image.width for image in images])

    # Create a new image with a black background
    new_image = Image.new('RGB', (new_width, new_height), color="white")

    y_offset = 0
    for image in images:
        paste_width = (new_width - image.width) // 2
        new_image.paste(image, (paste_width, y_offset))
        y_offset += image.height

        #determine past width so the image is centered with respect to new_width

        if add_borders and y_offset < new_height:

            if tight:
                border_mask = Image.new('RGB', (image.width, border_size), color=color)
                new_image.paste(border_mask, (paste_width, y_offset))

            else: 
                border_mask = Image.new('RGB', (new_width, border_size), color=color)
                new_image.paste(border_mask, (0, y_offset))
            y_offset += border_size
    
    return new_image, new_width


def stitch_images_h1(image1, image2, border_size=10, add_borders=True, background_color="white", border_color="black", create_white_fluff_around_border=False):
    # Calculate the total width with the border included
    total_width = image1.width + image2.width + (border_size if add_borders else 0)
    max_height = max(image1.height, image2.height)
    
    # Create a new image with the specified background color
    new_image = Image.new('RGB', (total_width, max_height), background_color)
    
    # Paste the first image
    new_image.paste(image1, (0, (max_height - image1.height) // 2))
    
    # Calculate the X offset for the second image, which includes the border if needed
    x_offset = image1.width + (border_size if add_borders else 0)
    
    # Paste the second image
    new_image.paste(image2, (x_offset, (max_height - image2.height) // 2))
    
    if add_borders:
        # Create the border
        border = Image.new('RGB', (border_size, max_height), border_color)
        # Paste the border between the images
        new_image.paste(border, (image1.width, 0))
    
    # Optionally, create white fluff around the border
    if create_white_fluff_around_border and add_borders:
        fluff_size = border_size // 10
        fluff = Image.new('RGB', (fluff_size, max_height), "white")
        fluff_offset = image1.width + (border_size - fluff_size) // 2
        new_image.paste(fluff, (fluff_offset, 0))
    
    return new_image


def stitch_images_h(image1, image2, border_size = 10, add_borders=True, background_color="white", border_color="black", create_white_fluff_around_border = False):

    # Read image using pillow 
    if add_borders:
        new_width = image1.width + image2.width + border_size
    else: 
        new_width = image1.width + image2.width

    new_height = max(image1.height, image2.height)

    # Create a new image with a black background
    new_image = Image.new('RGB', (new_width, new_height), color=background_color)
    
    # Paste image1 and image2 onto the new image
    new_image.paste(image1, (0, (new_height - image1.height) // 2))

    if add_borders:
        new_image.paste(image2, (image1.width + border_size, (new_height - image2.height) // 2))

        if create_white_fluff_around_border:
            new_image.paste(Image.new('RGB', (border_size//10, new_height), color=border_color), (image1.width +  (border_size - border_size//10)//2, 0))


        else: 
            new_image.paste(Image.new('RGB', (border_size, new_height), color=border_color), (image1.width, 0))


    else:
        new_image.paste(image2, (image1.width, (new_height - image2.height) // 2))

    return new_image, image1.width + border_size

def stitch_images_train(image1, image2, add_borders=True, case_num=0):
    
    border_size = 10
    initial_stitched_image,_ = stitch_images_h(image1, image2, border_size=border_size, add_borders= add_borders, border_color="black")  

    #create two white images 
    image1 = Image.new('RGB', (image1.width, image2.height), color="white")
    image2 = Image.new('RGB', (image2.width, image2.height), color="white")  

    white_images_stitched,_ = stitch_images_h(image1, image2, border_size=border_size, add_borders= add_borders, border_color="black")
    final_image, _ = stitch_images_v([initial_stitched_image, white_images_stitched], add_borders=add_borders, color="black", border_size=border_size, tight=False)


    return final_image

def stitch_images_test(stitched_images,  add_borders=True):

    border_size = 10
    image1 = stitched_images[0]

    stitched_images_h = []
    options = " ABCD"
    count = 0
    for image in stitched_images[1:]:
        
        count += 1
        #add white border to the bottom of image
        stitched_img,_ = stitch_images_h(image1, image, border_size=border_size, add_borders= add_borders, border_color="black")

        label_space = 80  # Adjust as needed
        stitched_img = ImageOps.expand(stitched_img, border=(label_space, 0, 0, 0), fill="white")
        
        # Draw the text onto the whitespace
        stitched_img = draw_text(stitched_img, f"({options[count]})", (20, 20))
        stitched_images_h.append(stitched_img)

    final_image, _ = stitch_images_v(stitched_images_h, add_borders=add_borders, color="black", border_size=30, tight=False)

    return final_image


def stitch_final_images(train0_image, test_image): 
    border_size = 10
    images = [train0_image, test_image]
    # final_image, _ = stitch_images_v(images, add_borders=True, border_size=10, color="black")
    final_image,_ = stitch_images_h(train0_image, test_image, add_borders=True, border_size=200, border_color ="black", create_white_fluff_around_border=True)

    #create a slightly wider image on both side 
    final_image = ImageOps.expand(final_image, border=10, fill="white")

    return final_image
