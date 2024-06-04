from PIL import Image, ImageOps, ImageFont, ImageDraw


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

    return new_image_with_border

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

def stitch_images_train(image1, image2, add_borders=True):
    initial_stitched_image,_ = stitch_images_h(image1, image2, border_size=8, add_borders= add_borders, border_color="black")  

    return initial_stitched_image

def stitch_images_test(stitched_images,  add_borders=True):
    image1 = stitched_images[0]

    stitched_images_h = []
    options = " ABCD"
    count = 0
    for image in stitched_images[1:]:
        count += 1
        stitched_img,_ = stitch_images_h(image1, image, border_size=10, add_borders= add_borders, border_color="black")

        # Add white border to the bottom of image
        stitched_img = ImageOps.expand(stitched_img, border=(0, 0, 0, 100), fill="white")

        stitched_img = draw_text(stitched_img, f"({options[count]})", (int((stitched_img.width - len(options[count]) * 80) / 2) - 10, stitched_img.height - 100), fontsize=80)

        stitched_images_h.append(stitched_img)

    return stitched_images_h
