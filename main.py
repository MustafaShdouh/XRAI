import os
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HttpOptions,
    SafetySetting,
)
from PIL import Image, ImageDraw, ImageColor, ImageFont
from pydantic import BaseModel

# Define the BoundingBox data model.
# The API is expected to return bounding box coordinates scaled on a 0 to 1000 range,
# along with a diagnosis label.
class BoundingBox(BaseModel):
    box_2d: list[int]  # Format: [y_min, x_min, y_max, x_max]
    label: str

def plot_bounding_boxes(image: Image.Image, bounding_boxes: list[BoundingBox]) -> Image.Image:
    """
    Draws bounding boxes and labels on the given image.
    The bounding box coordinates (0–1000) are scaled to the image size.
    """
    width, height = image.size
    draw = ImageDraw.Draw(image)
    # Load a default font. (Optionally, you can use a custom TTF font)
    font = ImageFont.load_default()
    colors = list(ImageColor.colormap.keys())
    
    for i, bbox in enumerate(bounding_boxes):
        y_min, x_min, y_max, x_max = bbox.box_2d
        # Scale coordinates from 0–1000 to actual pixels.
        abs_y_min = int(y_min / 1000 * height)
        abs_x_min = int(x_min / 1000 * width)
        abs_y_max = int(y_max / 1000 * height)
        abs_x_max = int(x_max / 1000 * width)

        color = colors[i % len(colors)]
        draw.rectangle(((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)), outline=color, width=4)
        draw.text((abs_x_min + 5, abs_y_min - 15), bbox.label, fill=color, font=font)
    
    return image

# Initialize the Google Gemini client.
# (For security, consider using environment variables to store your API key.)
client = genai.Client(api_key="AIzaSyB0JquVfZUaIcBw4PfidEWOeCfs30GTF0A")

# Set up the configuration for the analysis.
config = GenerateContentConfig(
    system_instruction="""
        Analyze this medical X-ray image and identify any visible diseases or abnormalities.
        For each observation, provide a bounding box in the format [y_min, x_min, y_max, x_max]
        scaled to a range of 0 to 1000, along with a label for the diagnosed condition.
        Limit the response to 5 observations.
    """,
    temperature=0.2,
    safety_settings=[
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ],
    response_mime_type="application/json",
    response_schema=list[BoundingBox],
)

# Set the path to the X-ray image.
image_path = os.path.join("Dataset", "IM-0011-0001-0002.jpeg")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Open the image as a PIL.Image instance.
image_instance = Image.open(image_path)

# Send the image + prompt to the Gemini API.
response = client.models.generate_content(
    model="gemini-2.0-flash",  # Use the appropriate model name for your access.
    contents=[
        image_instance,  # PIL image instance is directly accepted.
        "Identify diseases and corresponding bounding boxes in this X-ray."
    ],
    config=config,
)

# Process the API response.
if response.parsed:
    # The response is parsed into a list of BoundingBox objects.
    bounding_boxes = response.parsed
    print("Detection Results:")
    for bbox in bounding_boxes:
        print(f"Disease: {bbox.label}, Coordinates (0-1000 scale): {bbox.box_2d}")

    # Annotate the image with the bounding boxes and labels.
    annotated_image = plot_bounding_boxes(image_instance, bounding_boxes)
    annotated_image.show()  # This opens the default image viewer.
else:
    print("Error with Gemini response:", response.text)
