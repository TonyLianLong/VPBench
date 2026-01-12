import base64
import io

# Function to encode the image
def encode_image(image_input, compression_quality=None):
    """
    Encode the image to base64. Accepts either a file path or a PIL Image object.
    
    Parameters:
    - image_input: String (file path) or PIL.Image object
    - compression_quality: int, JPEG compression quality (0-100). If None, uses PNG format.
    
    Returns:
    - String: base64 encoded image
    """
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        buffered = io.BytesIO()
        if compression_quality is not None and compression_quality < 100:
            # Use JPEG compression with specified quality
            image_input.convert('RGB').save(buffered, format="JPEG", quality=compression_quality)
        else:
            # Use PNG format (no compression)
            image_input.convert('RGB').save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
