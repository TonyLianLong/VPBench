from openai import OpenAI
import os
import time
import json
import traceback
from .utils import encode_image

base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')

if "openrouter.ai" in base_url:
    api_key = os.environ.get('OPENROUTER_API_KEY')
else:
    api_key = None
if api_key is None:
    api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    if "openrouter.ai" in base_url:
        api_key_path = os.environ.get('OPENROUTER_API_KEY_PATH')
    else:
        api_key_path = os.environ.get('OPENAI_API_KEY_PATH')

    if api_key_path and os.path.exists(api_key_path):
        with open(api_key_path, 'r') as file:
            api_key = file.read().strip()

print(f"Using base URL: {base_url}")

client = OpenAI(api_key=api_key, base_url=base_url, timeout=1200)

def query_openai(image_paths, prompt, model_name="gpt-4o", retry=10, *, image_first, compression_quality=None, provider_only=None):
    """
    Query OpenAI API compatible models with the prompt and a list of image paths or PIL images.
    Temperature is set to 0.0 and retry is set to 10 by default.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    - model_name: String, the model name (default: "gpt-4o")
    - retry: Integer, the number of retries (default: 10).
    - image_first: Boolean, whether to place images before text in the prompt
    - compression_quality: int, JPEG compression quality (0-100). If None, uses PNG format.
    - provider_only: Iterable of provider identifiers to force when using OpenRouter-compatible APIs.
    """
    base64_images = [encode_image(image_path, compression_quality) for image_path in image_paths]
    
    input_dict_text = [{"type": "text", "text": prompt}]
    input_dicts_images = []
    for image in base64_images:
        # Determine MIME type based on compression
        mime_type = "image/jpeg" if compression_quality is not None and compression_quality < 100 else "image/png"
        input_dicts_images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image}",
                "detail": "low"
            }
        })
    
    if image_first:
        input_dicts = input_dicts_images + input_dict_text
    else:
        input_dicts = input_dict_text + input_dicts_images

    provider_list = None
    if provider_only:
        if isinstance(provider_only, (list, tuple, set)):
            provider_list = list(provider_only)
        else:
            provider_list = [provider_only]

    for r in range(retry):
        try:
            extra_body = {}
            if provider_list:
                extra_body["provider"] = {"only": provider_list}

            if "gpt-5.1" in model_name:
                kwargs = dict(
                    model=model_name.removesuffix("-fp32").removesuffix("-fp16"),
                    messages=[
                        {
                            "role": "user",
                            "content": input_dicts,
                        }
                    ],
                    max_completion_tokens=1024,
                    n=1,
                    temperature=0.0,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = client.chat.completions.create(**kwargs)
            elif "gpt-5" in model_name or "o4" in model_name:
                extra_body.update({
                    "reasoning_effort": "minimal",
                    "verbosity": "low",
                })
                response = client.chat.completions.create(
                    model=model_name.removesuffix("-fp32").removesuffix("-fp16"),
                    messages=[
                        {
                            "role": "user",
                            "content": input_dicts,
                        }
                    ],
                    max_completion_tokens=4096,
                    n=1,
                    extra_body=extra_body or None
                )
            else:
                kwargs = dict(
                    model=model_name.removesuffix("-fp32").removesuffix("-fp16"),
                    messages=[
                        {
                            "role": "user",
                            "content": input_dicts,
                        }
                    ],
                    max_completion_tokens=1024,
                    n=1,
                    temperature=0.0,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            # print("Text:", text)
            # dump the input messages to msg.json
            # import json
            # with open("msg.json", "w") as f:
            #     json.dump(input_dicts, f)
            if isinstance(text, str):
                return text
            else:
                assert len(text) == 1 and text[0]['type'] == 'text'
                return text[0]['text']
        except Exception as e:
            print(f"Error occurred (attempt {r+1}/{retry}): {e}")
            print(traceback.format_exc())
            time.sleep(1)
    return 'Failed: OpenAI API Query Error'
