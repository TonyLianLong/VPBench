import json
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from functools import partial
import inspect
from evaluation.models import query_openai
from evaluation.styles import MarkerStyle, MarkerType
from evaluation.config import DATASET_CONFIG, MODEL_CONFIG, get_data_root, get_images_dir
import io
import ast
import cv2
import numpy as np
import time
from openai import OpenAI


# ============================================================================
# Multiple Choice Matching Functions
# ============================================================================

# Initialize OpenAI client for multiple choice matching
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    api_key_path = os.environ.get('OPENAI_API_KEY_PATH')
    if api_key_path and os.path.exists(api_key_path):
        with open(api_key_path, 'r') as file:
            api_key = file.read().strip()

_mc_client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1") if api_key else None
_mc_model_name = 'gpt-3.5-turbo'


def build_prompt(question, options, prediction, valid_choices):
    """
    Builds the prompt for the GPT-3.5 turbo model to match an answer with several options of a single-choice question.

    If the GPT-3.5 model is unable to find a match, it will output (Z).
    Also, if the original prediction does not clearly lean towards any of the options, it will output (Z).

    Parameters:
    - question: String, the question.
    - options: String, the options. E.g. ['(A)', '(B)']
    - prediction: String, the answer. E.g. '(B)'
    """
    if not valid_choices:
        valid_choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
    valid_choices_str = ", ".join(valid_choices)
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (Z)"
        "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (Z)"\
        "Your should output one of the choices, {} (if they are valid options), or (Z)\n"
        "Example 1: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n"
        "Example 2: \n"
        "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        "Example 3: \n"
        "Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n(B) Point B is at the bottom of the spoon, which is not used for poking.\n(C) Point C is on the side of the pspoonot, which is not used for poking.\n(D) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n"
        "Example 4: \n"
        "Question: {}\nOptions:\n{}\n(Z) Failed\nAnswer: {}\nYour output: "
    )
    return tmpl.format(valid_choices_str, question, options, prediction)


def match_multiple_choice(question, options, prediction, valid_choices):
    """Match a prediction to the most similar multiple choice option."""
    if _mc_client is None:
        return '(Z) No API key available'

    prompt = build_prompt(question, options, prediction, valid_choices)
    retry_limit = 10

    for retry in range(retry_limit):
        try:
            response = _mc_client.chat.completions.create(
                model=_mc_model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(1)
    return '(Z) Failed to get multiple choice'


# ============================================================================
# Image Annotation Functions
# ============================================================================

def load_depth_data(split='val', data_root=None):
    """Load depth perception data from JSON"""
    assert split in ['val', 'test']
    if data_root:
        json_path = os.path.join(data_root, 'perceptioneval_depth_dev.json') if split == 'val' else os.path.join(data_root, 'perceptioneval_depth_test.json')
    else:
        depth_config = DATASET_CONFIG['RELATIVE_DEPTH']
        json_file = depth_config['dev_json'] if split == 'val' else depth_config['test_json']
        json_path = os.path.join(depth_config['data_root'], json_file)
    with open(json_path, 'r') as f:
        return json.load(f)


def load_sem_corr_data(data_root, split='val'):
    """Load semantic correspondence data from JSON"""
    assert split in ['val', 'test']

    with open(os.path.join(data_root, 'semantic_corr_raw_fixed.json'), 'r') as f:
        data = json.load(f)
    return data[split]


def draw_markers(image, label_dict, style_settings=None):
    """
    Draw markers and labels on image using OpenCV

    Args:
        image: numpy array or path to image
        label_dict: Dictionary containing coordinates for labels {label_id: (x, y)}
        style_settings: List of style dicts or single style dict for all markers
    Returns:
        numpy array: Annotated image
    """
    # Handle image input
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Make a copy to avoid modifying original
    img_draw = image.copy()

    # Default style settings
    default_style_dict = dict(
        marker_type=MarkerType.CIRCLE,
        color=(255, 0, 0),  # Red in RGB
        radius=5,
        thickness=2,
        show_caption=True,
        font_scale=0.5,
        text_offset=(0, -25),
        text_bg_color=(0, 0, 0),
        text_color=(255, 255, 255),
        text_label_override=None,
    )

    # Handle style settings
    if style_settings is None:
        style_settings = [MarkerStyle(**default_style_dict) for _ in range(len(label_dict))]
    elif isinstance(style_settings, dict):
        style_settings = [MarkerStyle(**style_settings) for _ in range(len(label_dict))]
    elif isinstance(style_settings, list):
        if len(style_settings) == 1:
            if isinstance(style_settings[0], dict):
                style_settings = [MarkerStyle(**style_settings[0])]
            style_settings = style_settings * len(label_dict)
        elif len(style_settings) != len(label_dict):
            raise ValueError(f"Invalid style settings length: {len(style_settings)}")
    elif isinstance(style_settings, MarkerStyle):
        style_settings = [style_settings] * len(label_dict)
    else:
        raise ValueError(f"Invalid style settings type: {type(style_settings)}")

    # Draw markers and labels
    for (label_id, (x, y)), current_style in zip(label_dict.items(), style_settings):
        assert isinstance(current_style, MarkerStyle), f"Style settings must be a MarkerStyle object, got {type(current_style)}"

        x, y = int(x), int(y)

        # Convert RGB to BGR for OpenCV
        bgr_color = current_style.color[::-1]
        bgr_text_bg_color = current_style.text_bg_color[::-1]
        bgr_text_color = current_style.text_color[::-1]

        # Draw marker based on type
        if current_style.marker_type == MarkerType.NONE:
            pass
        elif current_style.marker_type == MarkerType.DOT:
            cv2.circle(img_draw, (x, y), current_style.radius, bgr_color, -1)
        elif current_style.marker_type == MarkerType.CIRCLE:
            cv2.circle(img_draw, (x, y), current_style.radius, bgr_color, current_style.thickness)
        elif current_style.marker_type == MarkerType.SQUARE:
            half_size = current_style.radius
            pt1 = (x - half_size, y - half_size)
            pt2 = (x + half_size, y + half_size)
            cv2.rectangle(img_draw, pt1, pt2, bgr_color, current_style.thickness)
        elif current_style.marker_type == MarkerType.TRIANGLE:
            size = current_style.radius
            points = np.array([
                [x, y - size],
                [x - size, y + size],
                [x + size, y + size]
            ], np.int32)
            cv2.polylines(img_draw, [points], True, bgr_color, current_style.thickness)
            if current_style.thickness == -1:
                cv2.fillPoly(img_draw, [points], bgr_color)
        elif current_style.marker_type == MarkerType.DIAMOND:
            size = current_style.radius
            points = np.array([
                [x, y - size],
                [x + size, y],
                [x, y + size],
                [x - size, y]
            ], np.int32)
            cv2.polylines(img_draw, [points], True, bgr_color, current_style.thickness)
            if current_style.thickness == -1:
                cv2.fillPoly(img_draw, [points], bgr_color)
        elif current_style.marker_type == MarkerType.CROSS:
            size = current_style.radius
            cv2.line(img_draw, (x - size, y), (x + size, y), bgr_color, current_style.thickness)
            cv2.line(img_draw, (x, y - size), (x, y + size), bgr_color, current_style.thickness)
        else:
            raise ValueError(f"Invalid marker type: {current_style.marker_type}")

        # Add caption if enabled
        if current_style.show_caption:
            if current_style.text_label_override is not None:
                label = current_style.text_label_override
            else:
                label = str(label_id)

            # Select font based on bold/italic settings
            if current_style.font_bold and current_style.font_italic:
                font = cv2.FONT_HERSHEY_DUPLEX
            elif current_style.font_bold:
                font = cv2.FONT_HERSHEY_SIMPLEX
            elif current_style.font_italic:
                font = cv2.FONT_ITALIC
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX

            # Adjust thickness for bold text
            text_thickness = 2 if current_style.font_bold else 1

            text_size = cv2.getTextSize(label, font, current_style.font_scale, text_thickness)[0]

            # Position text using offset
            text_x = x - text_size[0] // 2 + current_style.text_offset[0]
            text_y = y + current_style.text_offset[1]

            # Ensure text stays within image bounds
            if text_y < 0:
                text_y = y + current_style.radius + 5

            # Draw text background
            padding = 2
            cv2.rectangle(img_draw,
                         (text_x - padding, text_y - padding),
                         (text_x + text_size[0], text_y + text_size[1] + padding),
                         bgr_text_bg_color,
                         -1)

            # Draw text with updated font settings
            cv2.putText(img_draw, label, (text_x, text_y + text_size[1]),
                       font, current_style.font_scale,
                       bgr_text_color, text_thickness)

    return img_draw


def annotate_depth_image(image_path, point_a, point_b, style_settings=None):
    """Wrapper function for depth perception task"""
    label_dict = {'A': point_a, 'B': point_b}

    # Draw markers and return as PIL Image
    annotated_img = draw_markers(image_path, label_dict, style_settings)
    return Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))


def annotate_sem_corr_image(image_path1, image_path2, point_ref, point_a, point_b, point_c, point_d, style_settings=None):
    """Wrapper function for semantic correspondence task"""
    label_dict_img1 = {'REF': point_ref}
    label_dict_img2 = {'A': point_a, 'B': point_b, 'C': point_c, 'D': point_d}

    # Draw markers and return as PIL Image
    if style_settings is None:
        style_settings1, style_settings2 = None, None
    elif isinstance(style_settings, dict):
        style_settings1, style_settings2 = style_settings, style_settings
    elif isinstance(style_settings, list):
        assert len(style_settings) == 2, f"Invalid style settings length: {len(style_settings)}"
        style_settings1, style_settings2 = style_settings[0], style_settings[1]
    else:
        raise ValueError(f"Invalid style settings type: {type(style_settings)}")

    annotated_img1 = draw_markers(image_path1, label_dict_img1, style_settings1)
    annotated_img2 = draw_markers(image_path2, label_dict_img2, style_settings2)
    return Image.fromarray(cv2.cvtColor(annotated_img1, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))


def get_image_path(data_root, img_str):
    """Get image path from data root and image string identifier."""
    parts = img_str.split('-')
    if img_str.endswith("1"):
        image_id = parts[1]
    elif img_str.endswith("2"):
        image_id = parts[2].split(':')[0]
    else:
        image_id = parts[1]
    category = parts[2].split(':')[1].split('_')[0]
    file_path = os.path.join(data_root, 'images', category, f'{image_id}.jpg')
    return file_path


def get_annotated_image_for_sem_corr(sample, data_root, style_settings=None, annotate_ratio=2):
    """Get annotated images for semantic correspondence task."""
    image_path1 = get_image_path(data_root, sample['images'][0])
    image_path2 = get_image_path(data_root, sample['images'][1])

    orig_img1 = Image.open(image_path1)
    orig_img2 = Image.open(image_path2)
    orig_size1 = orig_img1.size
    orig_size2 = orig_img2.size

    blink_size1 = sample["blink_image1_size"]
    blink_size2 = sample["blink_image2_size"]

    blink_size1 = (blink_size1[0] // annotate_ratio, blink_size1[1] // annotate_ratio)
    blink_size2 = (blink_size2[0] // annotate_ratio, blink_size2[1] // annotate_ratio)

    scale_x1 = blink_size1[0] / orig_size1[0]
    scale_y1 = blink_size1[1] / orig_size1[1]
    scale_x2 = blink_size2[0] / orig_size2[0]
    scale_y2 = blink_size2[1] / orig_size2[1]

    gold_point = sample["gold_point"]
    consider_points = sample["real_consider_points"]
    point_ref = [sample["src_kps"][gold_point][0] * scale_x1,
                sample["src_kps"][gold_point][1] * scale_y1]
    point_a = [sample["trg_kps"][consider_points[0]][0] * scale_x2,
            sample["trg_kps"][consider_points[0]][1] * scale_y2]
    point_b = [sample["trg_kps"][consider_points[1]][0] * scale_x2,
            sample["trg_kps"][consider_points[1]][1] * scale_y2]
    point_c = [sample["trg_kps"][consider_points[2]][0] * scale_x2,
            sample["trg_kps"][consider_points[2]][1] * scale_y2]
    point_d = [sample["trg_kps"][consider_points[3]][0] * scale_x2,
            sample["trg_kps"][consider_points[3]][1] * scale_y2]

    image1 = orig_img1.resize(blink_size1, Image.Resampling.LANCZOS)
    image2 = orig_img2.resize(blink_size2, Image.Resampling.LANCZOS)

    image1, image2 = annotate_sem_corr_image(image1, image2, point_ref, point_a, point_b, point_c, point_d, style_settings=style_settings)

    return image1, image2


# ============================================================================
# Main Benchmark Functions
# ============================================================================

def get_dataset_path_by_type(task_name, dataset_type=None):
    """
    Get dataset path based on task and dataset type.
    
    Args:
        task_name (str): Name of the task ('Relative_Depth' or 'Semantic_Correspondence')
        dataset_type (str, optional): Type of dataset ('BLINK', 'DA-2K', 'SPair-71k')
        
    Returns:
        str: Path to dataset directory
    """
    # Define dataset paths mapping (relative to release directory)
    dataset_paths = {
        'Relative_Depth': {
            'BLINK': './Dataset/BLINK_depth_data/depth',
            'DA-2K': './Dataset/DA-2K_data/depth',
            'default': 'BLINK'
        },
        'Semantic_Correspondence': {
            'BLINK': './Dataset/BLINK_semantic_data/semantic_correspondence',
            'SPair-71k': './Dataset/SPair-71k_data/semantic_correspondence',
            'default': 'BLINK'
        }
    }
    
    if task_name not in dataset_paths:
        # Fall back to original config
        return get_data_root(task_name, None)
    
    task_config = dataset_paths[task_name]
    
    if dataset_type and dataset_type in task_config:
        return task_config[dataset_type]
    else:
        # Use default dataset for task
        default_dataset = task_config.get('default', 'BLINK')
        return task_config.get(default_dataset)

def get_dataset_images_dir_by_type(task_name, dataset_type=None, custom_root=None):
    """
    Get images directory based on task and dataset type.
    
    Args:
        task_name (str): Name of the task
        dataset_type (str, optional): Type of dataset
        custom_root (str, optional): Custom root directory
        
    Returns:
        str: Path to images directory  
    """
    if custom_root:
        return os.path.join(custom_root, 'images')
    
    data_root = get_dataset_path_by_type(task_name, dataset_type)
    return os.path.join(data_root, 'images')

model_generate_funcs = {
    "gpt-4o": query_openai,
}

DEFAULT_CHECKPOINT_INTERVAL = 25

disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"

def compress_image_if_needed(image, max_width=512, target_width=512, jpeg_quality=100):
    """
    Apply JPEG compression if jpeg_quality < 100.
    
    Args:
        image: PIL Image object
        max_width: Maximum width threshold for resizing (unused)
        target_width: Target width for resizing (unused)
        jpeg_quality: JPEG compression quality (0-100)
        
    Returns:
        PIL Image object (compressed if needed)
    """
    if not isinstance(image, Image.Image):
        return image
    
    # Apply JPEG compression only if quality < 100
    if jpeg_quality < 100:
        buffer = io.BytesIO()
        # Use JPEG format for compression
        image.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer)
    
    # If jpeg_quality == 100, return image unchanged (no compression)
    return image

def analyze_answer(d, gpt_answer, all_choices):
    """
    extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

    Parameters:
    - d : data, the data containing the question and choices.
    - gpt_answer: String, the model output.
    - all_choices: List of strings, the list of all choices.

    Returns:
    - prediction, the extracted answer.
    """
    try:
        intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
        intersect_last = list(set(all_choices).intersection(set(gpt_answer.split('\n\n')[-1].split())))
        if gpt_answer in ["A", "B", "C", "D", "E"]:
            prediction = "(" + gpt_answer + ")"
        elif gpt_answer in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            prediction = gpt_answer
        elif (len(intersect) != 1 and len(intersect_last) != 1) or len(intersect) < 1:
            choice_labels = list(all_choices) if all_choices else []
            total_choices = len(d.get('choices', []))
            if not choice_labels or len(choice_labels) != total_choices:
                choice_labels = [f'({chr(ord("A") + i)})' for i in range(total_choices)]
            option_lines = []
            for idx in range(total_choices):
                label = choice_labels[idx] if idx < len(choice_labels) else f'({chr(ord("A") + idx)})'
                option_lines.append(f'{label} {d["choices"][idx]}')
            options = '\n'.join(option_lines)
            extracted_answer = match_multiple_choice(
                f"{d['question']}\nSelect from the following choices",
                options,
                gpt_answer,
                choice_labels
            )
            prediction = extracted_answer
        else:
            if len(intersect_last) == 1:
                intersect = intersect_last
                gpt_answer = gpt_answer.split('\n\n')[-1]
            prediction = intersect[0]
        return prediction
    except Exception as e:
        print(d, gpt_answer, all_choices, e)
        raise e


def _supports_kwarg(func, kwarg):
    """Return True if func accepts the given keyword argument."""
    target = func
    while isinstance(target, partial):
        target = target.func
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return True
    return kwarg in sig.parameters


def _load_existing_results(base_path):
    candidates = [f"{base_path}.tmp", base_path]
    for path in candidates:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            print(f"Warning: could not parse existing results at {path}: {exc}")
            continue

        if not isinstance(data, dict):
            continue

        results = {}
        for split in ['val', 'test']:
            entries = data.get(split, [])
            results[split] = entries if isinstance(entries, list) else []
        return results

    return {'val': [], 'test': []}


def _build_idx_map(entries):
    idx_map = {}
    for entry in entries:
        idx = entry.get('idx') if isinstance(entry, dict) else None
        if idx is None:
            continue
        # Skip entries with (Z) predictions when resuming
        prediction = entry.get('prediction', '(Z)')
        if prediction == '(Z)':
            continue
        idx_map[idx] = entry
    return idx_map


def _serialize_results(idx_maps):
    serialized = {}
    for split, idx_map in idx_maps.items():
        serialized[split] = sorted(idx_map.values(), key=lambda e: e.get('idx'))
    return serialized


def _write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)



def query_model(task_name, model_name, val_only=False, concat_images=False, suffix="", *, image_first, save_annotated_images=False, styles=None, prompt_override=None, original_dataset=False, data_root=None, dataset_type=None, jpeg_quality=100, jpeg_compression=None, resume=False, checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL, debug_run=False, hf_config=None, hf_val_split=None, hf_test_split=None):
    """
    loads the dataset from huggingface or from local json file, query the GPT 4V model with the prompt and images, and saves the result to a json file with specific format.

    Parameters:
    - task_name: String, the name of the task to evaluate.
    - model_name: String, the name of the model to evaluate.
    - val_only: Boolean, whether to only evaluate on validation set.
    - concat_images: Boolean, whether to concatenate the images horizontally.
    - suffix: String, the suffix to add to the output file name.
    - save_annotated_images: Boolean, whether to save annotated images sent to model
    - styles: List of DepthStyle, the styles to apply to the depth images
    - prompt_override: String, the prompt to use for the task
    - data_root: String, the root directory of the dataset
    - dataset_type: String, the type of dataset to use (BLINK, DA-2K, SPair-71k)
    - jpeg_quality: Integer, JPEG compression quality (0-100), 100 means no compression
    - resume: Boolean, whether to resume from existing results
    - checkpoint_interval: Integer, save intermediate progress every N successful samples (0 to disable)

    Returns:
    - outputs, The result is also saved to 'output_filename.json'.
    """
    if dataset_type:
        task_lower = task_name.lower()
        dataset_lower = dataset_type.lower().replace('-', '').replace('spair-71k', 'spair71k')
        base_output_folder = f'outputs_{dataset_lower}_{task_lower}'
    elif task_name == 'Semantic_Correspondence':
        base_output_folder = DATASET_CONFIG['SEMANTIC_OUTPUT_DIR']
    else:
        base_output_folder = output_save_folder

    output_dir = os.path.join(base_output_folder, model_name)
    if suffix:
        output_dir = os.path.join(output_dir, suffix)

    output_path = os.path.join(output_dir, f'{task_name.replace("_", " ")}.json')
    os.makedirs(output_dir, exist_ok=True)

    annotated_dir = None
    if save_annotated_images:
        annotated_dir = os.path.join(output_dir, 'annotated_images')
        os.makedirs(annotated_dir, exist_ok=True)

    split_requests = ['val'] if val_only else ['val', 'test']
    evaluated_splits = list(split_requests)

    if not resume and os.path.exists(output_path):
        print(f"Overwriting existing results at {output_path}")

    idx_maps = {'val': {}, 'test': {}}
    if resume:
        existing = _load_existing_results(output_path)
        total_existing = {'val': 0, 'test': 0}
        for split in ['val', 'test']:
            total_existing[split] = len(existing.get(split, []))
            idx_maps[split] = _build_idx_map(existing.get(split, []))
        if any(idx_maps[split] for split in ['val', 'test']):
            print(
                f"Resuming from {output_path} with "
                f"{len(idx_maps['val'])} val and {len(idx_maps['test'])} test entries "
                f"(filtered out {total_existing['val'] - len(idx_maps['val'])} val and "
                f"{total_existing['test'] - len(idx_maps['test'])} test entries with (Z) predictions)"
            )

    resolved_data_root = data_root
    new_requests_since_save = 0

    max_samples = 10 if debug_run else None

    if not original_dataset:
        for split in split_requests:
            processed_count = 0
            if task_name == 'Relative_Depth':
                if resolved_data_root is None:
                    resolved_data_root = get_dataset_path_by_type(task_name, dataset_type)
                images_dir = get_dataset_images_dir_by_type(task_name, dataset_type, resolved_data_root)
                test_data = load_depth_data(data_root=resolved_data_root, split=split)
                for idx, orig_d in enumerate(tqdm(test_data)):
                    if max_samples and processed_count >= max_samples:
                        break
                    if idx in idx_maps[split]:
                        continue

                    image_path = os.path.join(images_dir, orig_d['original_image'])
                    annotated_image = annotate_depth_image(image_path, orig_d['point_A'], orig_d['point_B'], styles)
                    annotated_image = compress_image_if_needed(annotated_image, jpeg_quality=jpeg_quality)

                    images = [annotated_image]

                    if prompt_override is None:
                        assert 'prompt' in orig_d
                        prompt = orig_d['prompt']
                    else:
                        prompt = prompt_override

                    if annotated_dir:
                        for i, img in enumerate(images):
                            img.save(f"{annotated_dir}/{task_name}_{split}_{idx}.png")

                    gpt_answer = model_generate_func(images, prompt, image_first=image_first, compression_quality=jpeg_compression)
                    gold_answer = f"({orig_d['human_answer_2']})"
                    all_choices = ['(A)', '(B)']

                    prediction = analyze_answer({'question': prompt, 'choices': ['A', 'B']}, gpt_answer, all_choices)

                    idx_maps[split][idx] = {
                        'idx': idx,
                        'answer': gold_answer,
                        'full_prediction': gpt_answer,
                        'prediction': prediction
                    }

                    processed_count += 1
                    new_requests_since_save += 1
                    if checkpoint_interval > 0 and new_requests_since_save >= checkpoint_interval:
                        serialized = _serialize_results(idx_maps)
                        _write_json(output_path, serialized)
                        new_requests_since_save = 0

            elif task_name == 'Semantic_Correspondence':
                if resolved_data_root is None:
                    resolved_data_root = get_dataset_path_by_type(task_name, dataset_type)
                test_data = load_sem_corr_data(data_root=resolved_data_root, split=split)
                for idx, orig_d in enumerate(tqdm(test_data)):
                    if max_samples and processed_count >= max_samples:
                        break
                    if idx in idx_maps[split]:
                        continue

                    image1, image2 = get_annotated_image_for_sem_corr(orig_d, resolved_data_root, styles)

                    image1 = compress_image_if_needed(image1, jpeg_quality=jpeg_quality)
                    image2 = compress_image_if_needed(image2, jpeg_quality=jpeg_quality)

                    images = [image1, image2]

                    if concat_images and len(images) > 1:
                        images = [concat_images_horizontally_with_margin(images, None)]

                    if prompt_override is None:
                        assert 'prompt' in orig_d
                        prompt = orig_d['prompt']
                    else:
                        prompt = prompt_override

                    if annotated_dir:
                        for i, img in enumerate(images):
                            img.save(f"{annotated_dir}/{task_name}_{split}_{orig_d['index']}_{i}.png")

                    gpt_answer = model_generate_func(images, prompt, image_first=image_first, compression_quality=jpeg_compression)
                    gold_answer = orig_d['real_answer']
                    all_choices = ['(A)', '(B)', '(C)', '(D)']
                    prediction = analyze_answer(orig_d, gpt_answer, all_choices)

                    idx_maps[split][idx] = {
                        'idx': idx,
                        'answer': gold_answer,
                        'full_prediction': gpt_answer,
                        'prediction': prediction
                    }

                    processed_count += 1
                    new_requests_since_save += 1
                    if checkpoint_interval > 0 and new_requests_since_save >= checkpoint_interval:
                        serialized = _serialize_results(idx_maps)
                        _write_json(output_path, serialized)
                        new_requests_since_save = 0

            else:
                raise ValueError(f'Task {task_name} is not supported with local json files')
    else:
        print(f"Using original benchmark dataset for task {task_name}")
        config_name = hf_config or task_name
        if hf_val_split == "":
            val_split = None
        elif hf_val_split is None:
            val_split = 'val'
        else:
            val_split = hf_val_split

        if hf_test_split == "":
            test_split = None
        elif hf_test_split is None:
            test_split = 'test'
        else:
            test_split = hf_test_split

        split_name_map = {
            'val': val_split,
            'test': test_split
        }
        active_splits = []
        for split in split_requests:
            target_split = split_name_map.get(split, split)
            if target_split is None:
                continue
            active_splits.append(split)
            processed_count = 0
            try:
                test_data = load_dataset(dataset_name, config_name, split=target_split)
            except ValueError as exc:
                raise ValueError(
                    f'Failed to load Hugging Face split "{target_split}" for config "{config_name}". '
                    'Adjust --hf-val-split/--hf-test-split or verify the config name.'
                ) from exc
            for orig_d in tqdm(test_data):
                if max_samples and processed_count >= max_samples:
                    break
                entry_idx = orig_d.get('idx', orig_d.get('id'))
                if entry_idx is None:
                    raise KeyError("Each sample must include an 'idx' or 'id' field.")
                if entry_idx in idx_maps[split]:
                    continue

                if 'choices' not in orig_d:
                    options = orig_d.get('options')
                    if isinstance(options, str):
                        try:
                            options = ast.literal_eval(options)
                        except (ValueError, SyntaxError):
                            pass
                    if isinstance(options, list):
                        orig_d['choices'] = options

                gold_answer = orig_d['answer']
                if isinstance(gold_answer, str) and not gold_answer.startswith('(') and len(gold_answer) == 1 and gold_answer.isalpha():
                    gold_answer = f'({gold_answer})'
                choice_count = len(orig_d.get('choices', []))
                all_choices = [f'({chr(ord("A") + i)})' for i in range(choice_count)] if choice_count else []
                images, prompt = load_prompt(task_name, orig_d, None, concat_images)

                if prompt_override is not None:
                    prompt = prompt_override

                if annotated_dir:
                    for i, img in enumerate(images):
                        img.save(f"{annotated_dir}/{task_name}_{split}_{entry_idx}_{i}.png")

                gpt_answer = model_generate_func(images, prompt, image_first=image_first, compression_quality=jpeg_compression)
                prediction = analyze_answer(orig_d, gpt_answer, all_choices)

                idx_maps[split][entry_idx] = {
                    'idx': entry_idx,
                    'answer': gold_answer,
                    'full_prediction': gpt_answer,
                    'prediction': prediction
                }

                processed_count += 1
                new_requests_since_save += 1
                if checkpoint_interval > 0 and new_requests_since_save >= checkpoint_interval:
                    serialized = _serialize_results(idx_maps)
                    _write_json(output_path, serialized)
                    new_requests_since_save = 0

        if not active_splits:
            raise ValueError(
                "No evaluation splits configured for the original dataset. "
                "Please provide --hf-val-split or --hf-test-split."
            )
        evaluated_splits = active_splits

    serialized_outputs = _serialize_results(idx_maps)
    if new_requests_since_save > 0 or not os.path.exists(output_path):
        _write_json(output_path, serialized_outputs)

    return serialized_outputs, output_path, evaluated_splits

def concat_images_horizontally_with_margin(images, output_filename, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - images: List of PIL images.
    - output_filename: String, the filename to save the concatenated image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - new_image, the concatenated image.
    """
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one
    if output_filename:
        new_image.save(output_filename)  # Save the result
    return new_image

def load_prompt(task_name, d, image_folder, concat_images=False):
    """
    Loads the prompt and images from huggingface data entry, saves the images to a folder, and returns a list of image paths, and the prompt.

    Parameters:
    - task_name: String, the name of the task.
    - d: data entry, the data dictionary containing the prompt and images.
    - image_folder: String, the folder to save the images.
    - concat_images: Boolean, whether to concatenate the images horizontally.
    Returns:
    - images: List of PIL images.
    - prompt: String, the prompt text.
    - d: Dictionary, the data dictionary with the image paths removed.
    """
    images = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in d and d[k]:
            image = d[k]
            images.append(image)
    if concat_images and len(images) > 1:
        images = [concat_images_horizontally_with_margin(images, f'{image_folder}/{d["idx"]}_merged.jpg' if image_folder else None)]
    prompt = d.get('prompt')
    if prompt is None:
        question = d.get('question')
        options = d.get('choices') or d.get('options')
        if question is None:
            raise KeyError("Expected 'prompt' or 'question' field in dataset sample.")
        prompt_parts = [question]
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except (ValueError, SyntaxError):
                pass
        if isinstance(options, list) and options:
            formatted_options = []
            for idx, option in enumerate(options):
                label = f"({chr(ord('A') + idx)})"
                formatted_options.append(f"{label} {option}")
            prompt_parts.append("\n".join(formatted_options))
            if 'choices' not in d:
                d['choices'] = options
        prompt = "\n\n".join(prompt_parts)
    else:
        options_field = d.get('options')
        if isinstance(options_field, str):
            try:
                options_field = ast.literal_eval(options_field)
            except (ValueError, SyntaxError):
                pass
        if 'choices' not in d and isinstance(options_field, list):
            d['choices'] = options_field
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    if 'blip' in model_name:
        prompt += '\nAnswer:'
    return images, prompt


def eval_task(task_name, model_name, val_only=False, concat_images=False, suffix="", *,
              image_first, save_annotated_images=False, styles=None, prompt_override=None, original_dataset=False, data_root=None, dataset_type=None, jpeg_quality=100, jpeg_compression=None, resume=False, checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL, debug_run=False, hf_config=None, hf_val_split=None, hf_test_split=None):
    outputs, output_path, evaluated_splits = query_model(task_name, model_name, val_only, concat_images, suffix,
                                      image_first=image_first, save_annotated_images=save_annotated_images,
                                      styles=styles, prompt_override=prompt_override, original_dataset=original_dataset, data_root=data_root, dataset_type=dataset_type, jpeg_quality=jpeg_quality, jpeg_compression=jpeg_compression,
                                      resume=resume, checkpoint_interval=checkpoint_interval, debug_run=debug_run, hf_config=hf_config, hf_val_split=hf_val_split, hf_test_split=hf_test_split)
    accu = {'val': 0, 'test': 0}
    candidate_splits = evaluated_splits or (['val'] if val_only else ['val', 'test'])
    splits = [split for split in candidate_splits if len(outputs.get(split, [])) > 0]
    if not splits:
        splits = candidate_splits
    for split in splits:
        if split not in accu:
            accu[split] = 0
        for d in outputs[split]:
            if d['answer'] == d['prediction']:
                accu[split] += 1
    
    print('-'*50)
    print(f'Task {task_name} Performance')
    for split in splits:
        print(f'{split} accuracy: {round(accu[split]/len(outputs[split])*100, 2)}%')
    
    # Generate summary.json with total image count and accuracy information
    # Get the output directory from the existing output_path
    output_dir = os.path.dirname(output_path)
    summary_path = f'{output_dir}/summary.json'
    
    # Calculate total images and accuracy for summary
    total_images = 0
    total_correct = 0
    summary_data = {
        "task_name": task_name,
        "model_name": model_name,
        "total_images": 0,
        "overall_accuracy": 0.0,
        "split_details": {}
    }
    
    for split in splits:
        split_total = len(outputs[split])
        split_correct = accu[split]
        split_accuracy = round(split_correct / split_total * 100, 2) if split_total > 0 else 0.0
        
        total_images += split_total
        total_correct += split_correct
        
        summary_data["split_details"][split] = {
            "total_images": split_total,
            "correct_predictions": split_correct,
            "accuracy_percentage": split_accuracy
        }
    
    summary_data["total_images"] = total_images
    summary_data["overall_accuracy"] = round(total_correct / total_images * 100, 2) if total_images > 0 else 0.0
    
    # Save summary.json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print(f'Summary saved to: {summary_path}')


def parse_args():
    # Example: python benchmark_executor.py --model_name gpt-4o --task_name Relative_Depth --val_only --save_annotated_images
    # Example: python benchmark_executor.py --model_name qwen/qwen3-vl-8b-instruct --task_name Relative_Depth --val_only --save_annotated_images
    
    # Example: python benchmark_executor.py --model_name gpt-4o --task_name Relative_Depth --val_only --save_annotated_images --image_first --suffix "image_first"
    # Example: python benchmark_executor.py --model_name qwen/qwen3-vl-8b-instruct --task_name Relative_Depth --val_only --save_annotated_images --image_first --suffix "image_first"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt-4o', help="select the model name")
    parser.add_argument("--task_name", type=str, default='Relative_Depth', help="select the task name")
    parser.add_argument("--val_only", action="store_true", help="only evaluate on validation set")
    parser.add_argument("--concat_images", action="store_true", help="concatenate images horizontally")
    parser.add_argument("--suffix", type=str, default='', help="suffix to add to the output file name")
    parser.add_argument("--image_first", action="store_true", help="whether to place images before text in the prompt")
    parser.add_argument("--save_annotated_images", action="store_true", help="save annotated images sent to model")
    parser.add_argument("--styles", type=str, help="JSON string with two style configurations for points A and B")
    parser.add_argument("--prompt", type=str, help="prompt to use for the task (default: use the prompt from the dataset)", default=None)
    parser.add_argument("--original-dataset", action="store_true", help="use the original dataset from huggingface instead of local json files. This will ignore the style settings.")
    parser.add_argument("--data-root", type=str, help="the root directory of the dataset", default=None)
    parser.add_argument("--dataset-type", type=str, help="specify dataset type (BLINK, DA-2K, SPair-71k)", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=100, help="JPEG compression quality (0-100), 100 means no compression")
    parser.add_argument("--jpeg-compression", type=int, default=None, help="JPEG compression quality for base64 encoding (0-100), None means no compression")
    parser.add_argument("--output_dir", type=str, default="", help="Specify the output directory for results")
    parser.add_argument("--overwrite", action="store_true", help="ignore existing progress and start fresh")
    parser.add_argument("--checkpoint_interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help="save intermediate progress every N successful samples")
    parser.add_argument("--debug_run", action="store_true", help="Only run the first 10 samples per split and append '_debug' to the output suffix")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Override the Hugging Face dataset when using --original-dataset")
    parser.add_argument("--hf-config", type=str, default=None, help="Select the Hugging Face config (e.g., Accounting) when using --original-dataset")
    parser.add_argument("--hf-val-split", type=str, default=None, help="Hugging Face split name for validation (default: val)")
    parser.add_argument("--hf-test-split", type=str, default=None, help="Hugging Face split name for test (default: test)")
    parser.add_argument("--provider-only", type=str, default=None, help="Restrict OpenRouter requests to the given provider identifiers")
    args = parser.parse_args()
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint_interval must be non-negative")
    return args


if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name

    base_generate_func = model_generate_funcs.get(model_name, query_openai)
    generate_kwargs = {"model_name": model_name}
    if args.provider_only and _supports_kwarg(base_generate_func, "provider_only"):
        try:
            args.provider_only = json.loads(args.provider_only)
        except json.JSONDecodeError:
            pass
        generate_kwargs["provider_only"] = args.provider_only
    model_generate_func = partial(base_generate_func, **generate_kwargs)
    
    if args.debug_run:
        base_suffix = args.suffix or ''
        if base_suffix.endswith('_debug'):
            args.suffix = base_suffix
        else:
            args.suffix = f"{base_suffix}_debug" if base_suffix else "_debug"
        print("Debug run enabled: limiting to 10 samples per split.")
    
    print(f'Using model: {model_name}, save suffix: {args.suffix if args.suffix else "no suffix"}')
    
    # image_save_folder = 'saved_images'
    if args.output_dir:
        output_save_folder = args.output_dir
    else:
        output_save_folder = DATASET_CONFIG['DEFAULT_OUTPUT_DIR']
    dataset_name = args.hf_dataset or MODEL_CONFIG['HUGGINGFACE_DATASET']
    

    need_disclaimer_tasks = MODEL_CONFIG['DISCLAIMER_TASKS']
    if args.task_name == 'all': 
        subtasks = MODEL_CONFIG['ALL_TASKS']
    else:
        subtasks = [args.task_name]

    for task_name in subtasks:
        # Parse styles if provided
        styles = None
        if args.styles:
            style_configs = json.loads(args.styles)
            if isinstance(style_configs, list):
                if isinstance(style_configs[0], list):
                    # List of list in terms of two images
                    styles = [[MarkerStyle(**cfg) for cfg in style_config] for style_config in style_configs]
                else:
                    styles = [MarkerStyle(**cfg) for cfg in style_configs]
            else:
                styles = MarkerStyle(**style_configs)
        
        # Set default data_root if not provided, or use dataset_type to determine path
        hf_only_tasks = {'MMMU', 'MMMU_Pro', 'MMMU_Pro_4', 'MMMU_Pro_10'}
        if task_name in hf_only_tasks and args.original_dataset:
            data_root = args.data_root  # Hugging Face datasets load remotely; no local root required.
        else:
            if args.data_root:
                data_root = args.data_root
            elif hasattr(args, 'dataset_type') and args.dataset_type:
                data_root = get_dataset_path_by_type(task_name, args.dataset_type)
            else:
                data_root = get_data_root(task_name, args.data_root)
        
        # Print dataset information
        dataset_info = f"task {task_name}"
        if hasattr(args, 'dataset_type') and args.dataset_type:
            dataset_info += f" with {args.dataset_type} dataset"
        if data_root:
            print(f'Using data root {data_root} for {dataset_info}')
        
        eval_task(task_name, model_name, args.val_only, args.concat_images, args.suffix,
                 image_first=args.image_first, save_annotated_images=args.save_annotated_images,
                 styles=styles, prompt_override=args.prompt, original_dataset=args.original_dataset,
                 data_root=data_root, dataset_type=getattr(args, 'dataset_type', None),
                 jpeg_quality=args.jpeg_quality, jpeg_compression=args.jpeg_compression, resume=not args.overwrite,
                 checkpoint_interval=args.checkpoint_interval, debug_run=args.debug_run, hf_config=args.hf_config,
                 hf_val_split=args.hf_val_split, hf_test_split=args.hf_test_split)
