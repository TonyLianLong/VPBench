"""
Configuration file for Visual Prompt Bias experiments.
Centralized configuration for dataset paths and other settings.
"""

import os

# Base paths - modify these to match your setup
BASE_DATASET_PATH = './Dataset'

# Dataset configuration
DATASET_CONFIG = {
    'BASE_PATH': BASE_DATASET_PATH,
    
    # Relative Depth task configuration
    'RELATIVE_DEPTH': {
        'data_root': os.path.join(BASE_DATASET_PATH, 'BLINK_depth_data', 'depth'),
        'images_dir': os.path.join(BASE_DATASET_PATH, 'BLINK_depth_data', 'depth', 'images'),
        'dev_json': 'perceptioneval_depth_dev.json',
        'test_json': 'perceptioneval_depth_test.json'
    },
    
    # Semantic Correspondence task configuration  
    'SEMANTIC_CORRESPONDENCE': {
        'data_root': os.path.join(BASE_DATASET_PATH, 'BLINK_semantic_data', 'semantic_correspondence'),
        'images_dir': os.path.join(BASE_DATASET_PATH, 'BLINK_semantic_data', 'semantic_correspondence', 'images'),
        'data_json': 'semantic_corr_raw_fixed.json'
    },
    
    # DA-2K dataset (alternative depth dataset)
    'DA_2K': {
        'data_root': os.path.join(BASE_DATASET_PATH, 'DA-2K_data', 'depth'),
        'images_dir': os.path.join(BASE_DATASET_PATH, 'DA-2K_data', 'depth', 'images'),
        'dev_json': 'perceptioneval_depth_dev.json',
        'test_json': 'perceptioneval_depth_test.json'
    },
    
    # SPair-71k dataset (alternative semantic correspondence)
    'SPAIR_71K': {
        'data_root': os.path.join(BASE_DATASET_PATH, 'SPair-71k_data', 'semantic_correspondence'),
        'images_dir': os.path.join(BASE_DATASET_PATH, 'SPair-71k_data', 'semantic_correspondence', 'images'),
        'data_json': 'semantic_corr_raw_fixed.json'
    },
    
    # Output and visualization paths
    'VISUALIZATION_OUTPUT': os.path.join(BASE_DATASET_PATH, 'visualizations'),
    'DEFAULT_OUTPUT_DIR': 'outputs_relative_depth',
    'SEMANTIC_OUTPUT_DIR': 'outputs_semantic_correspondence'
}

# Model configuration
MODEL_CONFIG = {
    'HUGGINGFACE_DATASET': 'BLINK-Benchmark/BLINK',
    'DISCLAIMER_TASKS': ['Forensic_Detection', 'Jigsaw', 'Art_Style'],
    'ALL_TASKS': [
        'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 
        'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 
        'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 
        'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 
        'Relative_Depth', 'Spatial_Relation'
    ]
}

def get_dataset_config(task_name):
    """
    Get dataset configuration for a specific task.
    
    Args:
        task_name (str): Name of the task ('Relative_Depth' or 'Semantic_Correspondence')
        
    Returns:
        dict: Configuration dictionary for the task
    """
    task_key = task_name.upper()
    if task_key in DATASET_CONFIG:
        return DATASET_CONFIG[task_key]
    else:
        raise ValueError(f"Unknown task: {task_name}")

def get_data_root(task_name, custom_root=None):
    """
    Get data root path for a task.
    
    Args:
        task_name (str): Name of the task
        custom_root (str, optional): Custom root path to override default
        
    Returns:
        str: Path to data root directory
    """
    if custom_root:
        return custom_root
    
    config = get_dataset_config(task_name)
    return config['data_root']

def get_images_dir(task_name, custom_root=None):
    """
    Get images directory path for a task.
    
    Args:
        task_name (str): Name of the task
        custom_root (str, optional): Custom root path to override default
        
    Returns:
        str: Path to images directory
    """
    if custom_root:
        return os.path.join(custom_root, 'images')
    
    config = get_dataset_config(task_name) 
    return config['images_dir']

def get_accuracy_output_dir(task_name):
    """
    Get default output directory for accuracy calculation.
    
    Args:
        task_name (str): Name of the task
        
    Returns:
        str: Default output directory path
    """
    if task_name.lower() == 'semantic_correspondence':
        return DATASET_CONFIG['SEMANTIC_OUTPUT_DIR']
    else:
        return DATASET_CONFIG['DEFAULT_OUTPUT_DIR']

def is_blink_dataset(dataset_type):
    """
    Check if the dataset type is a BLINK dataset that requires filtering.
    
    Args:
        dataset_type (str): Type of dataset
        
    Returns:
        bool: True if it's a BLINK dataset that should be filtered
    """
    blink_types = {"blink", "blink_depth", "blink_semantic"}
    return dataset_type.lower() in blink_types

# Relative path configuration for evalv2 datasets
RELATIVE_DATA_ROOTS = {
    'BLINK': {
        'semantic_correspondence': './Dataset/BLINK_semantic_data/semantic_correspondence',
        'relative_depth': './Dataset/BLINK_depth_data/depth',
    },
    'DA-2K': {
        'relative_depth': './Dataset/DA-2K_data/depth',
    },
    'SPair-71k': {
        'semantic_correspondence': './Dataset/SPair-71k_data/semantic_correspondence',
    }
}

def get_dataset_path_by_type(dataset_type, task_type):
    """
    Get dataset path using relative paths for evalv2 datasets.
    
    Args:
        dataset_type (str): Type of dataset ('BLINK', 'DA-2K', 'SPair-71k')
        task_type (str): Task type ('semantic_correspondence', 'relative_depth')
        
    Returns:
        str: Relative path to dataset directory

    """
    # Normalize dataset type names 
    dataset_mapping = {
        'BLINK': 'BLINK',
        'DA-2K': 'DA-2K', 
        'SPAIR-71K': 'SPair-71k',
        'SPAIR71K': 'SPair-71k'
    }
    
    dataset_type = dataset_type.upper()
    dataset_type = dataset_mapping.get(dataset_type, dataset_type)
    task_type = task_type.lower()
    
    if dataset_type not in RELATIVE_DATA_ROOTS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported: {list(RELATIVE_DATA_ROOTS.keys())}")
    
    dataset_tasks = RELATIVE_DATA_ROOTS[dataset_type]
    if task_type not in dataset_tasks:
        raise ValueError(f"Unsupported task type '{task_type}' for dataset '{dataset_type}'. Available: {list(dataset_tasks.keys())}")
    
    return dataset_tasks[task_type]

def get_task_from_name(task_name):
    """
    Map task names to task types for evalv2.
    
    Args:
        task_name (str): Original task name
        
    Returns:
        str: Standardized task type
    """
    task_mapping = {
        'semantic_correspondence': 'semantic_correspondence',
        'relative_depth': 'relative_depth'
    }
    
    task_key = task_name.lower()
    return task_mapping.get(task_key, task_key)

# ============================================================================
# Run Script Configuration
# ============================================================================

# Data and Output Directory Configuration
DATA_ROOT_CONFIG = {
    "Relative_Depth": {
        "BLINK": "./Dataset/BLINK_depth_data/depth",
        "DA-2K": "./Dataset/DA-2K_data/depth",
        "default": "BLINK"
    },
    "Semantic_Correspondence": {
        "BLINK": "./Dataset/BLINK_semantic_data/semantic_correspondence",
        "SPair-71k": "./Dataset/SPair-71k_data/semantic_correspondence",
        "default": "BLINK"
    }
}

OUTPUT_DIR_CONFIG = {
    "base_dir": "./outputs",
}

# Dataset mapping for different tasks
DATASET_CHOICES = {
    "Relative_Depth": ["BLINK", "DA-2K"],
    "Semantic_Correspondence": ["BLINK", "SPair-71k"]
}

default_style_relative_depth = [
    {
        "marker_type": "circle",
        "font_scale": 0.5,
        "radius": 5,
        "color": [255, 0, 0],
        "text_offset": [0, -25],
    },
    {
        "marker_type": "circle",
        "font_scale": 0.5,
        "radius": 5,
        "color": [255, 0, 0],
        "text_offset": [0, -25],
    }
]

default_style_semantic = [
    [
        {
            "marker_type": "circle",
            "radius": 5,
            "color": [255, 0, 0],
            "text_label_override": "REF"
        }
    ],
    [
        {
            "marker_type": "circle",
            "radius": 5,
            "color": [255, 0, 0],
            "text_label_override": "A"
        },
        {
            "marker_type": "circle",
            "radius": 5,
            "color": [255, 0, 0],
            "text_label_override": "B"
        },
        {
            "marker_type": "circle",
            "radius": 5,
            "color": [255, 0, 0],
            "text_label_override": "C"
        },
        {
            "marker_type": "circle",
            "radius": 5,
            "color": [255, 0, 0],
            "text_label_override": "D"
        }
    ]
]

default_suffix_relative_depth = "image_first_prompt_v2"
default_suffix_semantic = "image_first_prompt_v2"

# Prompts for different configurations
PROMPT_DEFAULT = "Two points are marked by {marker_type}s on the image, labeled by {label_a} and {label_b} beside each {marker_type}. Which point is closer to the camera?\nSelect from the following choices.\n(A) {label_a} is closer\n(B) {label_b} is closer"

PROMPT_NO_CAPTION = "Two points are marked by {marker_type}s on the image. Which point is closer to the camera?\nSelect from the following choices.\n(A) The point marked by the {color_a} {marker_type} is closer\n(B) The point marked by the {color_b} {marker_type} is closer"

PROMPT_NO_MARKER = "Two points are labeled by {label_a} and {label_b}. Which point is closer to the camera?\nSelect from the following choices.\n(A) {label_a} is closer\n(B) {label_b} is closer"

PROMPT_SEMANTIC_CORRESPONDENCE = "Humans can find corresponding points for different objects in the same category. For instance, if there are images of two different cats, then the left ear tip of one cat corresponds to the left ear tip of the other cat, and the right front paw of one cat corresponds to the right front paw of the other cat.\nGiven the following two images, a reference point is annotated on the first image, labeled with REF. You are given multiple {color}-{marker_type_d} points on the second image, choices of \"{label_a}, {label_b}, {label_c}, {label_d}\" are drawn beside each {marker_type}. Select between the choices on the second image and find the corresponding point for the reference point. Which point is corresponding to the reference point?\nSelect from the following choices.\n(A) Point {label_a}\n(B) Point {label_b}\n(C) Point {label_c}\n(D) Point {label_d}\n"

# Define the groups of parameters to be modified
EXPERIMENT_PARAMETERS = [
    {
        'name': 'radius',
        'values': [1, 3, 10, 15],
        'handler': lambda val: ({'radius': val}, f"radius_{val}")
    },
    {
        'name': 'marker_type',
        'values': ['square', 'diamond', 'triangle'],
        'handler': lambda val: ({'marker_type': val}, f"marker_type_{val}")
    },
    {
        'name': 'font_scale',
        'values': [0.2, 1.0],
        'handler': lambda val: (
            {'font_scale': val,'text_offset': [0, -35] if val == 1.0 else [0, -17]},f"font_scale_{val}"
        )
    },
    {
        'name': 'text_offset',
        'values': ['below', 'left', 'right'],
        'handler': lambda val: (
            {'text_offset': [0, 12] if val == 'below' else [-18, -7] if val == 'left' else [18, -7]},f"text_offset_{val}"
        )
    },
    {
        'name': 'text_label',
        'values': ['12'],
        'handler': lambda val: (
            [{'text_label_override': '1'}, {'text_label_override': '2'}],
            f"text_label_{val}"
        )
    },
    {
        'name': 'color',
        'values': ['green', 'blue', 'yellow'],
        'handler': lambda val: (
            {'color': [0, 255, 0] if val == 'green' else [0, 0, 255] if val == 'blue' else [255, 255, 0]},
            f"color_{val}"
        )
    },
    # no_caption groups
    {
        'name': 'no_caption_radius',
        'values': [1, 3, 10, 15],
        'handler': lambda val: (
            [
                {'radius': val, 'show_caption': False, 'color': [255, 0, 0]},
                {'radius': val, 'show_caption': False, 'color': [0, 0, 255]}
            ],
            f"no_caption_radius_{val}"
        )
    },
    {
        'name': 'no_caption_marker_type',
        'values': ['square', 'diamond', 'triangle'],
        'handler': lambda val: (
            [
                {'marker_type': val,'show_caption': False, 'color': [255, 0, 0]},
                {'marker_type': val,'show_caption': False, 'color': [0, 0, 255]}
            ],
            f"no_caption_shape_{val}"
        )
    },
    {
        'name': 'no_caption_different_colors',
        'values': ['default', 'blue_red', 'red_yellow', 'yellow_red'],
        'handler': lambda val: (
            [{'show_caption': False, 'color': [255, 0, 0]}, {'show_caption': False, 'color': [0, 0, 255]}],
            f"no_caption_{val}"
        ) if val == 'default' else (
            [{'show_caption': False, 'color': [0, 0, 255]}, {'show_caption': False, 'color': [255, 0, 0]}],
            f"no_caption_color_{val}"
        ) if val == 'blue_red' else (
            [{'show_caption': False, 'color': [255, 0, 0]}, {'show_caption': False, 'color': [255, 255, 0]}],
            f"no_caption_color_{val}"
        ) if val == 'red_yellow' else (
            [{'show_caption': False, 'color': [255, 255, 0]}, {'show_caption': False, 'color': [255, 0, 0]}],
            f"no_caption_color_{val}"
        )
    },
    # no_marker groups
    {
        'name': 'no_marker_font_scale',
        'values': [0.2, 0.5,1.0],
        'handler': lambda val: (
            {'font_scale': val, 'marker_type': 'none',"text_offset": [0,0]},
            f"no_marker_text_size_{val}"
        )
    },
    {
        'name': 'no_marker_text_label',
        'values': ['12'],
        'handler': lambda val: (
            [{'text_label_override': '1', 'marker_type': 'none',"text_offset": [0,0]}, {'text_label_override': '2', 'marker_type': 'none',"text_offset": [0,0]}],
            f"no_marker_text_label_{val}"
        )
    },
    # JPEG compression groups
    {
        'name': 'jpeg_compression_70',
        'values': [70],
        'handler': lambda val: (
            {'jpeg_compression': val},
            f"jpeg_compression_{val}"
        )
    },
    {
        'name': 'jpeg_compression_80',
        'values': [80],
        'handler': lambda val: (
            {'jpeg_compression': val},
            f"jpeg_compression_{val}"
        )
    },
    {
        'name': 'jpeg_compression_90',
        'values': [90],
        'handler': lambda val: (
            {'jpeg_compression': val},
            f"jpeg_compression_{val}"
        )
    }
]

EXPERIMENT_PARAMETERS_SEMANTIC_CORRESPONDENCE = [
    {
        'name': 'radius',
        'values': [1, 3, 10, 15],
        'handler': lambda val: ({'radius': val}, f"radius_{val}")
    },
    {
        'name': 'marker_type',
        'values': ['square', 'triangle', 'diamond'],
        'handler': lambda val: ({'marker_type': val}, f"marker_type_{val}")
    },
    {
        'name': 'color',
        'values': [ 'green', 'blue', 'yellow'],
        'handler': lambda val: (
            {'color': [0, 255, 0] if val == 'green' else [0, 0, 255] if val == 'blue' else [255, 255, 0]},
            f"color_{val}"
        )
    },
    {
        'name': 'font_scale',
        'values': [0.2, 1.0],
        'handler': lambda val: (
            {'font_scale': val, 'text_offset': [0, -17] if val == 0.2  else [0, -35]},
            f"font_scale_{val}"
        )
    },
    {
        'name': 'text_offset',
        'values': ['below', 'left', 'right'],
        'handler': lambda val: (
            {
                'ref_style': {'text_offset': [0, 12]},
                'target_style': {'text_offset': [0, 12]}
            } if val == 'below' else
            {
                'ref_style': {'text_offset': [-27, -7]},
                'target_style': {'text_offset': [-18, -7]}
            } if val == 'left' else
            {
                'ref_style': {'text_offset': [27, -7]},
                'target_style': {'text_offset': [18, -7]}
            },f"text_offset_{val}"
        ),
    },
    {
        'name': 'text_label',
        'values': [ '1234'],
        'handler': lambda val: (
            [{'text_label_override': str(i+1)} for i in range(4)],
            "text_label_1234"
        )
    },
    {
        'name': 'jpeg_compression_70',
        'values': [70],
        'handler': lambda val: (
            {'jpeg_compression': val},
            f"jpeg_compression_{val}"
        )
    },
    {
        'name': 'jpeg_compression_80',
        'values': [80],
        'handler': lambda val: (
            {'jpeg_compression': val},
            f"jpeg_compression_{val}"
        )
    },
    {
        'name': 'jpeg_compression_90',
        'values': [90],
        'handler': lambda val: (
            {'jpeg_compression': val},
            f"jpeg_compression_{val}"
        )
    }
]
