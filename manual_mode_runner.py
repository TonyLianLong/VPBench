import argparse
import subprocess
import sys
import json
import copy
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from evaluation.config import *

def get_dataset_path(task_name, dataset_type=None):
    """Get the appropriate dataset path based on task and dataset type.
    
    Args:
        task_name (str): Name of the task (e.g., 'Relative_Depth', 'Semantic_Correspondence')
        dataset_type (str, optional): Type of dataset (e.g., 'BLINK', 'DA-2K', 'SPair-71k')
        
    Returns:
        str: Path to the dataset
    """
    task_config = DATA_ROOT_CONFIG.get(task_name, {})
    
    if isinstance(task_config, dict) and dataset_type:
        return task_config.get(dataset_type)
    elif isinstance(task_config, dict):
        # Use default dataset for the task
        default_dataset = task_config.get('default', 'BLINK')
        return task_config.get(default_dataset)
    else:
        # Backward compatibility for simple string paths
        return task_config

def rgb_to_color_name(rgb):
    """Convert RGB color to a color name.

    Args:
        rgb (list): RGB color value.

    Returns:
        str: Color name or RGB value as a string.
    """
    color_map = {
        (255, 0, 0): "red",
        (0, 0, 255): "blue",
        (0, 255, 0): "green",
        (255, 255, 0): "yellow",
    }
    return color_map.get(tuple(rgb), str(rgb))


def format_arg(flag, value):
    return f"{flag}={shlex.quote(str(value))}"


def parse_provider_list(raw_value):
    """Parse provider list input which may be JSON or comma-separated."""
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if not text or text == "[]":
        return []

    parsed = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Fall back to comma-separated parsing
        parsed = [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        return []

    cleaned = []
    for item in parsed:
        if isinstance(item, str):
            candidate = item.strip()
        else:
            candidate = str(item).strip()
        if candidate:
            cleaned.append(candidate)
    return cleaned


ALLOWED_BASE_URL_PREFIXES = (
    "https://api.openai.com/v1",
    "https://openrouter.ai/api/v1",
)


def normalize_and_validate_base_url(url):
    """Normalize base URL and ensure it's one of the supported cloud endpoints."""
    if not url:
        return ""
    normalized = url.rstrip("/")
    if not any(normalized.startswith(prefix) for prefix in ALLOWED_BASE_URL_PREFIXES):
        raise ValueError(
            f"Unsupported base URL '{url}'. Only OpenAI ({ALLOWED_BASE_URL_PREFIXES[0]}) "
            f"and OpenRouter ({ALLOWED_BASE_URL_PREFIXES[1]}) endpoints are supported."
        )
    return normalized

def build_prompt(task_name, styles, suffix):
    if task_name == "Relative_Depth":
        if not styles:
            return None
        style_a, style_b = styles[0], styles[1]
        marker_type = style_a.get("marker_type", "circle")
        label_a = style_a.get("text_label_override", "A")
        label_b = style_b.get("text_label_override", "B")
        color_a = rgb_to_color_name(style_a.get("color", [255, 0, 0]))
        color_b = rgb_to_color_name(style_b.get("color", [255, 0, 0]))

        if 'no_marker' in suffix:
            return PROMPT_NO_MARKER.format(label_a=label_a, label_b=label_b)
        elif 'no_caption' in suffix:
            return PROMPT_NO_CAPTION.format(marker_type=marker_type, color_a=color_a, color_b=color_b)
        else:
            return PROMPT_DEFAULT.format(marker_type=marker_type, label_a=label_a, label_b=label_b)

    elif task_name == "Semantic_Correspondence":
        if not styles:
            return None
        print(f"styles: {styles[0]}")
        ref_style = styles[0][0]
        target_styles = styles[1]
        marker_type = ref_style.get("marker_type", "circle")
        if marker_type == "diamond":
            marker_type_d = "diamonded"
        else:
            marker_type_d = f"{marker_type}d"
        color = rgb_to_color_name(ref_style.get("color", [255, 0, 0]))
        label_a = target_styles[0].get("text_label_override", "A")
        label_b = target_styles[1].get("text_label_override", "B")
        label_c = target_styles[2].get("text_label_override", "C")
        label_d = target_styles[3].get("text_label_override", "D")
        return PROMPT_SEMANTIC_CORRESPONDENCE.format(
            marker_type_d=marker_type_d,marker_type=marker_type, color=color,
            label_a=label_a, label_b=label_b, label_c=label_c, label_d=label_d
        )
    else:
        return None
    
    
    
    
    
    
    
def construct_command(model_name, task_name, suffix, styles, data_root=None, output_dir=None, dataset_type=None, jpeg_quality=100, overwrite=False, checkpoint_interval=25, jpeg_compression=None, debug_run=False, provider_only=None, save_images=False):
    """Construct the command to run the test benchmark script.

    Args:
        model_name (str): Name of the model.
        task_name (str): Name of the task.
        suffix (str): Suffix for the command.
        styles (list or dict): Styles to be used in the command.
        data_root (str, optional): Data root directory path.
        output_dir (str, optional): Output directory path.
        dataset_type (str, optional): Dataset type (BLINK, DA-2K, SPair-71k).
        jpeg_quality (int): JPEG compression quality (0-100).
        jpeg_compression (int, optional): JPEG compression quality for base64 encoding (0-100).
        provider_only (list, optional): Provider filter list for OpenRouter-based runs.

    Returns:
        str: Constructed command as a string.
    """
    suffix_value = suffix or ""
    cmd_parts = [
        sys.executable,
        "./benchmark_executor.py",
        format_arg("--model_name", model_name),
        format_arg("--task_name", task_name),
        "--image_first",
        format_arg("--suffix", suffix_value)
    ]

    # Add save_annotated_images flag if requested
    if save_images:
        cmd_parts.append("--save_annotated_images")

    if debug_run:
        cmd_parts.append("--debug_run")

    # Add data_root if specified
    if data_root:
        cmd_parts.append(format_arg("--data-root", data_root))

    # Add dataset_type if specified
    if dataset_type:
        cmd_parts.append(format_arg("--dataset-type", dataset_type))

    # Add output_dir if specified  
    if output_dir:
        cmd_parts.append(format_arg("--output_dir", output_dir))

    # Add JPEG quality parameter
    cmd_parts.append(format_arg("--jpeg-quality", jpeg_quality))

    # Add JPEG compression parameter if specified
    if jpeg_compression is not None:
        cmd_parts.append(format_arg("--jpeg-compression", jpeg_compression))

    if overwrite:
        cmd_parts.append("--overwrite")

    if checkpoint_interval is not None:
        cmd_parts.append(format_arg("--checkpoint_interval", checkpoint_interval))

    if styles is not None:
        styles_json = json.dumps(styles)
        cmd_parts.append(f"--styles='{styles_json}'")
        prompt = build_prompt(task_name, styles, suffix)
        if prompt:
            cmd_parts.append(f"--prompt='{prompt}'")
    if provider_only:
        provider_json = json.dumps(provider_only)
        cmd_parts.append(format_arg("--provider-only", provider_json))

    return " ".join(cmd_parts)

def apply_style_update(modified_style, update_data, task_name):
    if task_name == "Relative_Depth":
        if isinstance(update_data, dict):
            for style in modified_style:
                style.update(update_data)
        elif isinstance(update_data, list):
            modified_style[:] = update_data
    elif task_name == "Semantic_Correspondence":
        if isinstance(update_data, dict):
            if 'ref_style' in update_data and 'target_style' in update_data:
                modified_style[0][0].update(update_data['ref_style'])
                for style in modified_style[1]: 
                    style.update(update_data['target_style'])
                del update_data['ref_style']
                del update_data['target_style']
            for style in modified_style[0]: 
                style.update(update_data)
            for style in modified_style[1]: 
                style.update(update_data)
        elif isinstance(update_data, list):
            modified_style[1] = update_data
            
def generate_compression_commands(args, default_style, default_suffix, task_kwargs=None):
    """Generate commands for compression rate testing.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        default_style (list or dict): Default style configuration.
        default_suffix (str): Default suffix for the commands.

    Returns:
        list: List of generated commands for compression testing.
    """
    commands = []
    task_kwargs = task_kwargs or {}
    provider_only = task_kwargs.get("provider_only")
    compression_qualities = [100, 90, 80, 70]  # JPEG quality levels
    
    # Get data_root and output_dir from config or args
    data_root = getattr(args, 'data_root', None) or get_dataset_path(args.task_name, getattr(args, 'dataset_type', None))
    output_dir = getattr(args, 'output_dir', None) or OUTPUT_DIR_CONFIG.get('base_dir')
    
    # Create task-specific output directory if needed
    if output_dir and args.task_name:
        task_output_dir = os.path.join(output_dir, args.task_name.lower())
        # Add dataset type to output directory if specified
        dataset_subdir = getattr(args, 'dataset_type', None)
        if dataset_subdir:
            subdir_name = dataset_subdir.lower().replace(' ', '_')
            task_output_dir = os.path.join(task_output_dir, subdir_name)
        output_dir = task_output_dir
    
    dataset_type_arg = getattr(args, 'dataset_type', None)

    for model_name in args.model_names:
        task_name = args.task_name
        for run_idx in range(args.run_time):
            for quality in compression_qualities:
                # Generate suffix for compression
                if quality == 100:
                    current_suffix = f"{default_suffix}_original"
                else:
                    current_suffix = f"{default_suffix}_compressed_{quality}"
                
                if run_idx > 0:
                    current_suffix += f"_run{run_idx + 1}"

                cmd = construct_command(
                    model_name,
                    task_name,
                    current_suffix,
                    default_style,
                    data_root=data_root,
                    output_dir=output_dir,
                    dataset_type=dataset_type_arg,
                    jpeg_quality=quality,
                    overwrite=args.overwrite,
                    checkpoint_interval=args.checkpoint_interval,
                    debug_run=args.debug_run,
                    provider_only=provider_only,
                    save_images=args.save_images
                )
                commands.append(cmd)

    return commands

def generate_commands(args, default_style, default_suffix, experiment_params, task_kwargs=None, default_only=False):
    """Generate all commands based on the provided arguments and default configurations.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        default_style (list or dict): Default style configuration.
        default_suffix (str): Default suffix for the commands.

    Returns:
        list: List of generated commands.
    """
    commands = []
    task_kwargs = task_kwargs or {}
    provider_only = task_kwargs.get("provider_only")
    
    # Get data_root and output_dir from config or args
    data_root = getattr(args, 'data_root', None) or get_dataset_path(args.task_name, getattr(args, 'dataset_type', None))
    output_dir = getattr(args, 'output_dir', None) or OUTPUT_DIR_CONFIG.get('base_dir')
    
    # Create task-specific output directory if needed
    if output_dir and args.task_name:
        task_output_dir = os.path.join(output_dir, args.task_name.lower())
        # Add dataset type to output directory if specified
        dataset_subdir = getattr(args, 'dataset_type', None)
        if dataset_subdir:
            subdir_name = dataset_subdir.lower().replace(' ', '_')
            task_output_dir = os.path.join(task_output_dir, subdir_name)
        output_dir = task_output_dir
    
    dataset_type_arg = getattr(args, 'dataset_type', None)

    for model_name in args.model_names:
        task_name = args.task_name
        for run_idx in range(args.run_time):
            # Generate default command
            current_suffix = default_suffix
            if run_idx > 0:
                current_suffix += f"_run{run_idx + 1}"

            default_cmd = construct_command(
                model_name,
                task_name,
                current_suffix,
                default_style,
                data_root=data_root,
                output_dir=output_dir,
                dataset_type=dataset_type_arg,
                overwrite=args.overwrite,
                checkpoint_interval=args.checkpoint_interval,
                debug_run=args.debug_run,
                provider_only=provider_only,
                save_images=args.save_images
            )
            commands.append(default_cmd)

            if default_only:
                continue

            # Generate experimental commands for each group
            for group in experiment_params:
                for param_val in group['values']:
                    modified_style = copy.deepcopy(default_style)
                    
                    update_data, suffix_part = group['handler'](param_val)
                    
                    # Extract jpeg_compression from update_data if present
                    jpeg_compression = None
                    if isinstance(update_data, dict) and 'jpeg_compression' in update_data:
                        jpeg_compression = update_data['jpeg_compression']
                        # Remove jpeg_compression from update_data so it doesn't get added to styles
                        update_data = {k: v for k, v in update_data.items() if k != 'jpeg_compression'}
                    
                    apply_style_update(modified_style, update_data, args.task_name)

                    suffix = f"{default_suffix}_{suffix_part}"
                    if run_idx > 0:
                        suffix += f"_run{run_idx + 1}"

                    # Construct and append command
                    full_cmd = construct_command(
                        model_name,
                        task_name,
                        suffix,
                        modified_style,
                        data_root=data_root,
                        output_dir=output_dir,
                        dataset_type=dataset_type_arg,
                        overwrite=args.overwrite,
                        checkpoint_interval=args.checkpoint_interval,
                        jpeg_compression=jpeg_compression,
                        debug_run=args.debug_run,
                        provider_only=provider_only,
                        save_images=args.save_images
                    )
                    commands.append(full_cmd)

    return commands

def main():
    """Main function to parse arguments and execute commands.
    This function sets up the command line argument parser, generates commands based on the provided arguments, and executes them in parallel using multiple threads.
    
    run example:
    python ./run_script_visual_prompt_bias/manual_mode_runner.py \
        --openai_base_url=https://api.openai.com/v1 \
        --model_names gpt-4o \
        --task_name Relative_Depth \
        --dataset_type BLINK \
        --run_time 1 \
        --num_threads 8 \
        --show_scripts
        
    # Run with DA-2K dataset instead of default BLINK:
    python ./run_script_visual_prompt_bias/manual_mode_runner.py \
        --model_names gpt-4o \
        --task_name Relative_Depth \
        --dataset_type DA-2K \
        --run_time 1 \
        --num_threads 8
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_time", type=int, default=1)
    parser.add_argument("--model_names", nargs='+', default=['gpt-4o'])
    parser.add_argument("--task_name", type=str, default="Relative_Depth")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default="",
        help="Base URL (OpenAI or OpenRouter only)",
    )
    parser.add_argument("--show_scripts", action="store_true", help="Print and save all generated scripts")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root directory (uses config default if not specified)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory (uses config default if not specified)")
    parser.add_argument("--dataset_type", type=str, default=None, help="Specify dataset type (e.g., BLINK, DA-2K, SPair-71k). Uses task default if not specified.")
    parser.add_argument(
        "--provider-only",
        type=str,
        default=None,
        help="JSON list or comma-separated OpenRouter provider identifiers to forward to benchmark commands",
    )
    parser.add_argument("--compression_test", action="store_true", help="Run compression rate testing (qualities: 100, 90, 80, 70)")
    parser.add_argument("--limit_run_num", type=int, default=None, help="Run only the first N generated commands")
    parser.add_argument("--dry_run", action="store_true", help="Show commands without executing them")
    parser.add_argument("--default_only", action="store_true", help="Only run the default configuration without experimental variants")
    parser.add_argument("--overwrite", action="store_true", help="Force evaluations to start fresh without resuming progress")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Save intermediate JSON after N successful queries")
    parser.add_argument("--debug_run", action="store_true", help="Run only the first 10 samples per split and save outputs with a '_debug' suffix")
    parser.add_argument("--save_images", action="store_true", help="Save annotated images to filesystem (disabled by default)")
    parser.add_argument("--save_commands", action="store_true", help="Save generated commands to generated_commands.txt (disabled by default)")
    args = parser.parse_args()

    if args.checkpoint_interval < 0:
        parser.error("--checkpoint_interval must be non-negative")

    # Validate dataset_type if specified
    if args.dataset_type:
        valid_datasets = DATASET_CHOICES.get(args.task_name, [])
        if args.dataset_type not in valid_datasets:
            parser.error(f"Invalid dataset_type '{args.dataset_type}' for task '{args.task_name}'. Valid choices: {valid_datasets}")

    if args.limit_run_num is not None and args.limit_run_num < 1:
        parser.error("--limit_run_num must be a positive integer")

    task_kwargs = {}

    if args.task_name == "Relative_Depth":
        default_style = default_style_relative_depth
        default_suffix = default_suffix_relative_depth
        experiment_params = EXPERIMENT_PARAMETERS
    elif args.task_name == "Semantic_Correspondence":
        default_style = default_style_semantic
        default_suffix = default_suffix_semantic
        experiment_params = EXPERIMENT_PARAMETERS_SEMANTIC_CORRESPONDENCE
    else:
        raise ValueError(f"Unsupported task: {args.task_name}. Supported tasks: Relative_Depth, Semantic_Correspondence")

    provider_env = os.getenv("PROVIDER_ONLY")
    if provider_env is None:
        provider_env = os.getenv("PROVIDERS")

    provider_raw = args.provider_only if args.provider_only is not None else provider_env
    if not provider_raw:
        provider_raw = "[]"
    provider_list = parse_provider_list(provider_raw)
    base_url_candidate = args.openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
    base_url_for_provider = normalize_and_validate_base_url(base_url_candidate)
    if provider_list:
        if not base_url_for_provider:
            raise ValueError(
                "Provider filters require an OpenRouter base URL. "
                "Set --openai_base_url=https://openrouter.ai/api/v1 (or OPENAI_BASE_URL)."
            )
        if "openrouter" in base_url_for_provider.lower():
            task_kwargs["provider_only"] = provider_list
        else:
            raise ValueError("Provider filters are only supported with the OpenRouter endpoint.")

    # Generate all commands
    if args.compression_test:
        commands = generate_compression_commands(args, default_style, default_suffix, task_kwargs)
    else:
        commands = generate_commands(
            args,
            default_style,
            default_suffix,
            experiment_params,
            task_kwargs,
            default_only=args.default_only,
        )

    if args.limit_run_num is not None:
        commands = commands[:args.limit_run_num]
    
    # Add environment setup if needed
    processed_commands = []
            
    for command in commands:
        if base_url_for_provider:
            processed_commands.append(f"export OPENAI_BASE_URL={base_url_for_provider} && {command}")
        else:
            processed_commands.append(command)
    
    # Save commands to a file (only if save_commands is enabled)
    if args.save_commands:
        with open("./generated_commands.txt", "w") as f:
            for cmd in processed_commands:
                f.write(cmd + "\n")
            
    # Execute commands in parallel unless this is a dry run
    if args.dry_run:
        for command in processed_commands:
            print(command)
    else:
        def run_command(command):
            print(f"Executing: {command}")
            try:
                subprocess.run(command, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                print(f"Command failed with error: {e}")

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            executor.map(run_command, processed_commands)

if __name__ == "__main__":
    main()
