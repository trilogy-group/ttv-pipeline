"""
Configuration merger that handles precedence rules for prompt overrides.

This module implements the configuration merging logic that ensures
HTTP prompts have the same precedence as CLI arguments over base configuration.
Precedence order: HTTP > CLI > config file
"""

import logging
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigMerger:
    """
    Handles configuration merging with proper precedence rules.
    
    Precedence order (highest to lowest):
    1. HTTP request parameters
    2. CLI arguments  
    3. Configuration file values
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_effective_config(
        self,
        base_config: Dict[str, Any],
        cli_args: Optional[Dict[str, Any]] = None,
        http_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build effective configuration with proper precedence.
        
        Args:
            base_config: Base configuration from file
            cli_args: CLI arguments (optional)
            http_overrides: HTTP request overrides (optional)
            
        Returns:
            Merged configuration dictionary
        """
        # Start with a deep copy of base config
        effective_config = deepcopy(base_config)
        
        # Track what was overridden for logging
        overrides_applied = []
        
        # Apply CLI arguments (second priority)
        if cli_args:
            for key, value in cli_args.items():
                if value is not None:  # Only override if value is provided
                    old_value = effective_config.get(key)
                    effective_config[key] = value
                    overrides_applied.append(f"CLI: {key}={value} (was: {old_value})")
        
        # Apply HTTP overrides (highest priority)
        if http_overrides:
            for key, value in http_overrides.items():
                if value is not None:  # Only override if value is provided
                    old_value = effective_config.get(key)
                    effective_config[key] = value
                    overrides_applied.append(f"HTTP: {key}={value} (was: {old_value})")
        
        # Log the overrides that were applied
        if overrides_applied:
            self.logger.info(f"Configuration overrides applied: {', '.join(overrides_applied)}")
        else:
            self.logger.info("No configuration overrides applied")
        
        return effective_config
    
    def validate_prompt_override_parity(
        self,
        base_config: Dict[str, Any],
        cli_prompt: Optional[str] = None,
        http_prompt: Optional[str] = None
    ) -> str:
        """
        Validate that prompt override behavior matches CLI precedence.
        
        Args:
            base_config: Base configuration
            cli_prompt: CLI prompt override
            http_prompt: HTTP prompt override
            
        Returns:
            The effective prompt value
        """
        config_prompt = base_config.get('prompt')
        
        # Apply precedence: HTTP > CLI > config
        if http_prompt is not None:
            effective_prompt = http_prompt
            source = "HTTP"
        elif cli_prompt is not None:
            effective_prompt = cli_prompt
            source = "CLI"
        else:
            effective_prompt = config_prompt
            source = "config"
        
        self.logger.info(f"Effective prompt from {source}: {effective_prompt[:50]}...")
        return effective_prompt
    
    def merge_for_job(
        self,
        base_config: Dict[str, Any],
        job_prompt: str
    ) -> Dict[str, Any]:
        """
        Merge configuration for a specific job with HTTP prompt override.
        
        Args:
            base_config: Base pipeline configuration
            job_prompt: Prompt from HTTP request
            
        Returns:
            Effective configuration for the job
        """
        # HTTP prompt takes highest precedence
        http_overrides = {'prompt': job_prompt}
        
        effective_config = self.build_effective_config(
            base_config=base_config,
            cli_args=None,  # No CLI args in API context
            http_overrides=http_overrides
        )
        
        # Ensure the prompt override is properly applied
        assert effective_config['prompt'] == job_prompt, \
            f"Prompt override failed: expected {job_prompt}, got {effective_config.get('prompt')}"
        
        return effective_config
    
    def extract_cli_args_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract CLI-equivalent arguments from configuration for testing precedence.
        
        This is used to simulate CLI arguments when testing the merger.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of CLI-equivalent arguments
        """
        # Map config keys to CLI argument equivalents
        cli_mappings = {
            'prompt': 'prompt',
            'default_backend': 'backend',
            'size': 'size',
            'frame_num': 'frame_num',
            'sample_steps': 'sample_steps',
            'guide_scale': 'guide_scale',
            'base_seed': 'seed',
            'output_dir': 'output_dir'
        }
        
        cli_args = {}
        for config_key, cli_key in cli_mappings.items():
            if config_key in config:
                cli_args[cli_key] = config[config_key]
        
        return cli_args
    
    def args_to_dict(self, args) -> Dict[str, Any]:
        """
        Convert argparse Namespace to dictionary, filtering out None values.
        
        This is useful for integrating with existing CLI tools that use argparse.
        
        Args:
            args: argparse.Namespace object or dict-like object
            
        Returns:
            Dictionary with non-None values from the args
        """
        if hasattr(args, '__dict__'):
            # Handle argparse.Namespace
            args_dict = vars(args)
        else:
            # Handle dict-like objects
            args_dict = dict(args)
        
        # Filter out None values to avoid overriding config with empty values
        return {k: v for k, v in args_dict.items() if v is not None}
    
    def merge_with_args(
        self,
        base_config: Dict[str, Any],
        args,
        http_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to merge configuration with argparse args.
        
        This method handles the mapping from CLI argument names to config keys.
        
        Args:
            base_config: Base configuration dictionary
            args: argparse.Namespace object or dict-like object
            http_overrides: HTTP request overrides (optional)
            
        Returns:
            Merged configuration dictionary
        """
        if not args:
            cli_args = None
        else:
            # Convert args to dict and map CLI names to config keys
            args_dict = self.args_to_dict(args)
            cli_args = self._map_cli_to_config_keys(args_dict)
        
        return self.build_effective_config(
            base_config=base_config,
            cli_args=cli_args,
            http_overrides=http_overrides
        )
    
    def _map_cli_to_config_keys(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map CLI argument names to configuration keys.
        
        Args:
            cli_args: Dictionary with CLI argument names as keys
            
        Returns:
            Dictionary with configuration keys
        """
        # Reverse mapping from CLI names to config keys
        cli_to_config_mappings = {
            'prompt': 'prompt',
            'backend': 'default_backend',
            'size': 'size',
            'frame_num': 'frame_num',
            'sample_steps': 'sample_steps',
            'guide_scale': 'guide_scale',
            'seed': 'base_seed',
            'output_dir': 'output_dir'
        }
        
        config_args = {}
        for cli_key, value in cli_args.items():
            # Map CLI key to config key if mapping exists, otherwise use as-is
            config_key = cli_to_config_mappings.get(cli_key, cli_key)
            config_args[config_key] = value
        
        return config_args