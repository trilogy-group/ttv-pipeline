"""
Tests for configuration merger and prompt override parity.
"""

import pytest
from api.config_merger import ConfigMerger


class TestConfigMerger:
    """Test configuration merging functionality"""
    
    def test_build_effective_config_base_only(self, config_merger):
        """Test effective config with only base configuration"""
        base_config = {
            "prompt": "Base prompt",
            "size": "1280*720",
            "backend": "veo3",
            "frame_num": 81
        }
        
        result = config_merger.build_effective_config(base_config)
        
        assert result == base_config
        assert result["prompt"] == "Base prompt"
    
    def test_build_effective_config_cli_override(self, config_merger):
        """Test effective config with CLI overrides"""
        base_config = {
            "prompt": "Base prompt",
            "size": "1280*720",
            "backend": "veo3"
        }
        
        cli_args = {
            "prompt": "CLI prompt",
            "backend": "wan2.1"
        }
        
        result = config_merger.build_effective_config(base_config, cli_args=cli_args)
        
        assert result["prompt"] == "CLI prompt"
        assert result["backend"] == "wan2.1"
        assert result["size"] == "1280*720"  # Unchanged from base
    
    def test_build_effective_config_http_override(self, config_merger):
        """Test effective config with HTTP overrides"""
        base_config = {
            "prompt": "Base prompt",
            "size": "1280*720",
            "backend": "veo3"
        }
        
        http_overrides = {
            "prompt": "HTTP prompt",
            "size": "1920*1080"
        }
        
        result = config_merger.build_effective_config(base_config, http_overrides=http_overrides)
        
        assert result["prompt"] == "HTTP prompt"
        assert result["size"] == "1920*1080"
        assert result["backend"] == "veo3"  # Unchanged from base
    
    def test_build_effective_config_precedence(self, config_merger):
        """Test precedence order: HTTP > CLI > config"""
        base_config = {
            "prompt": "Base prompt",
            "size": "1280*720",
            "backend": "veo3"
        }
        
        cli_args = {
            "prompt": "CLI prompt",
            "size": "1920*1080"
        }
        
        http_overrides = {
            "prompt": "HTTP prompt"
        }
        
        result = config_merger.build_effective_config(
            base_config, 
            cli_args=cli_args, 
            http_overrides=http_overrides
        )
        
        # HTTP should override CLI and config
        assert result["prompt"] == "HTTP prompt"
        # CLI should override config when no HTTP override
        assert result["size"] == "1920*1080"
        # Config should be used when no overrides
        assert result["backend"] == "veo3"
    
    def test_build_effective_config_none_values(self, config_merger):
        """Test that None values don't override existing config"""
        base_config = {
            "prompt": "Base prompt",
            "size": "1280*720"
        }
        
        cli_args = {
            "prompt": None,  # Should not override
            "size": "1920*1080"
        }
        
        http_overrides = {
            "prompt": "HTTP prompt",
            "backend": None  # Should not add new key
        }
        
        result = config_merger.build_effective_config(
            base_config,
            cli_args=cli_args,
            http_overrides=http_overrides
        )
        
        assert result["prompt"] == "HTTP prompt"
        assert result["size"] == "1920*1080"
        assert "backend" not in result


class TestPromptOverrideParity:
    """Test prompt override behavior matches CLI precedence"""
    
    def test_validate_prompt_override_parity_http_priority(self, config_merger):
        """Test HTTP prompt has highest priority"""
        base_config = {"prompt": "Config prompt"}
        
        result = config_merger.validate_prompt_override_parity(
            base_config=base_config,
            cli_prompt="CLI prompt",
            http_prompt="HTTP prompt"
        )
        
        assert result == "HTTP prompt"
    
    def test_validate_prompt_override_parity_cli_priority(self, config_merger):
        """Test CLI prompt overrides config when no HTTP prompt"""
        base_config = {"prompt": "Config prompt"}
        
        result = config_merger.validate_prompt_override_parity(
            base_config=base_config,
            cli_prompt="CLI prompt",
            http_prompt=None
        )
        
        assert result == "CLI prompt"
    
    def test_validate_prompt_override_parity_config_fallback(self, config_merger):
        """Test config prompt used when no overrides"""
        base_config = {"prompt": "Config prompt"}
        
        result = config_merger.validate_prompt_override_parity(
            base_config=base_config,
            cli_prompt=None,
            http_prompt=None
        )
        
        assert result == "Config prompt"
    
    def test_validate_prompt_override_parity_no_config_prompt(self, config_merger):
        """Test behavior when no config prompt exists"""
        base_config = {}
        
        result = config_merger.validate_prompt_override_parity(
            base_config=base_config,
            cli_prompt="CLI prompt",
            http_prompt=None
        )
        
        assert result == "CLI prompt"


class TestJobConfigMerging:
    """Test configuration merging for specific jobs"""
    
    def test_merge_for_job_basic(self, config_merger, sample_pipeline_config):
        """Test basic job configuration merging"""
        job_prompt = "Job-specific prompt"
        
        result = config_merger.merge_for_job(sample_pipeline_config, job_prompt)
        
        assert result["prompt"] == job_prompt
        # Other config should remain unchanged
        assert result["size"] == sample_pipeline_config["size"]
        assert result["default_backend"] == sample_pipeline_config["default_backend"]
    
    def test_merge_for_job_preserves_config(self, config_merger, sample_pipeline_config):
        """Test that job merging preserves all other configuration"""
        job_prompt = "New job prompt"
        original_prompt = sample_pipeline_config["prompt"]
        
        result = config_merger.merge_for_job(sample_pipeline_config, job_prompt)
        
        # Prompt should be overridden
        assert result["prompt"] == job_prompt
        assert result["prompt"] != original_prompt
        
        # All other keys should be preserved
        for key, value in sample_pipeline_config.items():
            if key != "prompt":
                assert result[key] == value
    
    def test_merge_for_job_assertion(self, config_merger, sample_pipeline_config):
        """Test that job merging assertion works correctly"""
        job_prompt = "Test prompt"
        
        # This should work without assertion error
        result = config_merger.merge_for_job(sample_pipeline_config, job_prompt)
        assert result["prompt"] == job_prompt


class TestCLIArgsExtraction:
    """Test CLI arguments extraction from configuration"""
    
    def test_extract_cli_args_from_config(self, config_merger, sample_pipeline_config):
        """Test extracting CLI-equivalent arguments from config"""
        cli_args = config_merger.extract_cli_args_from_config(sample_pipeline_config)
        
        expected_mappings = {
            "prompt": sample_pipeline_config["prompt"],
            "backend": sample_pipeline_config["default_backend"],
            "size": sample_pipeline_config["size"],
            "frame_num": sample_pipeline_config["frame_num"],
            "sample_steps": sample_pipeline_config["sample_steps"],
            "guide_scale": sample_pipeline_config["guide_scale"],
            "seed": sample_pipeline_config["base_seed"],
            "output_dir": sample_pipeline_config["output_dir"]
        }
        
        for cli_key, expected_value in expected_mappings.items():
            assert cli_args[cli_key] == expected_value
    
    def test_extract_cli_args_partial_config(self, config_merger):
        """Test extracting CLI args from partial configuration"""
        partial_config = {
            "prompt": "Test prompt",
            "size": "1280*720",
            # Missing other keys
        }
        
        cli_args = config_merger.extract_cli_args_from_config(partial_config)
        
        assert cli_args["prompt"] == "Test prompt"
        assert cli_args["size"] == "1280*720"
        # Should not contain keys for missing config values
        assert "frame_num" not in cli_args
        assert "backend" not in cli_args


class TestUtilityMethods:
    """Test utility methods for CLI integration"""
    
    def test_args_to_dict_with_namespace(self, config_merger):
        """Test converting argparse.Namespace to dictionary"""
        import argparse
        
        # Create a mock Namespace object
        args = argparse.Namespace()
        args.prompt = "Test prompt"
        args.size = "1280*720"
        args.frame_num = None  # Should be filtered out
        args.backend = "veo3"
        
        result = config_merger.args_to_dict(args)
        
        expected = {
            "prompt": "Test prompt",
            "size": "1280*720", 
            "backend": "veo3"
        }
        
        assert result == expected
        assert "frame_num" not in result  # None values filtered out
    
    def test_args_to_dict_with_dict(self, config_merger):
        """Test converting dictionary to filtered dictionary"""
        args_dict = {
            "prompt": "Test prompt",
            "size": None,  # Should be filtered out
            "backend": "veo3",
            "frame_num": 81
        }
        
        result = config_merger.args_to_dict(args_dict)
        
        expected = {
            "prompt": "Test prompt",
            "backend": "veo3",
            "frame_num": 81
        }
        
        assert result == expected
        assert "size" not in result  # None values filtered out
    
    def test_merge_with_args_namespace(self, config_merger, sample_pipeline_config):
        """Test merging configuration with argparse.Namespace"""
        import argparse
        
        base_config = sample_pipeline_config.copy()
        
        # Create mock args
        args = argparse.Namespace()
        args.prompt = "CLI prompt from args"
        args.size = "1920*1080"
        args.frame_num = None  # Should not override
        
        result = config_merger.merge_with_args(base_config, args)
        
        assert result["prompt"] == "CLI prompt from args"
        assert result["size"] == "1920*1080"
        assert result["frame_num"] == sample_pipeline_config["frame_num"]  # Unchanged
    
    def test_merge_with_args_and_http_overrides(self, config_merger, sample_pipeline_config):
        """Test merging with both CLI args and HTTP overrides"""
        import argparse
        
        base_config = sample_pipeline_config.copy()
        
        # Create mock args - use CLI argument names (will be mapped to config keys)
        args = argparse.Namespace()
        args.prompt = "CLI prompt"
        args.backend = "wan2.1"  # CLI argument name (maps to default_backend)
        
        http_overrides = {
            "prompt": "HTTP prompt"  # Should take precedence
        }
        
        result = config_merger.merge_with_args(base_config, args, http_overrides)
        
        assert result["prompt"] == "HTTP prompt"  # HTTP wins
        assert result["default_backend"] == "wan2.1"  # CLI wins (no HTTP override)
    
    def test_merge_with_none_args(self, config_merger, sample_pipeline_config):
        """Test merging with None args"""
        base_config = sample_pipeline_config.copy()
        
        result = config_merger.merge_with_args(base_config, None)
        
        # Should be identical to base config
        assert result == base_config
    
    def test_cli_to_config_key_mapping(self, config_merger):
        """Test CLI argument name to config key mapping"""
        cli_args = {
            "prompt": "Test prompt",
            "backend": "veo3",
            "size": "1920*1080",
            "seed": 42,
            "unknown_arg": "should_pass_through"
        }
        
        result = config_merger._map_cli_to_config_keys(cli_args)
        
        expected = {
            "prompt": "Test prompt",
            "default_backend": "veo3",  # backend -> default_backend
            "size": "1920*1080",
            "base_seed": 42,  # seed -> base_seed
            "unknown_arg": "should_pass_through"  # unmapped args pass through
        }
        
        assert result == expected


class TestPromptOverrideParityIntegration:
    """Integration tests to verify prompt override behavior matches CLI precedence"""
    
    def test_cli_prompt_override_matches_existing_behavior(self, config_merger, sample_pipeline_config):
        """Test that CLI prompt override matches the existing pipeline behavior"""
        base_config = sample_pipeline_config.copy()
        cli_prompt = "CLI override prompt"
        
        # Test the old way (direct override)
        old_way_prompt = cli_prompt if cli_prompt else base_config.get("prompt")
        
        # Test the new way (using ConfigMerger)
        cli_args = {'prompt': cli_prompt}
        effective_config = config_merger.build_effective_config(
            base_config=base_config,
            cli_args=cli_args,
            http_overrides=None
        )
        new_way_prompt = effective_config.get("prompt")
        
        # Both should produce the same result
        assert old_way_prompt == new_way_prompt == cli_prompt
    
    def test_no_cli_prompt_matches_existing_behavior(self, config_merger, sample_pipeline_config):
        """Test that no CLI prompt override matches the existing pipeline behavior"""
        base_config = sample_pipeline_config.copy()
        cli_prompt = None
        
        # Test the old way (direct override)
        old_way_prompt = cli_prompt if cli_prompt else base_config.get("prompt")
        
        # Test the new way (using ConfigMerger)
        cli_args = {'prompt': cli_prompt} if cli_prompt else None
        effective_config = config_merger.build_effective_config(
            base_config=base_config,
            cli_args=cli_args,
            http_overrides=None
        )
        new_way_prompt = effective_config.get("prompt")
        
        # Both should produce the same result
        assert old_way_prompt == new_way_prompt == base_config["prompt"]
    
    def test_http_prompt_takes_precedence_over_cli(self, config_merger, sample_pipeline_config):
        """Test that HTTP prompt takes precedence over CLI prompt (API context)"""
        base_config = sample_pipeline_config.copy()
        cli_prompt = "CLI prompt"
        http_prompt = "HTTP prompt"
        
        # Simulate API context where both CLI and HTTP prompts might exist
        cli_args = {'prompt': cli_prompt}
        http_overrides = {'prompt': http_prompt}
        
        effective_config = config_merger.build_effective_config(
            base_config=base_config,
            cli_args=cli_args,
            http_overrides=http_overrides
        )
        
        # HTTP should win
        assert effective_config["prompt"] == http_prompt
    
    def test_complex_precedence_scenario(self, config_merger):
        """Test complex precedence scenario with multiple overrides"""
        base_config = {
            "prompt": "Config prompt",
            "size": "1280*720",
            "backend": "veo3",
            "frame_num": 81,
            "guide_scale": 5.0
        }
        
        cli_args = {
            "prompt": "CLI prompt",
            "size": "1920*1080",
            "frame_num": 121
        }
        
        http_overrides = {
            "prompt": "HTTP prompt",
            "backend": "wan2.1"
        }
        
        effective_config = config_merger.build_effective_config(
            base_config=base_config,
            cli_args=cli_args,
            http_overrides=http_overrides
        )
        
        # Verify precedence: HTTP > CLI > config
        assert effective_config["prompt"] == "HTTP prompt"  # HTTP wins
        assert effective_config["backend"] == "wan2.1"      # HTTP wins
        assert effective_config["size"] == "1920*1080"      # CLI wins (no HTTP override)
        assert effective_config["frame_num"] == 121         # CLI wins (no HTTP override)
        assert effective_config["guide_scale"] == 5.0       # Config wins (no overrides)
    
    def test_empty_overrides_preserve_config(self, config_merger, sample_pipeline_config):
        """Test that empty overrides preserve original configuration"""
        base_config = sample_pipeline_config.copy()
        
        effective_config = config_merger.build_effective_config(
            base_config=base_config,
            cli_args={},
            http_overrides={}
        )
        
        # Should be identical to base config
        assert effective_config == base_config
    
    def test_none_overrides_preserve_config(self, config_merger, sample_pipeline_config):
        """Test that None overrides preserve original configuration"""
        base_config = sample_pipeline_config.copy()
        
        effective_config = config_merger.build_effective_config(
            base_config=base_config,
            cli_args=None,
            http_overrides=None
        )
        
        # Should be identical to base config
        assert effective_config == base_config