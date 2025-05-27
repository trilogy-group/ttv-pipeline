# Reference

*Source: [DeepWiki Analysis](https://deepwiki.com/trilogy-group/ttv-pipeline/6-api-and-configuration-reference)*

This page provides reference information for the TTV Pipeline codebase, including licensing terms, project structure, development guidelines, and code organization principles. For information about setup and installation, see [Getting Started](02-getting-started.md).

**Key Source Files:**
- [`.gitignore`](../.gitignore) - Version control exclusions
- [`LICENSE`](../LICENSE) - Project licensing terms
- [`README.md`](../README.md) - Project overview and documentation
- [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample) - Configuration template

## Licensing

The TTV Pipeline is released under the MIT License, providing broad permissions for use, modification, and distribution.

**MIT License Key Features:**
- **Commercial Use**: Free for commercial applications
- **Modification**: Full permission to modify source code
- **Distribution**: Can redistribute original or modified versions
- **Private Use**: No restrictions on private usage
- **Warranty**: No warranty or liability provisions

**License Requirements:**
- Include original copyright notice
- Include license text in distributions
- No additional restrictions on usage

*Source: [`LICENSE`](../LICENSE)*

## Project Structure

The TTV Pipeline follows a modular architecture with clear separation between configuration, source code, output artifacts, and external dependencies.

### Directory Layout

**Core Project Structure:**
```
ttv-pipeline/
├── docs/                    # Documentation files
├── generators/              # Video generation backends
│   ├── local/              # Local GPU generators
│   ├── remote/             # Remote API generators
│   └── factory.py          # Generator factory
├── output/                 # Generated video outputs
├── keyframes/              # Extracted keyframe images
├── models/                 # Local model files
├── frameworks/             # External frameworks (e.g., Wan2.1)
├── pipeline.py             # Main pipeline orchestrator
├── pipeline_config.yaml    # Configuration (git-ignored)
├── pipeline_config.yaml.sample  # Configuration template
├── requirements.txt        # Python dependencies
├── setup.sh               # Installation script
├── Dockerfile             # Container configuration
└── README.md              # Project documentation
```

*Source: [`.gitignore`](../.gitignore)*

### Ignored Files and Directories

The project excludes several categories of files from version control to maintain repository cleanliness and security:

**Python Environment Files:**
- `.venv/`, `venv/`, `env/` - Virtual environments
- `__pycache__/` - Python bytecode cache
- `*.py[cod]` - Compiled Python files

**Configuration and Secrets:**
- `pipeline_config.yaml` - User configuration with API keys
- `*.log` - Log files

**Output Artifacts:**
- `output/` - Generated videos
- `keyframes/` - Extracted frames
- `*.mp4`, `*.png`, `*.jpg` - Media files

**External Dependencies:**
- `models/` - Downloaded model files
- `frameworks/` - External framework repositories

**Development Tools:**
- `.vscode/`, `.idea/` - IDE configuration files
- Development and debugging artifacts

*Source: [`.gitignore`](../.gitignore)*

## Development Workflow

### Development Lifecycle

**Setup Phase:**
1. Clone repository
2. Run `setup.sh` for initial configuration
3. Copy `pipeline_config.yaml.sample` to `pipeline_config.yaml`
4. Configure API keys and backend settings
5. Install dependencies via `requirements.txt`

**Development Phase:**
1. Create feature branches for new functionality
2. Follow code organization principles
3. Update configuration templates as needed
4. Test with multiple backends
5. Update documentation

**Deployment Phase:**
1. Build Docker containers for production
2. Configure container orchestration
3. Set up monitoring and logging
4. Deploy with appropriate resource allocation

## Code Organization Principles

### File Naming Conventions

**Core Files:**
- `pipeline.py` - Main orchestration logic
- `pipeline_config.yaml` - User configuration
- `setup.sh` - Installation automation
- `README.md` - Project documentation
- `LICENSE` - Legal terms

**Directory Conventions:**
- `video_generators/` - Backend implementations
- `docs/` - Documentation files
- `output/` - Generated artifacts

### Directory Structure Principles

The codebase follows these organizational principles:

1. **Separation of Concerns**: Configuration, source code, and outputs are kept in separate directories
2. **Security**: Sensitive files like `pipeline_config.yaml` are excluded from version control
3. **Modularity**: Video generators are organized in a dedicated subdirectory
4. **External Dependencies**: Models and frameworks are isolated in separate directories

**Security Considerations:**
- Configuration files with secrets are git-ignored
- API keys and credentials never committed to repository
- Sensitive data segregated from source code

### Configuration Management

The configuration system uses a template-based approach:

**Template System:**
- **Template**: `pipeline_config.yaml.sample` provides the structure and documentation
- **Instance**: `pipeline_config.yaml` contains actual configuration values
- **Security**: The instance file is git-ignored to protect API keys and sensitive settings

**Configuration Flow:**
1. Developer copies sample to actual config
2. Fills in API keys and preferences
3. Git ignores actual config to prevent secret exposure
4. Template updates are tracked in version control

*Source: [`.gitignore`](../.gitignore)*

## File Type Conventions

### Media File Handling

**Input Media:**
- **Images**: JPEG, PNG formats for keyframes and inputs
- **Videos**: MP4 format for video inputs and references

**Output Media:**
- **Generated Videos**: MP4 format with H.264 encoding
- **Intermediate Frames**: PNG format for processing
- **Keyframes**: JPEG format for frame extraction

### Development Artifacts

**Python Files:**
- **Source Code**: `.py` files with clear module structure
- **Configuration**: YAML files for settings
- **Documentation**: Markdown files for reference

**Build Artifacts:**
- **Containers**: Dockerfile for deployment
- **Dependencies**: requirements.txt for Python packages
- **Scripts**: Shell scripts for automation

### Archive and Distribution

**Version Control:**
- Include only source code and templates
- Exclude user data and generated content
- Maintain clean repository structure

**Distribution:**
- Package source code without sensitive data
- Include sample configurations
- Provide clear setup instructions

## Contributing Guidelines

### Code Standards

**Python Code Quality:**
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Include docstrings for classes and functions
- Maintain consistent error handling

**Configuration Standards:**
- Update sample configuration when adding new options
- Document all configuration parameters
- Provide sensible defaults where possible
- Include validation for critical settings

### Security Practices

**Secret Management:**
- Never commit API keys or credentials
- Use environment variables for sensitive data
- Validate all external inputs
- Implement proper error handling without information leakage

**Access Control:**
- Use principle of least privilege
- Validate all API inputs
- Implement proper authentication for web interfaces
- Monitor access patterns and usage

### Testing and Development

**Testing Requirements:**
- Test with multiple backend configurations
- Validate error handling paths
- Test resource cleanup and management
- Verify security controls

**Development Environment:**
- Use virtual environments for isolation
- Pin dependency versions for reproducibility
- Test container builds before deployment
- Validate configuration templates

**Documentation Updates:**
- Update documentation with code changes
- Maintain accurate API references
- Include examples for new features
- Test documentation procedures

---

## Quick Reference

### Configuration Files
- **`pipeline_config.yaml.sample`**: Template with all options
- **`pipeline_config.yaml`**: User configuration (git-ignored)
- **`requirements.txt`**: Python dependencies
- **`.gitignore`**: Version control exclusions

### Core Scripts
- **`pipeline.py`**: Main application entry point
- **`setup.sh`**: Automated setup and installation
- **`Dockerfile`**: Container build configuration

### Directory Structure
- **`generators/`**: Video generation backend implementations
- **`docs/`**: Comprehensive project documentation
- **`output/`**: Generated video outputs (git-ignored)
- **`models/`**: Local model files (git-ignored)

### Development Commands
```bash
# Setup
./setup.sh

# Configuration
cp pipeline_config.yaml.sample pipeline_config.yaml

# Run pipeline
python pipeline.py

# Container build
docker build -t ttv-pipeline .
```

---

## Next Steps

- **Setup**: See [Getting Started](02-getting-started.md) for installation instructions
- **Configuration**: See [Getting Started](02-getting-started.md) for backend configuration
- **Architecture**: See [Core Pipeline](03-core-pipeline.md) for system overview
- **Deployment**: See [Deployment and Containers](08-deployment-and-containers.md) for production setup
