# Overview

*Source: [DeepWiki Analysis](https://deepwiki.com/trilogy-group/ttv-pipeline/1-overview)*

## System Purpose and Architecture

The TTV Pipeline solves the core challenge of generating long-form videos by breaking down complex prompts into manageable segments and supporting multiple generation backends including local GPU processing and cloud APIs. The system implements a clean abstraction layer that enables seamless switching between different video generation technologies.

## High-Level System Flow

The system components bridge between conceptual understanding and actual code implementation, enabling developers to easily locate and understand the implementation.

```mermaid
graph TD
    A[Text Prompt Input] --> B[pipeline.py]
    B --> C[OpenAI Enhancement]
    C --> D{Generation Mode}
    D -- keyframe --> E[keyframe_generator.py]
    D -- chaining --> F[Direct I2V Processing]
    E --> G[create_video_generator]
    F --> G
    G --> H[VideoGeneratorInterface]
    H --> I[Wan21Generator]
    H --> J[RunwayMLGenerator]
    H --> K[Veo3Generator]
    H --> L[MinimaxGenerator]
    I --> M[Local GPU Processing]
    J --> N[Runway ML API]
    K --> O[Google Veo 3 API]
    L --> P[Minimax API]
    M --> Q[Video Segments]
    N --> Q
    O --> Q
    P --> Q
    Q --> R[ffmpeg Concatenation]
    R --> S[Final Video Output]
```

## Core Pipeline Components

The system is built around a modular architecture with clear separation of concerns between prompt processing, video generation, and output handling.

```mermaid
graph TD
    subgraph "Configuration System"
        A1[load_config] --> A2[pipeline_config.yaml]
        A2 --> A3[Backend Selection Logic]
    end
    
    subgraph "Prompt Enhancement"
        B1[OpenAI API + Instructor] --> B2[Enhanced Prompt JSON]
        B2 --> B3[Segment Prompts]
    end
    
    subgraph "Keyframe Generation"
        C1[keyframe_generator.py] --> C2[Stability AI API]
        C1 --> C3[OpenAI Image API]
        C2 --> C4[Generated Keyframes]
        C3 --> C4
    end
    
    subgraph "Video Generation"
        D1[generators/factory.py:create] --> D2[VideoGeneratorInterface]
        D2 --> D3[Backend Registry]
        D3 --> D4[Generator Implementations]
    end
    
    A3 --> D1
    B3 --> C1
    B3 --> D1
    C4 --> D1
    D4 --> E[pipeline.py:create_final_video]
```

## Video Generation Backend Architecture

The system implements a sophisticated abstraction layer that supports multiple video generation backends through a unified interface, enabling seamless switching between local GPU processing and cloud APIs.

### Backend Abstraction and Factory Pattern

The backend architecture uses a factory pattern with actual class names and module structure:


## Generation Modes and Processing Flow

The system supports two distinct generation modes, each optimized for different use cases and processing requirements.

### Keyframe vs Chaining Mode Implementation

```mermaid
graph TD
    A[Text Prompt] --> B[OpenAI Enhancement]
    B --> C[Segment Prompts]
    
    %% Split into two paths based on generation mode
    C --> D{Generation Mode}
    
    %% Keyframe Mode Path
    D -->|Keyframe| E[keyframe_generator.py]
    E --> F[Stability/OpenAI APIs]
    F --> G[Generated Keyframes]
    G --> H[FLF2V Model]
    H --> I[Parallel Processing]
    
    %% Chaining Mode Path
    D -->|Chaining| J[Initial Frame]
    J --> K[I2V Processing]
    K --> L[Video Segment]
    L --> M[Extract Last Frame]
    M --> N[Next Segment]
    N --> K
    
    %% Both paths converge at final video creation
    I --> O[Video Segments]
    L --> O
    O --> P[ffmpeg Concatenation]
    P --> Q[Final Video]
```


## Configuration-Driven Architecture

The entire system behavior is controlled through a comprehensive YAML configuration system that drives backend selection, generation parameters, and processing options.

### Key Configuration Parameters

**Backend Selection:**
- `default_backend`: Choose from `"wan2.1"`, `"runway"`, `"veo3"`, `"minimax"` - [`generators/factory.py`](../generators/factory.py)

**Backend-Specific Settings:**
- `wan2_dir`: Local Wan2.1 setup - [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py)
- `runway_ml`: Runway API configuration - [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py)
- `google_veo`: Veo3 API configuration - [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py)
- `minimax`: Minimax API configuration - [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py)

**Generation Control:**
- `generation_mode`: Choose `"keyframe"` or `"chaining"`
- `parallel_segments`: Enable parallel processing

### Configuration Loading and Processing

**Configuration System Implementation:**
- **Main Loading**: [`pipeline.py`](../pipeline.py)
- **Sample Configuration**: [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample)
- **Factory Processing**: [`generators/factory.py`](../generators/factory.py)

## Key Architectural Patterns

The system employs several key design patterns that enable flexibility, maintainability, and extensibility:

1. **Factory Pattern**: Abstracts backend creation and selection
2. **Interface Segregation**: Clear contracts for all video generators
3. **Configuration-Driven Design**: Runtime behavior controlled by YAML
4. **Fallback Mechanisms**: Automatic backend switching on failure
5. **Modular Architecture**: Clear separation of concerns across components

---

## Next Steps

- **Setup**: See [Getting Started](02-getting-started.md) for installation
- **Configuration**: See [Video Generation Backends](04-video-generation-backends.md) for backend setup
- **Architecture Details**: See [Architecture and Interface](05-architecture-and-interface.md) for implementation details
- **Deployment**: See [Deployment and Containers](08-deployment-and-containers.md) for production setup
