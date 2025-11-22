# Multi-Provider LLM Support - Enhancement Guide

## Overview
The ATLAS Academic Task Learning Agent System now supports multiple LLM providers with OpenAI as the default. This enhancement makes the system more flexible and accessible.

## Supported Providers

### 1. OpenAI (Default)
- **Models**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **API Endpoint**: https://api.openai.com/v1
- **Default Model**: `gpt-4`

### 2. NVIDIA
- **Model**: nemotron-4-340b-instruct
- **API Endpoint**: https://integrate.api.nvidia.com/v1

## Key Changes

### 1. Enhanced `LLMConfig` Class
- Now accepts a `provider` parameter
- Automatically configures base URL and model based on provider
- Default provider is OpenAI

```python
config = LLMConfig(provider="openai")  # Default
config = LLMConfig(provider="nvidia")  # NVIDIA
```

### 2. New `LLMClient` Class
- Renamed from `NeMoLLaMa` for better clarity
- Supports multiple providers through unified interface
- Backward compatible (NeMoLLaMa is an alias)

```python
# OpenAI (default)
llm = LLMClient(api_key="your-openai-key")

# NVIDIA
llm = LLMClient(api_key="your-nvidia-key", provider="nvidia")
```

### 3. Convenience Function: `create_llm_client()`
- Automatically detects API keys from environment
- Simplifies client creation
- Recommended approach

```python
# Auto-detect from environment
llm = create_llm_client()  # Uses OpenAI
llm = create_llm_client(provider="nvidia")  # Uses NVIDIA
```

### 4. Updated `run_all_system()` Function
- New parameters: `api_key`, `provider`
- Automatic API key detection from environment
- Defaults to OpenAI

```python
await run_all_system(
    profile_json,
    calendar_json,
    task_json,
    provider="openai"  # or "nvidia"
)
```

### 5. Updated `load_json_and_test()` Function
- New parameters: `api_key`, `provider`
- Pass-through to `run_all_system()`

## Usage Examples

### Quick Start with OpenAI (Default)

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."
```

```python
# Run the system
coordinator_output, output = load_json_and_test(
    profile_path="data/profile.json",
    calendar_path="data/calendar.json",
    task_path="data/task.json"
)
```

### Using NVIDIA

```bash
# Set environment variable
export NVIDIA_API_KEY="nvapi-..."
```

```python
coordinator_output, output = load_json_and_test(
    profile_path="data/profile.json",
    calendar_path="data/calendar.json",
    task_path="data/task.json",
    provider="nvidia"
)
```

### Custom API Key

```python
coordinator_output, output = load_json_and_test(
    profile_path="data/profile.json",
    calendar_path="data/calendar.json",
    task_path="data/task.json",
    api_key="your-custom-key",
    provider="openai"
)
```

### Direct LLM Client Usage

```python
# Method 1: Convenience function (recommended)
llm = create_llm_client()  # OpenAI default
llm = create_llm_client(provider="nvidia")

# Method 2: Direct instantiation
llm = LLMClient(api_key=os.getenv("OPENAI_API_KEY"))
llm = LLMClient(api_key=os.getenv("NVIDIA_API_KEY"), provider="nvidia")
```

## Environment Variables

The system checks for these environment variables in order:

1. **OpenAI**:
   - `OPENAI_API_KEY` (environment variable)
   - `OPENAI_API_KEY` (global variable in code)

2. **NVIDIA**:
   - `NVIDIA_API_KEY` (environment variable)
   - `NEMOTRON_4_340B_INSTRUCT_KEY` (global variable in code)

## Backward Compatibility

All existing code using `NeMoLLaMa` will continue to work:

```python
# Old code still works
llm = NeMoLLaMa(NEMOTRON_4_340B_INSTRUCT_KEY)

# Equivalent to
llm = LLMClient(NEMOTRON_4_340B_INSTRUCT_KEY, provider="nvidia")
```

## Error Handling

The system provides clear error messages when API keys are missing:

```
Error: No API key found for openai
Set OPENAI_API_KEY environment variable or pass api_key parameter
```

## Model Configuration

You can modify the default models in `LLMConfig`:

```python
class LLMConfig:
    def __init__(self, provider: str = "openai"):
        # ...
        elif self.provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-4"  # Change to "gpt-4-turbo" or "gpt-3.5-turbo"
```

## Benefits

1. **Flexibility**: Easy to switch between providers
2. **Default to OpenAI**: More widely accessible
3. **Backward Compatible**: Existing NVIDIA code still works
4. **Better Documentation**: Clear usage examples
5. **Environment-Aware**: Auto-detects API keys
6. **Error Handling**: Clear messages for missing configuration

## Migration Guide

### From NVIDIA-only to Multi-Provider

**Before:**
```python
llm = NeMoLLaMa(NEMOTRON_4_340B_INSTRUCT_KEY)
```

**After (Using NVIDIA):**
```python
llm = LLMClient(NEMOTRON_4_340B_INSTRUCT_KEY, provider="nvidia")
# or
llm = create_llm_client(provider="nvidia")
```

**After (Using OpenAI - New Default):**
```python
llm = LLMClient(OPENAI_API_KEY)
# or
llm = create_llm_client()  # Uses OpenAI by default
```

## Testing

Test both providers to ensure functionality:

```python
# Test OpenAI
output = load_json_and_test(
    profile_path="data/profile.json",
    calendar_path="data/calendar.json",
    task_path="data/task.json",
    provider="openai"
)

# Test NVIDIA
output = load_json_and_test(
    profile_path="data/profile.json",
    calendar_path="data/calendar.json",
    task_path="data/task.json",
    provider="nvidia"
)
```

## Additional Features

### Graph Visualization
The system now saves workflow graphs to files instead of requiring Jupyter:

```python
# Graph saved automatically as workflow_graph.png
# Can be opened with any image viewer
```

## Support

For issues or questions:
1. Check environment variables are set correctly
2. Verify API keys are valid
3. Ensure correct provider name ("openai" or "nvidia")
4. Review error messages for specific guidance
