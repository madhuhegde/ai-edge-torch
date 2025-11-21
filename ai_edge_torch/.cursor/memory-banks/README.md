# Memory Banks

This directory contains organized documentation for features, implementations, and concepts in AI Edge Torch.

## What are Memory Banks?

Memory banks are dedicated folders containing comprehensive markdown documentation for specific features or implementations. They help:
- Organize related documentation in one place
- Make documentation easy to find and maintain
- Provide context and examples for complex features
- Serve as learning resources

## Available Memory Banks

### 📁 [custom_rms_norm/](./custom_rms_norm/)
**Custom RMS Normalization Operator**

Implementation of RMS normalization as a custom operator using `stablehlo.custom_call`:
- Overview and introduction
- Usage guide with examples
- Detailed implementation
- Current limitations and workarounds

**Status**: ✅ Implemented (4 files)

**Quick Links**:
- [Overview](./custom_rms_norm/overview.md) - What is custom RMS norm
- [Usage Guide](./custom_rms_norm/usage.md) - How to use
- [Memory Bank Index](./custom_rms_norm/README.md) - Full navigation

**Note**: ⚠️ TFLite conversion currently blocked by VHLO serialization

## Structure

```
.cursor/memory-banks/
├── README.md (this file)
└── custom_rms_norm/          # Custom RMS norm operator
    ├── README.md              # Navigation index
    ├── overview.md            # Introduction and concepts
    ├── usage.md               # Usage guide and examples
    ├── implementation.md      # Technical implementation
    └── limitations.md         # Current limitations
```

## Creating New Memory Banks

Follow the [documentation rule](./../rules/documentation.mdc):

```bash
# 1. Create memory bank folder
mkdir -p .cursor/memory-banks/feature-name/

# 2. Create documentation files
touch .cursor/memory-banks/feature-name/README.md
touch .cursor/memory-banks/feature-name/overview.md
touch .cursor/memory-banks/feature-name/usage.md
```

## Documentation Standards

Each memory bank should include:
- ✅ **README.md** - Navigation and quick links
- ✅ **overview.md** - High-level introduction
- ✅ **usage.md** or **examples.md** - How to use
- ✅ Markdown format (.md) for all files
- ✅ Clear titles and table of contents
- ✅ Code examples with syntax highlighting
- ✅ Cross-references to related docs

## See Also

- [Documentation Rule](../rules/documentation.mdc) - How to create memory banks
- [Python Environment Rule](../rules/python-env.mdc) - Python setup
- [Cursor Rules](../rules/cursor-rules.mdc) - Rule creation guide

---

**Location**: `.cursor/memory-banks/`
**Purpose**: Organized documentation for complex features and implementations
