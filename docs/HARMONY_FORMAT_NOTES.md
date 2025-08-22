# Harmony Format Analysis Notes

## Overview

The red-teaming framework includes sophisticated parsing of the Harmony format used by the gpt-oss-20b model. This document explains the format analysis and common "warnings" that appear during execution.

## Harmony Format Structure

The gpt-oss-20b model outputs responses in a structured format:

```
<|start|>assistant<|channel|>analysis<|message|>Internal reasoning content...<|end|>
<|start|>assistant<|channel|>final<|message|>Final response to user<|return|>
```

### Format Components

- `<|start|>` - Begins a new section
- `assistant` - Indicates this is an assistant response
- `<|channel|>` - Separates role from channel type
- `analysis|thinking|commentary|final` - The channel type
- `<|message|>` - Begins the actual content
- `<|end|>|<|return|>` - Ends the section

## Format Health Analysis

The framework automatically analyzes response format health and may report:

### "Malformed" Responses

**Note**: The framework may report responses as "malformed" even when they're working correctly. This is intentional conservative analysis.

Common "issues" that are actually normal:

1. **missing_end_tokens**: Previously overcounted due to not recognizing `<|return|>` as valid ending
   - ✅ **FIXED**: Now properly counts both `<|end|>` and `<|return|>` as valid endings

2. **corrupted_channel_?analysis**: Slight variations in channel names
   - Example: `?analysis` instead of `analysis`
   - These are usually minor tokenization artifacts and don't break functionality

3. **mixed_format_patterns**: Different format variations in same response
   - Often occurs when model switches between thinking and output sections
   - Usually indicates rich internal reasoning, not actual errors

### Format Health Rates

Typical healthy ranges:
- **Thinking content detection**: 70-95% (good)
- **Complete format rate**: 60-80% (acceptable)
- **Malformed rate**: 10-30% (normal with conservative parsing)

High malformed rates (>50%) may indicate actual issues, but rates of 20-30% are typical and don't affect functionality.

## Performance Impact

Format parsing is diagnostic only and doesn't affect:
- Model response generation
- Scoring and evaluation
- Export functionality
- Core red-teaming capabilities

## Troubleshooting

### If you see 100% malformed rate:
1. ✅ Check that both `<|end|>` and `<|return|>` are counted as valid endings (fixed in latest version)
2. Verify model is using Harmony chat template: `cfg.model.use_harmony_chat_template = True`
3. Check sample responses in debug output for actual format structure

### If responses are empty or scoring fails:
1. Verify model is loaded correctly
2. Check that judge system can parse responses
3. Review actual response content in debug output

## Configuration

Harmony format analysis can be configured:

```python
# Enable/disable enhanced format analysis
cfg.judge.use_enhanced_judge = True

# Weight for harmony-specific scoring
cfg.judge.harmony_weight = 0.2

# Threshold for flagging issues
cfg.judge.threshold = 0.6
```

## Summary

The "malformed" warnings are primarily diagnostic and indicate the framework's conservative approach to format validation. The core red-teaming functionality works correctly even with these warnings present.

**Key Point**: A response can be flagged as "malformed" by the analysis system while still being perfectly functional for scoring, evaluation, and export purposes.