#!/usr/bin/env python3
"""
Response validation and error handling for red-teaming framework.

This module provides comprehensive validation for model responses to handle
malformed/corrupted text and improve overall framework reliability.
"""

import re
import unicodedata
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Comprehensive response validator with quality scoring and error recovery."""
    
    def __init__(self, min_response_length: int = 10, max_response_length: int = 10000):
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.encoding_errors = 0
        self.corruption_patterns = self._compile_corruption_patterns()
        
    def _compile_corruption_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for detecting corrupted responses."""
        patterns = [
            # Excessive repetition of special characters
            re.compile(r'[^\w\s]{10,}'),
            # Unicode replacement characters indicating encoding issues
            re.compile(r'[\ufffd]{2,}'),
            # Excessive whitespace or newlines
            re.compile(r'\s{50,}'),
            # HTML/XML artifacts that shouldn't be in model responses
            re.compile(r'<[^>]{100,}>'),
            # Excessive punctuation repetition
            re.compile(r'[.!?]{10,}'),
            # Binary-like patterns
            re.compile(r'[01]{50,}'),
            # Garbled text with many special characters
            re.compile(r'[^\w\s,.!?\'";:()\-]{20,}'),
        ]
        return patterns
        
    def validate_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive response validation with detailed quality assessment
        
        Args:
            response: The model response to validate
            context: Optional context information (prompt, model settings, etc.)
            
        Returns:
            Dict containing validation results:
            - is_valid: bool - Whether response passes validation
            - quality_score: float - Quality score [0.0, 1.0]
            - issues: List[str] - List of detected issues
            - corrected_response: str - Cleaned/corrected version if possible
            - metadata: Dict - Additional validation metadata
        """
        if not isinstance(response, str):
            return self._create_validation_result(
                is_valid=False,
                quality_score=0.0,
                issues=["invalid_type"],
                corrected_response="",
                original_response=response
            )
            
        # Initialize validation result
        issues = []
        quality_score = 1.0
        corrected_response = response
        
        # Length validation
        if len(response) < self.min_response_length:
            issues.append("response_too_short")
            quality_score *= 0.3
            
        elif len(response) > self.max_response_length:
            issues.append("response_too_long")
            quality_score *= 0.7
            corrected_response = self._truncate_response_safely(response)
            
        # Encoding validation
        try:
            response.encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append("encoding_error")
            quality_score *= 0.2
            corrected_response = self._fix_encoding_errors(response)
            self.encoding_errors += 1
            logger.warning(f"Encoding error in response: {e}")
            
        # Corruption detection
        corruption_score, corruption_issues = self._detect_corruption(response)
        issues.extend(corruption_issues)
        quality_score *= corruption_score
        
        if corruption_score < 0.5:
            corrected_response = self._attempt_corruption_recovery(response)
            
        # Content quality assessment
        content_score, content_issues = self._assess_content_quality(response)
        issues.extend(content_issues)
        quality_score *= content_score
        
        # Determine overall validity
        is_valid = (quality_score > 0.3 and 
                   len(corrected_response) >= self.min_response_length and
                   "critical_corruption" not in issues)
        
        return self._create_validation_result(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            corrected_response=corrected_response,
            original_response=response,
            context=context
        )
        
    def _detect_corruption(self, response: str) -> tuple[float, List[str]]:
        """Detect various forms of text corruption
        
        Returns:
            tuple: (corruption_score, issues_list)
            corruption_score: 1.0 = no corruption, 0.0 = severely corrupted
        """
        issues = []
        corruption_score = 1.0
        
        # Check against corruption patterns
        for pattern in self.corruption_patterns:
            matches = pattern.findall(response)
            if matches:
                issues.append("pattern_corruption")
                corruption_score *= 0.7
                
        # Check character diversity (too many repeated chars indicates corruption)
        char_diversity = len(set(response)) / max(1, len(response))
        if char_diversity < 0.05:  # Very low diversity
            issues.append("low_character_diversity")
            corruption_score *= 0.4
            
        # Check for excessive non-printable characters
        non_printable_count = sum(1 for c in response if not c.isprintable() and c not in '\n\t\r')
        if non_printable_count > len(response) * 0.1:
            issues.append("excessive_non_printable")
            corruption_score *= 0.3
            
        # Check for garbled Unicode
        try:
            normalized = unicodedata.normalize('NFKC', response)
            if len(normalized) < len(response) * 0.5:
                issues.append("unicode_corruption")
                corruption_score *= 0.5
        except Exception:
            issues.append("unicode_normalization_error")
            corruption_score *= 0.2
            
        # Critical corruption check
        if corruption_score < 0.2:
            issues.append("critical_corruption")
            
        return corruption_score, issues
        
    def _assess_content_quality(self, response: str) -> tuple[float, List[str]]:
        """Assess the content quality of the response
        
        Returns:
            tuple: (content_score, issues_list)
        """
        issues = []
        content_score = 1.0
        
        # Check for meaningful content (not just repetitive text)
        words = response.split()
        if not words:
            issues.append("no_words")
            return 0.0, issues
            
        unique_words = set(words)
        word_diversity = len(unique_words) / len(words)
        
        if word_diversity < 0.1:  # Very repetitive
            issues.append("highly_repetitive")
            content_score *= 0.4
        elif word_diversity < 0.3:
            issues.append("somewhat_repetitive")
            content_score *= 0.7
            
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s for s in sentences if len(s.strip()) > 5]
        
        if not valid_sentences:
            issues.append("no_valid_sentences")
            content_score *= 0.2
        elif len(valid_sentences) == 1 and len(words) > 50:
            issues.append("run_on_sentence")
            content_score *= 0.8
            
        return content_score, issues
        
    def _truncate_response_safely(self, response: str) -> str:
        """Truncate response while preserving sentence boundaries."""
        if len(response) <= self.max_response_length:
            return response
            
        # Try to truncate at sentence boundary
        sentences = re.split(r'([.!?]+)', response[:self.max_response_length])
        if len(sentences) > 2:
            # Keep complete sentences
            truncated = ''.join(sentences[:-2])
            if len(truncated) > self.min_response_length:
                return truncated + "..."
                
        # Fallback to word boundary
        words = response[:self.max_response_length].split()
        return ' '.join(words[:-1]) + "..."
        
    def _fix_encoding_errors(self, response: str) -> str:
        """Attempt to fix encoding errors in the response."""
        try:
            # Try common encoding fixes
            fixed = response.encode('utf-8', errors='ignore').decode('utf-8')
            return fixed
        except Exception:
            # Last resort: remove problematic characters
            return ''.join(c if ord(c) < 128 else '' for c in response)
            
    def _attempt_corruption_recovery(self, response: str) -> str:
        """Attempt to recover from text corruption."""
        # Remove excessive special character patterns
        recovered = response
        for pattern in self.corruption_patterns:
            recovered = pattern.sub(' ', recovered)
            
        # Clean up whitespace
        recovered = re.sub(r'\s+', ' ', recovered).strip()
        
        # If still too corrupted, return empty
        if len(recovered) < self.min_response_length:
            return ""
            
        return recovered
        
    def _create_validation_result(self, is_valid: bool, quality_score: float, 
                                issues: List[str], corrected_response: str,
                                original_response: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized validation result dictionary."""
        return {
            "is_valid": is_valid,
            "quality_score": round(quality_score, 4),
            "issues": issues,
            "corrected_response": corrected_response,
            "metadata": {
                "original_length": len(str(original_response)),
                "corrected_length": len(corrected_response),
                "issue_count": len(issues),
                "has_critical_issues": any(issue in ["critical_corruption", "encoding_error"] for issue in issues),
                "context": context or {}
            }
        }
        
    def get_validator_stats(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring."""
        return {
            "encoding_errors": self.encoding_errors,
            "validation_thresholds": {
                "min_length": self.min_response_length,
                "max_length": self.max_response_length
            }
        }


def validate_model_response(response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenient function for validating a single response."""
    validator = ResponseValidator()
    return validator.validate_response(response, context)


def create_fallback_response(original_response: str, validation_result: Dict[str, Any]) -> str:
    """Create a safe fallback response when validation fails."""
    if validation_result["is_valid"]:
        return validation_result["corrected_response"]
        
    # For invalid responses, create a descriptive fallback
    issues = validation_result["issues"]
    
    if "response_too_short" in issues:
        return f"[Model response too short: {len(original_response)} characters]"
    elif "critical_corruption" in issues:
        return "[Model response severely corrupted - content unrecoverable]"
    elif "encoding_error" in issues:
        return "[Model response contains encoding errors]"
    else:
        return f"[Model response validation failed: {', '.join(issues[:3])}]"