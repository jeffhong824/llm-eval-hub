"""
Pydantic v1 compatibility patch for Python 3.12.

This module patches pydantic v1's evaluate_forwardref function to work with Python 3.12,
which changed the signature of ForwardRef._evaluate() to require a recursive_guard parameter.

This patch must be imported BEFORE any langchain imports.
"""

import os

# Set environment variables to disable LangSmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")

# Monkey patch pydantic v1's evaluate_forwardref BEFORE any langchain imports
def patch_pydantic_v1():
    """Patch pydantic v1's evaluate_forwardref for Python 3.12 compatibility."""
    try:
        import pydantic.v1.typing as pydantic_v1_typing
        
        # Check if already patched
        if hasattr(pydantic_v1_typing.evaluate_forwardref, '_patched'):
            return
        
        # Patch evaluate_forwardref function to handle Python 3.12's ForwardRef._evaluate signature
        # Python 3.12 changed ForwardRef._evaluate to require recursive_guard as keyword-only arg
        original_evaluate_forwardref = pydantic_v1_typing.evaluate_forwardref
        
        def patched_evaluate_forwardref(type_, globalns, localns=None):
            """Patched version that handles Python 3.12's ForwardRef._evaluate signature."""
            # The issue is that pydantic v1 calls type_._evaluate(globalns, localns, set())
            # but Python 3.12 requires: type_._evaluate(globalns, localns, set(), recursive_guard=set())
            # We need to intercept the call and add recursive_guard
            try:
                # Try original first
                return original_evaluate_forwardref(type_, globalns, localns)
            except TypeError:
                # If it fails, manually call _evaluate with recursive_guard
                if hasattr(type_, '_evaluate'):
                    try:
                        # Python 3.12 signature: _evaluate(globalns, localns, type_params=None, *, recursive_guard)
                        return type_._evaluate(globalns, localns, set(), recursive_guard=set())
                    except (TypeError, AttributeError):
                        # Fallback: try without recursive_guard (older Python)
                        return type_._evaluate(globalns, localns, set())
                raise
        
        # Mark as patched
        patched_evaluate_forwardref._patched = True
        pydantic_v1_typing.evaluate_forwardref = patched_evaluate_forwardref
    except (ImportError, AttributeError):
        pass  # pydantic v1 not available or already patched

# Apply patch immediately when this module is imported
patch_pydantic_v1()
