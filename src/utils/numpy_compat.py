"""
NumPy Compatibility Layer for HeraAI

This module provides comprehensive compatibility between NumPy 2.0+ and ChromaDB
by restoring deprecated attributes and patching compatibility issues.
"""

import numpy as np
import warnings
import sys

def setup_numpy_compatibility():
    """
    Setup comprehensive NumPy compatibility for ChromaDB and other libraries
    that haven't updated to NumPy 2.0 yet.
    """
    # Suppress all NumPy 2.0 deprecation warnings
    warnings.filterwarnings("ignore", message=".*np\\.float_.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*np\\.int_.*deprecated.*") 
    warnings.filterwarnings("ignore", message=".*np\\.uint.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*np\\.complex_.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*np\\.bool_.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*was removed in the NumPy 2.0 release.*")
    
    # Add back all deprecated attributes as aliases for compatibility
    compatibility_map = {
        'float_': np.float64,
        'int_': np.int64,
        'uint': np.uint64,
        'complex_': np.complex128,
        'bool_': np.bool_,
        'long': np.int64,
        'unicode_': np.str_,
        'string_': np.str_,
        'object_': np.object_,
        'void': np.void,
        # Additional compatibility aliases
        'float': np.float64,
        'int': np.int64,
        'complex': np.complex128,
        'bool': np.bool_,
    }
    
    for old_name, new_type in compatibility_map.items():
        if not hasattr(np, old_name):
            try:
                setattr(np, old_name, new_type)
            except Exception:
                pass  # Some attributes might be read-only

def patch_chromadb_types():
    """
    Comprehensive monkey patch for ChromaDB compatibility.
    This pre-patches the problematic modules before they're imported.
    """
    try:
        import types
        from typing import Union
        
        # Patch chromadb.api.types before it's imported
        if 'chromadb.api.types' not in sys.modules:
            fake_types = types.ModuleType('chromadb.api.types')
            
            # Add all the type definitions that ChromaDB needs
            fake_types.ImageDType = Union[np.uint64, np.int64, np.float64]
            fake_types.NDArray = np.ndarray
            fake_types.OneOrMany = Union[list, tuple]
            
            # Add other common ChromaDB type definitions
            fake_types.Embedding = list
            fake_types.Embeddings = list
            fake_types.Document = str
            fake_types.Documents = list
            fake_types.ID = str
            fake_types.IDs = list
            fake_types.Metadata = dict
            fake_types.Metadatas = list
            fake_types.Distance = float
            fake_types.Distances = list
            
            sys.modules['chromadb.api.types'] = fake_types
            
    except Exception as e:
        print(f"Warning: ChromaDB pre-patching failed: {e}")

def monkey_patch_numpy_getattr():
    """
    Monkey patch numpy's __getattr__ to handle missing attributes gracefully.
    """
    try:
        original_getattr = getattr(np, '__getattr__', None)
        
        def patched_getattr(name):
            # Handle the specific attributes that were removed
            if name == 'float_':
                return np.float64
            elif name == 'int_':
                return np.int64
            elif name == 'uint':
                return np.uint64
            elif name == 'complex_':
                return np.complex128
            elif name == 'bool_':
                return np.bool_
            elif name in ['long', 'unicode_', 'string_']:
                return getattr(np, {'long': 'int64', 'unicode_': 'str_', 'string_': 'str_'}[name])
            
            # Fall back to original behavior
            if original_getattr:
                return original_getattr(name)
            else:
                raise AttributeError(f"module 'numpy' has no attribute '{name}'")
        
        # Only patch if we're using NumPy 2.0+
        if hasattr(np, '__version__') and np.__version__.startswith('2.'):
            np.__getattr__ = patched_getattr
            
    except Exception as e:
        print(f"Warning: NumPy __getattr__ patching failed: {e}")

# Initialize compatibility when module is imported
setup_numpy_compatibility()
monkey_patch_numpy_getattr()

print("🔧 NumPy 2.0 compatibility layer initialized") 