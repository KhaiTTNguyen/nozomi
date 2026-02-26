#!/usr/bin/env python3
"""
Test minimal requirements by trying to import all used modules
"""

import sys
import importlib

# Core modules your code actually uses (from analysis)
REQUIRED_MODULES = [
    'numpy',
    'scipy', 
    'pandas',
    'matplotlib',
    'seaborn',
    'torch',
    'pycuda',
    'dipy',
    'pytest'
]

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing minimal requirements...")
    failed = []
    
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed imports: {failed}")
        return False
    else:
        print(f"\n✅ All {len(REQUIRED_MODULES)} core modules installed!")
        return True

def test_pycuda_functionality():
    """Test basic PyCUDA functionality."""
    try:
        import pycuda.autoinit
        import pycuda.driver as drv
        import pycuda.gpuarray as gpuarray
        
        # Test basic GPU operations
        gpu_count = drv.Device.count()
        print(f"✅ PyCUDA: {gpu_count} GPU(s) available")
        
        if gpu_count > 0:
            # Test basic GPU array creation
            test_array = gpuarray.zeros(100, dtype=float)
            print("✅ PyCUDA array creation works")
        
        return True
    except Exception as e:
        print(f"❌ PyCUDA test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TESTING MINIMAL REQUIREMENTS")
    print("=" * 40)
    
    imports_ok = test_imports()
    print()
    
    if imports_ok:
        cuda_ok = test_pycuda_functionality()
        
        if cuda_ok:
            print("\nSUCCESS: Minimal requirements installed!")
            return 0
        else:
            print("\n⚠️  WARNING: Core imports work but PyCUDA has issues")
            return 1
    else:
        print("\n❌ FAILURE: Missing required modules")
        return 2

if __name__ == "__main__":
    sys.exit(main())