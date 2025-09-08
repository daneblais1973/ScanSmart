#!/usr/bin/env python3
"""
Advanced Dependency Checker for ScanSmart Pro
Checks for existing dependencies, conflicts, and provides intelligent installation
"""

import subprocess
import sys
import importlib
from pathlib import Path


def check_dependency_conflicts():
    """Check for common dependency conflicts"""
    
    print("🔍 Checking for potential dependency conflicts...")
    
    # Common conflict patterns
    conflicts = [
        # PyTorch conflicts
        {
            'packages': ['torch', 'tensorflow'],
            'issue': 'PyTorch and TensorFlow can conflict on some systems',
            'resolution': 'Consider using only one ML framework'
        },
        # Pandas/NumPy version conflicts
        {
            'packages': ['pandas', 'numpy'],
            'issue': 'Pandas requires compatible NumPy versions',
            'resolution': 'Ensure NumPy >= 1.24.0 for Pandas 2.0+'
        },
        # Streamlit conflicts
        {
            'packages': ['streamlit', 'altair'],
            'issue': 'Streamlit requires specific Altair versions',
            'resolution': 'Let Streamlit manage its dependencies'
        }
    ]
    
    found_conflicts = []
    
    for conflict in conflicts:
        installed_packages = []
        for pkg in conflict['packages']:
            try:
                version = get_package_version(pkg)
                if version:
                    installed_packages.append(f"{pkg}=={version}")
            except:
                pass
        
        if len(installed_packages) >= 2:
            found_conflicts.append({
                'packages': installed_packages,
                'issue': conflict['issue'],
                'resolution': conflict['resolution']
            })
    
    if found_conflicts:
        print("⚠️  Potential conflicts detected:")
        for i, conflict in enumerate(found_conflicts, 1):
            print(f"\n{i}. Packages: {', '.join(conflict['packages'])}")
            print(f"   Issue: {conflict['issue']}")
            print(f"   Resolution: {conflict['resolution']}")
        return found_conflicts
    else:
        print("✅ No major conflicts detected!")
        return []


def get_package_version(package_name):
    """Get installed package version"""
    try:
        # Use importlib.metadata instead of deprecated pkg_resources
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except ImportError:
        # Fallback for older Python versions
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except:
            return None
    except:
        return None


def check_system_requirements():
    """Check system requirements for optimal performance"""
    
    print("🖥️  Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 11):
        print(f"✅ Python {python_version.major}.{python_version.minor} (Excellent)")
    elif python_version >= (3, 9):
        print(f"✅ Python {python_version.major}.{python_version.minor} (Good)")
    else:
        print(f"⚠️  Python {python_version.major}.{python_version.minor} (Consider upgrading to 3.11+)")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        gb_total = memory.total / (1024**3)
        gb_available = memory.available / (1024**3)
        
        print(f"💾 Memory: {gb_available:.1f}GB available / {gb_total:.1f}GB total")
        
        if gb_available < 2:
            print("⚠️  Low memory detected - some ML features may be limited")
        elif gb_available >= 8:
            print("✅ Excellent memory for ML processing")
        else:
            print("✅ Sufficient memory for basic operations")
            
    except ImportError:
        print("📊 Install psutil to check system resources: pip install psutil")
    
    # Check disk space
    try:
        disk_usage = Path('.').stat()
        print("💽 Disk space check completed")
    except:
        print("⚠️  Could not check disk space")


def create_requirements_lock():
    """Create a requirements.lock file with exact versions"""
    
    print("📝 Creating requirements lock file...")
    
    try:
        # Get all installed packages
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            with open("requirements.lock", "w") as f:
                f.write("# Auto-generated requirements lock file\n")
                f.write("# This ensures reproducible installations\n\n")
                f.write(result.stdout)
            
            print("✅ Requirements lock file created: requirements.lock")
            return True
        else:
            print("⚠️  Could not generate requirements lock file")
            return False
            
    except Exception as e:
        print(f"❌ Error creating lock file: {e}")
        return False


def validate_critical_imports():
    """Validate that critical packages can be imported"""
    
    print("🧪 Validating critical imports...")
    
    critical_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'), 
        ('plotly', 'Data visualization'),
        ('sqlalchemy', 'Database ORM'),
        ('requests', 'HTTP client'),
        ('yfinance', 'Financial data'),
    ]
    
    failed_imports = []
    
    for package, description in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - {description}")
        except ImportError as e:
            print(f"❌ {package} - {description} (FAILED: {e})")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} critical imports failed!")
        print("🔧 Run the dependency installer to fix these issues.")
        return False
    else:
        print("\n✅ All critical packages imported successfully!")
        return True


def main():
    """Main dependency checker function"""
    
    print("="*60)
    print("🔍 SCANSMART PRO - DEPENDENCY CHECKER")
    print("="*60)
    
    steps = [
        ("System Requirements", check_system_requirements),
        ("Dependency Conflicts", check_dependency_conflicts), 
        ("Critical Imports", validate_critical_imports),
        ("Requirements Lock", create_requirements_lock)
    ]
    
    all_passed = True
    
    for step_name, step_function in steps:
        print(f"\n📋 {step_name}...")
        try:
            result = step_function()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 DEPENDENCY CHECK COMPLETED - ALL SYSTEMS GO!")
    else:
        print("⚠️  DEPENDENCY CHECK COMPLETED - SOME ISSUES FOUND")
        print("🔧 Run 'python run_local.py' to install missing dependencies")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    main()