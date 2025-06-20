"""
Main entry point for LiDAR Particle Filter.

This module provides a simple demonstration of the particle filter
functionality when run as a script.
"""

from examples.basic_usage import main as run_example


def main():
    """Main entry point."""
    print("LiDAR Particle Filter")
    print("=====================")
    print("Running basic usage example...")
    print()
    
    try:
        run_example()
    except Exception as e:
        print(f"Error running example: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install numpy numba")


if __name__ == "__main__":
    main()
