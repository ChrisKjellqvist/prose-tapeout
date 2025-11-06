#!/usr/bin/env python3
"""
ARM Executable Analyzer
Analyzes ARM executables using arm-none-eabi toolchain to extract:
1. Size of each section
2. Size of the top 50 functions
"""

import subprocess
import sys
import re
from typing import List, Tuple, Dict
from pathlib import Path


def check_toolchain():
    """Check if arm-none-eabi tools are available."""
    tools = ['arm-none-eabi-size', 'arm-none-eabi-nm', 'arm-none-eabi-objdump']
    missing = []
    
    for tool in tools:
        try:
            subprocess.run([tool, '--version'], 
                         capture_output=True, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
    
    if missing:
        print(f"Error: Missing ARM toolchain utilities: {', '.join(missing)}", 
              file=sys.stderr)
        print("Please install arm-none-eabi toolchain", file=sys.stderr)
        return False
    return True


def get_section_sizes(executable: str) -> Dict[str, int]:
    """
    Get the size of each section using arm-none-eabi-size.
    
    Args:
        executable: Path to the executable file
        
    Returns:
        Dictionary mapping section names to their sizes in bytes
    """
    try:
        # Use -A flag for SysV format which lists all sections
        result = subprocess.run(
            ['arm-none-eabi-size', '-A', executable],
            capture_output=True,
            text=True,
            check=True
        )
        
        sections = {}
        lines = result.stdout.strip().split('\n')
        
        # Skip header lines
        for line in lines[2:]:  # First line is filename, second is header
            parts = line.split()
            if len(parts) >= 2:
                section_name = parts[0]
                try:
                    size = int(parts[1])
                    sections[section_name] = size
                except (ValueError, IndexError):
                    continue
        
        return sections
        
    except subprocess.CalledProcessError as e:
        print(f"Error running arm-none-eabi-size: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return {}


def get_function_sizes(executable: str) -> List[Tuple[str, int]]:
    """
    Get function sizes using arm-none-eabi-nm.
    
    Args:
        executable: Path to the executable file
        
    Returns:
        List of tuples (function_name, size_in_bytes) sorted by size descending
    """
    try:
        # Use -S to print size, -l for line numbers (helps with identification)
        # Use --size-sort to sort by size
        result = subprocess.run(
            ['arm-none-eabi-nm', '-S', '--size-sort', '-C', executable],
            capture_output=True,
            text=True,
            check=True
        )
        
        functions = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            # Format: address size type name
            # Example: 08000194 00000020 T main
            parts = line.split()
            if len(parts) >= 4:
                try:
                    address = parts[0]
                    size = int(parts[1], 16)  # Size is in hex
                    symbol_type = parts[2]
                    symbol_name = ' '.join(parts[3:])  # Handle names with spaces
                    
                    # Filter for function symbols (T, t, W, w)
                    # T/t = text section (code), W/w = weak symbols
                    if symbol_type in ['T', 't', 'W', 'w'] and size > 0:
                        functions.append((symbol_name, size))
                except (ValueError, IndexError):
                    continue
        
        # Sort by size descending
        functions.sort(key=lambda x: x[1], reverse=True)
        
        return functions
        
    except subprocess.CalledProcessError as e:
        print(f"Error running arm-none-eabi-nm: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return []


def format_size(size_bytes: int) -> str:
    """Format size in bytes with human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB ({size_bytes} B)"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB ({size_bytes} B)"


def print_section_sizes(sections: Dict[str, int]):
    """Print section sizes in a formatted table."""
    if not sections:
        print("No sections found.")
        return
    
    print("\n" + "="*70)
    print("SECTION SIZES")
    print("="*70)
    print(f"{'Section Name':<30} {'Size (bytes)':>15} {'Size':>20}")
    print("-"*70)
    
    total_size = 0
    # Sort sections by size descending
    sorted_sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
    
    for section_name, size in sorted_sections:
        total_size += size
        print(f"{section_name:<30} {size:>15,} {format_size(size):>20}")
    
    print("-"*70)
    print(f"{'TOTAL':<30} {total_size:>15,} {format_size(total_size):>20}")
    print("="*70)


def print_function_sizes(functions: List[Tuple[str, int]], top_n: int = 50):
    """Print top N function sizes in a formatted table."""
    if not functions:
        print("\nNo functions found.")
        return
    
    print("\n" + "="*90)
    print(f"TOP {min(top_n, len(functions))} FUNCTIONS BY SIZE")
    print("="*90)
    print(f"{'Rank':<6} {'Function Name':<50} {'Size (bytes)':>15} {'Size':>15}")
    print("-"*90)
    
    for rank, (func_name, size) in enumerate(functions[:top_n], 1):
        # Truncate long function names
        display_name = func_name if len(func_name) <= 50 else func_name[:47] + "..."
        print(f"{rank:<6} {display_name:<50} {size:>15,} {format_size(size):>15}")
    
    print("="*90)
    
    if len(functions) > top_n:
        remaining = len(functions) - top_n
        total_remaining = sum(size for _, size in functions[top_n:])
        print(f"\n{remaining} additional functions totaling {format_size(total_remaining)}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <executable>")
        print("\nAnalyzes an ARM executable and reports:")
        print("  1. Size of each section in bytes")
        print("  2. Size of the top 50 functions in bytes")
        sys.exit(1)
    
    executable = sys.argv[1]
    
    # Check if file exists
    if not Path(executable).is_file():
        print(f"Error: File '{executable}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Check if toolchain is available
    if not check_toolchain():
        sys.exit(1)
    
    print(f"\nAnalyzing: {executable}")
    
    # Get and display section sizes
    sections = get_section_sizes(executable)
    print_section_sizes(sections)
    
    # Get and display function sizes
    functions = get_function_sizes(executable)
    print_function_sizes(functions, top_n=50)
    
    print()


if __name__ == '__main__':
    main()
