#!/usr/bin/env python3
"""
Example usage of the improved AST parsing functionality.
This shows practical examples of how to use the enhanced parsing
in your actual workflow.
"""

import sys
import os

# Add the current directory to the path so we can import AST_parsing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Example of basic usage with automatic preprocessing."""
    print("📝 Example 1: Basic usage with automatic preprocessing")
    print("-" * 50)
    
    # Example problematic code (similar to what user encountered)
    problematic_code = """<|im_start|>assistant
def generate_integers(a, b):
    return [i for i in range(a, b + 1)]
Error at call: generate_integers(1, 5)
<|im_end|>"""
    
    print("Code to parse:")
    print(problematic_code)
    
    try:
        # Import the parsing functions (this will work if tree-sitter is installed)
        from AST_parsing import parse_to_ast, get_ast_leaf_nodes
        
        # Parse with automatic preprocessing (default behavior)
        ast_root = parse_to_ast(problematic_code, ps_language="python", preprocess=True, verbose_errors=False)
        
        if ast_root:
            print("✅ Successfully parsed!")
            
            # Extract AST nodes
            nodes = get_ast_leaf_nodes(ast_root)
            print(f"Extracted {len(nodes)} AST nodes")
            
            # Show first few nodes
            print("Sample nodes:")
            for i, node in enumerate(nodes[:5]):
                print(f"  {i+1}. {node['type']}: {node['text']}")
        else:
            print("❌ Parsing failed")
            
    except ImportError:
        print("⚠️  Tree-sitter modules not available, showing preprocessing only:")
        from AST_parsing import preprocess_code_for_parsing
        
        preprocessed = preprocess_code_for_parsing(problematic_code, "python")
        print("Preprocessed code:")
        print(preprocessed)

def example_fallback_strategies():
    """Example of using fallback parsing strategies."""
    print("\n📝 Example 2: Using fallback parsing strategies")
    print("-" * 50)
    
    # Example with multiple issues
    complex_problematic_code = """<|im_start|>user
Write a function to process data
<|im_end|>
<|im_start|>assistant
def process_data(data):
Error at call: process_data([1, 2, 3])
    result = []
    for item in data:
Error at ERROR: item
        result.append(item * 2)
    return result
<|im_end|>
Error at binary_operator: *"""
    
    print("Complex problematic code:")
    print(complex_problematic_code)
    
    try:
        from AST_parsing import parse_to_ast_with_fallback, get_ast_leaf_nodes
        
        # Try fallback parsing
        ast_root, strategy = parse_to_ast_with_fallback(complex_problematic_code, verbose_errors=True)
        
        if ast_root:
            print(f"✅ Successfully parsed using strategy: {strategy}")
            
            # Extract AST nodes
            nodes = get_ast_leaf_nodes(ast_root)
            print(f"Extracted {len(nodes)} AST nodes")
        else:
            print("❌ All parsing strategies failed")
            
    except ImportError:
        print("⚠️  Tree-sitter modules not available, showing preprocessing only:")
        from AST_parsing import preprocess_code_for_parsing
        
        preprocessed = preprocess_code_for_parsing(complex_problematic_code, "python")
        print("Preprocessed code:")
        print(preprocessed)

def example_csv_processing():
    """Example of processing CSV files with improved parsing."""
    print("\n📝 Example 3: CSV processing with improved parsing")
    print("-" * 50)
    
    # Create a sample CSV with problematic data
    import csv
    import tempfile
    
    sample_functions = [
        "def simple_function(): return 42",
        "<|im_start|>assistant\ndef with_tokens(x):\n    return x * 2\n<|im_end|>",
        """def with_errors():
Error at call: some_function()
    return True
Error at ERROR: variable""",
        "def clean_function():\n    return 'Hello World'"
    ]
    
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
        temp_input_path = temp_input.name
        writer = csv.writer(temp_input)
        for func in sample_functions:
            writer.writerow([func])
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_output:
        temp_output_path = temp_output.name
    
    try:
        print(f"Created test CSV with {len(sample_functions)} functions")
        print("Sample functions:")
        for i, func in enumerate(sample_functions, 1):
            preview = func.replace('\n', '\\n')[:60] + "..." if len(func) > 60 else func.replace('\n', '\\n')
            print(f"  {i}. {preview}")
        
        try:
            from AST_parsing import process_functions_from_csv
            
            print(f"\nProcessing CSV with improved parsing...")
            process_functions_from_csv(temp_input_path, temp_output_path, verbose_errors=True, use_fallback=True)
            
            # Show results
            with open(temp_output_path, 'r') as f:
                content = f.read()
                print(f"\nOutput file size: {len(content)} characters")
            
        except ImportError:
            print("⚠️  Tree-sitter modules not available for full CSV processing")
            print("    Install tree-sitter modules to use this functionality")
    
    finally:
        # Clean up
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)

def show_usage_patterns():
    """Show common usage patterns for the improved parsing."""
    print("\n📋 Common usage patterns")
    print("-" * 50)
    
    patterns = [
        ("Basic parsing with preprocessing", 
         "ast_root = parse_to_ast(code, preprocess=True)"),
        
        ("Parsing with verbose error reporting", 
         "ast_root = parse_to_ast(code, verbose_errors=True)"),
        
        ("Robust parsing with fallback strategies", 
         "ast_root, strategy = parse_to_ast_with_fallback(code)"),
        
        ("Preprocessing only (for inspection)", 
         "clean_code = preprocess_code_for_parsing(code, 'python')"),
        
        ("Batch CSV processing with fallback", 
         "process_functions_from_csv(input_file, output_file, use_fallback=True)"),
        
        ("Syntax validation before parsing", 
         "is_valid, issues = validate_code_syntax(code, 'python')"),
    ]
    
    for description, code_example in patterns:
        print(f"• {description}:")
        print(f"  {code_example}")
        print()

if __name__ == "__main__":
    print("🛠️  AST Parsing - Practical Usage Examples")
    print("This script shows how to use the improved AST parsing")
    print("functionality in your actual workflow.\n")
    
    # Run examples
    example_basic_usage()
    example_fallback_strategies()
    example_csv_processing()
    show_usage_patterns()
    
    print("=" * 60)
    print("✅ Examples completed!")
    print("\n🔧 Quick fixes for your issue:")
    print("1. Use parse_to_ast(code, preprocess=True) to automatically handle special tokens")
    print("2. Use verbose_errors=False to reduce noise in output")
    print("3. Use parse_to_ast_with_fallback() for maximum success rate")
    print("4. The preprocessing removes both special tokens AND error messages")



