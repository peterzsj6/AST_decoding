#!/usr/bin/env python3
"""
Standalone demonstration of the preprocessing improvements.
This shows how the enhanced preprocessing handles the user's problematic code
without requiring tree-sitter dependencies.
"""

import re

def preprocess_code_for_parsing(code, ps_language="python"):
    """
    Preprocess code to handle special tokens and improve parsing success.
    
    Args:
        code: Original code string
        ps_language: Programming language
    
    Returns:
        Preprocessed code string
    """
    if not code or not isinstance(code, str):
        return code
    
    # Remove or replace common problematic tokens
    problematic_patterns = [
        # Instruction-following tokens
        (r'<\|im_start\|>', ''),
        (r'<\|im_end\|>', ''),
        (r'<\|assistant\|>', ''),
        (r'<\|user\|>', ''),
        (r'<\|system\|>', ''),
        (r'<\|end\|>', ''),
        # Other common problematic patterns
        (r'<\|.*?\|>', ''),  # Any remaining <|...|> patterns
        # Remove error messages that got mixed into code
        (r'Error at \w+:', ''),
        (r'Error at ERROR:', ''),
    ]
    
    preprocessed_code = code
    for pattern, replacement in problematic_patterns:
        preprocessed_code = re.sub(pattern, replacement, preprocessed_code)
    
    # Language-specific preprocessing
    if ps_language == "python":
        # Handle common Python-specific issues
        # Remove lines that contain error messages or problematic tokens
        lines = preprocessed_code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Skip problematic lines
            if (stripped_line and 
                not stripped_line.startswith('Error at') and
                not stripped_line.startswith('<|') and 
                not stripped_line.endswith('|>') and
                not stripped_line in ['<|', '|>', '<|>', 'im_end', 'im_start'] and
                not any(token in stripped_line for token in ['<|', '|>', 'Error at'])):
                cleaned_lines.append(line)
        preprocessed_code = '\n'.join(cleaned_lines)
        
        # Clean up extra whitespace
        preprocessed_code = re.sub(r'\n\s*\n\s*\n', '\n\n', preprocessed_code)  # Remove excessive blank lines
        preprocessed_code = preprocessed_code.strip()
    
    return preprocessed_code

def demonstrate_user_problem():
    """Demonstrate the solution to the user's specific problem."""
    print("🎯 SOLUTION TO YOUR SPECIFIC PROBLEM")
    print("=" * 60)
    
    # The exact error pattern from the user
    user_problematic_code = """def generate_integers(a, b)
Error at call: def generate_integers(a, b)
Error at ERROR: generate_integers
Error at ERROR: <|
Error at expression_statement: im_end|>
<|im_start|>assistant
Error at binary_operator: im_end|>
<|im_start|>assistant
Error at binary_operator: im_end|>
<|im_start
Error at ERROR: >
<|
Error at ERROR: >"""
    
    print("❌ BEFORE: Your problematic code:")
    print(user_problematic_code)
    print("\n" + "-" * 40)
    
    # Apply preprocessing
    cleaned_code = preprocess_code_for_parsing(user_problematic_code, "python")
    
    print("✅ AFTER: Preprocessed code:")
    print(cleaned_code)
    print("\n" + "-" * 40)
    
    # Show what was filtered out
    original_lines = user_problematic_code.split('\n')
    cleaned_lines = cleaned_code.split('\n') if cleaned_code else []
    
    print("🔍 Analysis:")
    print(f"  Original lines: {len(original_lines)}")
    print(f"  Cleaned lines: {len(cleaned_lines)}")
    print(f"  Removed lines: {len(original_lines) - len(cleaned_lines)}")
    
    print("\n📋 Lines that were removed:")
    for i, line in enumerate(original_lines, 1):
        if line.strip() and line not in cleaned_lines:
            print(f"  {i:2d}. {repr(line)}")
    
    # Check if the result would be valid Python
    if cleaned_code.strip():
        print(f"\n🔬 Syntax check:")
        try:
            compile(cleaned_code, '<string>', 'exec')
            print("✅ Result is valid Python syntax!")
        except SyntaxError as e:
            print(f"⚠️  Syntax issue remains: {e}")
            print("   (This might need manual fixing of the incomplete function)")
    else:
        print("❌ No valid code remained after preprocessing")

def show_solution_strategies():
    """Show different strategies for handling problematic code."""
    print("\n🛠️  SOLUTION STRATEGIES")
    print("=" * 60)
    
    strategies = [
        ("Strategy 1: Automatic Preprocessing", 
         "Use parse_to_ast(code, preprocess=True) - this is now the default",
         "Automatically removes special tokens and error messages"),
        
        ("Strategy 2: Quiet Error Mode", 
         "Use parse_to_ast(code, verbose_errors=False) - this is now the default",
         "Reduces noise by filtering out error messages about special tokens"),
        
        ("Strategy 3: Fallback Parsing", 
         "Use parse_to_ast_with_fallback(code) for maximum success rate",
         "Tries multiple parsing strategies automatically"),
        
        ("Strategy 4: Manual Preprocessing", 
         "Use preprocess_code_for_parsing(code) to clean code first",
         "Gives you control over the preprocessing step"),
        
        ("Strategy 5: Batch Processing", 
         "Use process_functions_from_csv(input, output, use_fallback=True)",
         "Processes large datasets with improved error handling"),
    ]
    
    for title, usage, description in strategies:
        print(f"📌 {title}")
        print(f"   Usage: {usage}")
        print(f"   Benefit: {description}")
        print()

def create_practical_example():
    """Create a practical example for the user's workflow."""
    print("💼 PRACTICAL EXAMPLE FOR YOUR WORKFLOW")
    print("=" * 60)
    
    example_code = '''
# Here's how to use the improved parsing in your workflow:

from AST_parsing import parse_to_ast, parse_to_ast_with_fallback, preprocess_code_for_parsing

# Method 1: Basic usage (recommended for most cases)
def parse_code_safely(code):
    """Parse code with automatic preprocessing and quiet errors."""
    ast_root = parse_to_ast(
        code, 
        ps_language="python", 
        preprocess=True,          # Remove special tokens automatically
        verbose_errors=False      # Reduce noise in output
    )
    return ast_root

# Method 2: Maximum robustness (recommended for batch processing)
def parse_code_robustly(code):
    """Parse code with multiple fallback strategies."""
    ast_root, strategy = parse_to_ast_with_fallback(
        code, 
        ps_language="python",
        verbose_errors=False      # Set to True for debugging
    )
    print(f"Parsed using strategy: {strategy}")
    return ast_root

# Method 3: Manual preprocessing (for custom control)
def parse_with_custom_preprocessing(code):
    """Parse code with manual preprocessing control."""
    # First, clean the code
    cleaned_code = preprocess_code_for_parsing(code, "python")
    print(f"Cleaned code: {repr(cleaned_code)}")
    
    # Then parse
    ast_root = parse_to_ast(
        cleaned_code, 
        ps_language="python",
        preprocess=False,         # Already preprocessed
        verbose_errors=True       # Show any remaining errors
    )
    return ast_root

# Method 4: Batch processing (for CSV files)
def process_your_csv_file(input_csv, output_csv):
    """Process your CSV file with improved parsing."""
    from AST_parsing import process_functions_from_csv
    
    process_functions_from_csv(
        input_csv, 
        output_csv,
        verbose_errors=False,     # Set to True for debugging
        use_fallback=True         # Use multiple parsing strategies
    )

# Example usage:
if __name__ == "__main__":
    # Your problematic code
    your_code = """def generate_integers(a, b)
Error at call: def generate_integers(a, b)
<|im_start|>assistant
    return [i for i in range(a, b + 1)]
<|im_end|>"""
    
    print("Testing with your problematic code...")
    
    # Try the different methods
    print("Method 1 (basic):")
    result1 = parse_code_safely(your_code)
    print(f"Success: {result1 is not None}")
    
    print("\\nMethod 2 (robust):")
    result2 = parse_code_robustly(your_code)
    print(f"Success: {result2 is not None}")
    
    print("\\nMethod 3 (manual preprocessing):")
    result3 = parse_with_custom_preprocessing(your_code)
    print(f"Success: {result3 is not None}")
'''
    
    print(example_code)

if __name__ == "__main__":
    print("🚀 AST Parsing Improvements Demo")
    print("This demonstrates the solutions to your parsing problems.\n")
    
    # Run demonstrations
    demonstrate_user_problem()
    show_solution_strategies()
    create_practical_example()
    
    print("\n" + "=" * 60)
    print("🎉 SUMMARY OF IMPROVEMENTS")
    print("=" * 60)
    print("✅ Special tokens (<|im_start|>, <|im_end|>, etc.) are automatically removed")
    print("✅ Error messages mixed into code are filtered out")
    print("✅ Verbose error output is reduced and filtered")
    print("✅ Multiple parsing strategies provide fallback options")
    print("✅ Batch processing includes comprehensive statistics")
    print("✅ Syntax validation helps identify remaining issues")
    
    print("\n🔧 TO FIX YOUR ISSUE:")
    print("Replace your current parse_to_ast() calls with:")
    print("  ast_root = parse_to_ast(code, preprocess=True, verbose_errors=False)")
    print("Or for maximum robustness:")
    print("  ast_root, strategy = parse_to_ast_with_fallback(code)")
