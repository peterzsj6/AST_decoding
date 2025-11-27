#!/usr/bin/env python3
"""
Comprehensive test script for improved AST parsing functionality.
This demonstrates how the enhanced parsing handles:
1. Special tokens like <|im_start|>, <|im_end|>
2. Error messages mixed into code
3. Malformed syntax
4. Various parsing strategies
"""

import re

def preprocess_code_for_parsing(code, ps_language="python"):
    """
    Preprocess code to handle special tokens and improve parsing success.
    (Standalone version for testing without tree-sitter dependencies)
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

def test_user_problematic_code():
    """Test the exact problematic code from the user's error message."""
    
    # The exact problematic code from the user
    problematic_code = """def generate_integers(a, b)
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

    print("🧪 Testing the exact problematic code from user's error message...")
    print("=" * 80)
    
    print("Original problematic code:")
    print(repr(problematic_code))
    print("\nVisual representation:")
    print(problematic_code)
    print("\n" + "-" * 50)
    
    # Test preprocessing
    preprocessed_code = preprocess_code_for_parsing(problematic_code, "python")
    print("After preprocessing:")
    print(repr(preprocessed_code))
    print("\nVisual representation:")
    print(preprocessed_code)
    print("\n" + "-" * 50)
    
    # Analyze what was removed
    removed_lines = []
    original_lines = problematic_code.split('\n')
    preprocessed_lines = preprocessed_code.split('\n') if preprocessed_code else []
    
    print("Analysis of changes:")
    print(f"  Original lines: {len(original_lines)}")
    print(f"  Preprocessed lines: {len(preprocessed_lines)}")
    
    # Show which lines were removed
    for i, line in enumerate(original_lines):
        if line.strip() and line not in preprocessed_lines:
            removed_lines.append(f"Line {i+1}: {repr(line)}")
    
    if removed_lines:
        print(f"  Removed {len(removed_lines)} problematic lines:")
        for removed_line in removed_lines:
            print(f"    {removed_line}")
    else:
        print("  No lines were removed")
    
    # Test if the result is valid Python
    if preprocessed_code.strip():
        print(f"\nSyntax validation:")
        try:
            compile(preprocessed_code, '<string>', 'exec')
            print("✅ Preprocessed code is valid Python syntax")
        except SyntaxError as e:
            print(f"❌ Syntax error in preprocessed code: {e}")
        except Exception as e:
            print(f"⚠️  Other compilation issue: {e}")
    else:
        print("❌ Preprocessing resulted in empty code")

def test_various_error_patterns():
    """Test various types of error patterns that might appear in code."""
    
    test_cases = [
        # Case 1: Mixed error messages and code
        ("""def function_name():
Error at call: some_function()
    return True
Error at ERROR: variable_name
""", "Mixed error messages and code"),
        
        # Case 2: Special tokens mixed with code
        ("""<|im_start|>assistant
def calculate_sum(a, b):
    return a + b
<|im_end|>
Error at binary_operator: +""", "Special tokens with error messages"),
        
        # Case 3: Malformed function definition
        ("""def incomplete_function(
Error at ERROR: missing_param
Error at call: incomplete_function(""", "Malformed function definition"),
        
        # Case 4: Valid code that should parse normally
        ("""def valid_function(x, y):
    '''This is a valid function'''
    result = x + y
    return result""", "Valid code without issues"),
        
        # Case 5: Complex mixed content
        ("""<|im_start|>user
Write a function
<|im_end|>
<|im_start|>assistant
def user_requested_function():
Error at call: user_requested_function()
    pass
<|im_end|>
Error at ERROR: function_end""", "Complex mixed content"),
        
        # Case 6: Only error messages
        ("""Error at call: some_function()
Error at ERROR: variable_name
Error at binary_operator: +
<|im_start|>assistant
<|im_end|>""", "Only error messages and tokens"),
    ]
    
    print("\n🧪 Testing various error patterns...")
    print("=" * 80)
    
    for i, (code, description) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {description} ---")
        print("Original:")
        print(code)
        
        preprocessed = preprocess_code_for_parsing(code, "python")
        print("\nPreprocessed:")
        print(preprocessed)
        
        # Check if result is valid Python
        if preprocessed.strip():
            try:
                compile(preprocessed, '<string>', 'exec')
                print("✅ Result is valid Python syntax")
            except SyntaxError as e:
                print(f"❌ Syntax error: {e}")
            except Exception as e:
                print(f"⚠️  Other issue: {e}")
        else:
            print("❌ Preprocessing resulted in empty code")
        
        print("-" * 40)

def test_syntax_validation():
    """Test the syntax validation functionality."""
    
    def validate_code_syntax(code, ps_language="python"):
        """Basic syntax validation for code before AST parsing."""
        issues = []
        
        if not code or not isinstance(code, str):
            return False, ["Invalid or empty code"]
        
        if ps_language == "python":
            # Basic Python syntax checks
            lines = code.split('\n')
            
            # Check for basic structure
            has_meaningful_content = any(line.strip() and not line.strip().startswith('#') for line in lines)
            
            if not has_meaningful_content:
                issues.append("No meaningful code content found")
            
            # Check for unmatched parentheses/brackets
            open_chars = {'(': ')', '[': ']', '{': '}'}
            stack = []
            for char in code:
                if char in open_chars:
                    stack.append(open_chars[char])
                elif char in open_chars.values():
                    if not stack or stack.pop() != char:
                        issues.append("Unmatched brackets/parentheses")
                        break
            
            if stack:
                issues.append("Unclosed brackets/parentheses")
            
            # Check for basic Python syntax requirements
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                issues.append(f"Python syntax error: {e}")
            except Exception:
                # Other compilation errors are OK for AST parsing
                pass
        
        return len(issues) == 0, issues
    
    test_codes = [
        "def valid_function(): return True",
        "def invalid_function(: return True",  # Missing closing paren
        "def unclosed_function():\n    if True:",  # Unclosed if
        "",  # Empty code
        "# Just a comment",  # Only comment
        "def func():\n    return [1, 2, 3",  # Unclosed bracket
    ]
    
    print("\n🧪 Testing syntax validation...")
    print("=" * 80)
    
    for i, code in enumerate(test_codes, 1):
        print(f"\n--- Validation Test {i} ---")
        print(f"Code: {repr(code)}")
        
        is_valid, issues = validate_code_syntax(code, "python")
        print(f"Valid: {is_valid}")
        if issues:
            print(f"Issues: {issues}")

def demonstrate_improvements():
    """Demonstrate the key improvements made to AST parsing."""
    
    print("\n🚀 Demonstrating AST Parsing Improvements")
    print("=" * 80)
    
    print("Key improvements made:")
    print("1. ✅ Preprocessing to remove special tokens (<|im_start|>, <|im_end|>, etc.)")
    print("2. ✅ Filtering of error messages mixed into code")
    print("3. ✅ Better error reporting with filtering")
    print("4. ✅ Syntax validation before parsing")
    print("5. ✅ Multiple fallback parsing strategies")
    print("6. ✅ Comprehensive statistics and reporting")
    
    print("\n📋 How to use the improvements:")
    print("1. For basic usage with automatic preprocessing:")
    print("   ast_root = parse_to_ast(code, preprocess=True)")
    print()
    print("2. For robust parsing with fallback strategies:")
    print("   ast_root, strategy = parse_to_ast_with_fallback(code)")
    print()
    print("3. For batch processing from CSV:")
    print("   process_functions_from_csv(input_file, output_file, use_fallback=True)")
    print()
    print("4. For debugging with detailed errors:")
    print("   ast_root = parse_to_ast(code, verbose_errors=True)")

if __name__ == "__main__":
    print("🚀 Comprehensive AST Parsing Test Suite")
    print("This script demonstrates the enhanced parsing capabilities")
    print("for handling problematic code with special tokens and error messages.\n")
    
    # Run tests
    test_user_problematic_code()
    test_various_error_patterns()
    test_syntax_validation()
    demonstrate_improvements()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("\n💡 Summary of solutions:")
    print("   - Special tokens like <|im_start|>, <|im_end|> are automatically removed")
    print("   - Error messages mixed into code are filtered out")
    print("   - Multiple parsing strategies ensure maximum success rate")
    print("   - Verbose error reporting helps with debugging")
    print("   - Syntax validation catches issues before AST parsing")



