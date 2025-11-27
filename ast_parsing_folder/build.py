from tree_sitter import Language, Parser

Language.build_library(
# Store the library in the `build` directory
'my-languages.so',

# Include one or more languages
[
"tree-sitter-languages/tree-sitter-python",
"tree-sitter-languages/tree-sitter-java",
"tree-sitter-languages/tree-sitter-javascript",
"tree-sitter-languages/tree-sitter-rust",
"tree-sitter-languages/tree-sitter-php/php",
]
)