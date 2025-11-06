#!/bin/bash

# Format Python files
echo "Formatting Python files..."
black .

# Format C++ files (excluding CMakeFiles directories)
echo "Formatting C++ files..."
find . -type f \( -iname "*.hpp" -o -iname "*.cpp" \) -not -path "*/CMakeFiles/*" | while read file; do
    echo "Formatting $file"
    clang-format -i "$file"
done

echo "Formatting complete!"
