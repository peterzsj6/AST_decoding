#!/bin/bash
# Check progress of MBPP AST parsing

LOG_FILE="/data/home/zhangsj/AST_decoding/mbpp_processing.log"
OUTPUT_FILE="/data/home/zhangsj/Data/MBPP/mbpp_ast_parsed.parquet"

echo "=== MBPP Processing Status ==="
echo ""

# Check if process is running
if pgrep -f "process_mbpp_ast.py" > /dev/null; then
    echo "✓ Process is running"
    PID=$(pgrep -f "process_mbpp_ast.py" | head -1)
    echo "  PID: $PID"
else
    echo "✗ Process is not running"
fi

echo ""

# Check log file
if [ -f "$LOG_FILE" ]; then
    echo "=== Recent Log Output ==="
    tail -20 "$LOG_FILE"
    echo ""
    
    # Extract progress info
    if grep -q "Progress:" "$LOG_FILE"; then
        echo "=== Latest Progress ==="
        grep "Progress:" "$LOG_FILE" | tail -1
    fi
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""

# Check output file
if [ -f "$OUTPUT_FILE" ]; then
    echo "=== Output File Status ==="
    ls -lh "$OUTPUT_FILE"
    echo ""
    
    # Try to get row count if pandas is available
    python3 -c "
import pandas as pd
try:
    df = pd.read_parquet('$OUTPUT_FILE')
    print(f'✓ Output file exists with {len(df)} rows')
    if len(df) > 0:
        print(f'  Columns: {list(df.columns)}')
        # Check AST_span format
        first_span = df.iloc[0]['AST_span']
        import json
        spans = json.loads(first_span) if isinstance(first_span, str) else first_span
        print(f'  First task has {len(spans)} AST spans')
except Exception as e:
    print(f'✗ Error reading output file: {e}')
" 2>/dev/null || echo "Could not read output file details"
else
    echo "✗ Output file not created yet: $OUTPUT_FILE"
fi



