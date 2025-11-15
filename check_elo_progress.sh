#!/bin/bash
echo "ELO Test Progress Monitor"
echo "========================="
echo ""
tail -50 /proc/$(pgrep -f "auto_elo_test.py" | head -1)/fd/1 2>/dev/null || echo "Test not running or completed"
