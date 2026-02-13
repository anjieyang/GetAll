#!/bin/bash
# Count core agent lines (excluding channels/, cli/, providers/ adapters)
cd "$(dirname "$0")" || exit 1

echo "getall core agent line count"
echo "================================"
echo ""

for dir in agent agent/tools bus config cron heartbeat session utils; do
  count=$(find "getall/$dir" -maxdepth 1 -name "*.py" -exec cat {} + | wc -l)
  printf "  %-16s %5s lines\n" "$dir/" "$count"
done

root=$(cat getall/__init__.py getall/__main__.py | wc -l)
printf "  %-16s %5s lines\n" "(root)" "$root"

echo ""
total=$(find getall -name "*.py" ! -path "*/channels/*" ! -path "*/cli/*" ! -path "*/providers/*" | xargs cat | wc -l)
echo "  Core total:     $total lines"
echo ""
echo "  (excludes: channels/, cli/, providers/)"
