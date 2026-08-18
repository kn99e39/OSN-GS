#!/bin/bash
# Validity gate for a run of `run_2dgs_vs_vanilla_30k.sh`.
#
# `--adc_max_gaussians` is a crash guard, not an experimental variable. If it
# ever binds, OSN-GS's density control spends the remaining budget on clones
# before splits, so splitting stops entirely -- which silently destroys the
# arm's geometry and invalidates any comparison drawn from it. A bound cap is
# visible as an ADC step that reports `split_parents=0` while still cloning.
#
# Usage: bash scripts/devtools/check_adc_cap.sh <logfile> [...]
set -uo pipefail

status=0
for log in "$@"; do
  bound=$(tr '\r' '\n' < "${log}" \
    | grep "OSN-GS ADC: iteration" \
    | grep "split_parents=0 " \
    | grep -v "clone_parents=0 " \
    | head -1)
  peak=$(tr '\r' '\n' < "${log}" \
    | grep -oE "gaussians=[0-9]+" | cut -d= -f2 | sort -n | tail -1)
  if [ -n "${bound}" ]; then
    echo "INVALID ${log}: primitive cap bound (clones with zero splits) -- $(echo "${bound}" | grep -oE 'iteration=[0-9]+|gaussians=[0-9]+' | tr '\n' ' ')"
    status=1
  else
    echo "OK      ${log}: cap never bound, peak primitives=${peak:-unknown}"
  fi
done
exit "${status}"
