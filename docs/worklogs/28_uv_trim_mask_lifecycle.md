# 23. UV Trim-Mask Lifecycle

Date: 2026-07-15

## Work

Maintenance now rebuilds every patch UV support mask after it refreshes Gaussian
UV bindings, when trimming is enabled. The maintenance report exposes
support_masks_refreshed.

## Verification

Ran the NURBS surface and training-regression test modules.

Result: 24 tests passed.

## Result

The initialization-time trim mask no longer remains stale after maintenance UV
projection. The TODO lifecycle item was removed. Mask and UV version metadata
remains future diagnostic work.
