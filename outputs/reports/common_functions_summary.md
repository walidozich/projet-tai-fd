# Common Functions Summary

Implemented reusable functions for the shared project pipeline.

## Data loading

- `load_image(path, rgb=True)`
- `load_mask(path)`
- `save_mask(mask, path)`

## Visualization

- `show_image_mask(image, mask, title='', save_path=None)`
- `overlay_mask(image, mask, color=(255, 0, 0), alpha=0.45)`

## Preprocessing and features

- `normalize_minmax(features)`
- `standardize_zscore(features)`
- `resize_for_clustering(image, max_size)`
- `extract_rgb_features(image)`
- `extract_rgb_xy_features(image)`
- `extract_gray_gradient_xy_features(image)`

## Connected components helpers

- `remove_small_components(mask, min_area)`
- `keep_largest_component(mask)`

## Verification

A smoke test was run on `scene2` and passed. It verified image/mask loading, feature extraction, scaling, resizing, connected components, overlay creation, and mask saving.
