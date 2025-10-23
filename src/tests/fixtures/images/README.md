# Test Image Fixtures

This directory contains test images for integration testing.

## Directory Structure

```
with_humans/     - Images containing one or more humans
without_humans/  - Images without any humans (objects, landscapes, etc.)
```

## Current Images

### Without Humans (4 images)
- `solid_color.jpg` - Solid color image
- `gradient.jpg` - Color gradient
- `shapes.jpg` - Geometric shapes (circles, rectangles)
- `noise.jpg` - Random noise

### With Humans (0 real images)
- `stick_figure.jpg` - Placeholder (YOLO won't detect this)

## Adding Real Images

To properly test human detection, add actual photos to `with_humans/`:

**Option 1: Download from free stock photo sites**
```bash
# Example using wget
wget -O tests/fixtures/images/with_humans/person1.jpg "https://..."
```

**Option 2: Use your own photos**
- Copy JPG/PNG images to the appropriate folder
- Ensure images are properly licensed for use in tests

**Option 3: Use COCO validation samples**
- Download from COCO dataset
- Select images with "person" annotations

## Usage in Tests

```python
import os
from pathlib import Path

# Get path to fixtures
fixtures_dir = Path(__file__).parent / "fixtures" / "images"

# Load images with humans
with_humans = list((fixtures_dir / "with_humans").glob("*.jpg"))

# Load images without humans  
without_humans = list((fixtures_dir / "without_humans").glob("*.jpg"))
```
