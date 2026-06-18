# Astro Image Processor

## Overview

Astro Image Processor is a Python-based project designed to process and analyze astronomical images taken through a telescope (specifically Sky_Watcher Virtuoso GTI 150P). The goal is to build a reusable system that can enhance and study different types of celestial objects such as planets, the Moon, and star fields using the same core tools.

Instead of writing separate programs for each object, this project uses a modular pipeline that adapts processing methods depending on the target image.

---

## Goals

* Build a reusable image processing system for astronomy
* Learn how different celestial objects require different processing techniques
* Apply Python to real-world scientific and engineering-style problems
* Improve astrophotography image quality and extract useful information

---

## What the Program Does

The system can:

* Load astronomical images
* Reduce noise caused by atmosphere and camera limitations
* Enhance contrast and sharpness
* Apply object-specific processing modes (Mars, Moon, stars, etc.)
* Output improved images for analysis and visualization

---

## Processing Modes

The project uses configurable modes depending on the target object:

* **Mars Mode**

  * Color correction (red/orange enhancement)
  * Detail sharpening

* **Moon Mode**

  * Edge enhancement for craters and surface features
  * High-contrast mapping

* **Star Field Mode**

  * Brightness thresholding
  * Star detection and counting

---

## 🔧 Technologies Used

* Python
* NumPy
* OpenCV
* Matplotlib

---

## Project Structure

```
astro-image-processor/
│
├── main.py              # Runs the processing pipeline
├── processing.py       # Core image processing functions
├── configs.py          # Settings for each mode (Mars, Moon, Stars)
├── examples/           # Sample telescope images
│   ├── mars.jpg
│   ├── moon.jpg
│   └── stars.jpg
└── outputs/            # Processed results
```

---

## Why I Built This

I am interested in aerospace engineering and astronomy, and I wanted to combine telescope observation with programming. I really want to understand how engineers and scientists process real astronomical data and turn raw images into meaningful information and beautiful astrophotographies.

---

## Contact

Instagram: https://instagram.com/engineeringthecosmos2026/
GitHub: https://github.com/mksw0615-oss
