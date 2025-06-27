# Cross-Camera Player Mapping System

A computer vision system for detecting and matching players across different camera views in sports analytics applications.

## Features

- **Multi-view Player Detection**: YOLOv11-based player detection in broadcast and tactical camera feeds
- **Cross-Camera Matching**: Advanced algorithms to identify same players across different views
- **Visual Analytics**: Generate annotated videos and comparison visualizations
- **Performance Metrics**: Detailed statistics and reporting for analysis

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended) with CUDA 11.7
- FFmpeg (for video processing)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Reaper095/Cross-Camera-Player-Mapping-System.git
   cd Cross-Camera-Player-Mapping-System
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

```bash
python main.py \
    --broadcast_path data/videos/broadcast.mp4 \
    --tacticam_path data/videos/tacticam.mp4 \
    --model_path data/models/<your_player_detection_model_name>.pt \
    --output_dir results \
    --log_level INFO
```

### Command Line Options

```bash
python main.py \
    --broadcast_path <path> \
    --tacticam_path <path> \
    --model_path <path> \
    --output_dir <dir> \
    [--max_frames 1000] \
    [--skip_frames 1] \
    [--detection_confidence 0.5] \
    [--detection_iou 0.5] \
    [--matching_threshold 0.4] \
    [--max_distance 200.0] \
    [--log_level INFO]
```

## Output Structure

```
results/
├── detections/         # JSON files with detection data
├── visualizations/     # Annotated videos and plots
│   ├── broadcast_annotated.mp4
│   ├── tacticam_annotated.mp4
│   ├── comparison_video.mp4
│   └── detection_summary.png
└── results/            # CSV reports and statistics
    ├── player_mappings.csv
    └── detection_stats.json
```

## Configuration

Key parameters in `config.json` (or via command line):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `detection_confidence` | Minimum confidence for player detection | 0.1 |
| `detection_iou` | Intersection-over-Union threshold for NMS | 0.5 |
| `matching_threshold` | Similarity score threshold for player matching | 0.3 |
| `max_distance` | Maximum spatial distance for matching (pixels) | 1200.0 |
| `color_hist_bins` | Bins for color histogram features | 32 |
| `hog_orientations` | HOG feature orientations | 9 |

## Troubleshooting

### Common Issues

**1. No matches found:**
- Verify videos are temporally aligned
- Reduce `matching_threshold` (try 0.1-0.4)
- Increase `max_distance` (try 1200-1300)

**2. Low detection count:**
- Lower `detection_confidence` (try 0.2-0.3)
- Check video quality and resolution
- Verify model is appropriate for your sport/players

**3. Performance problems:**
- Increase `skip_frames` value
- Reduce `max_frames` for testing
- Use GPU acceleration

## Development

### Project Structure

```
player_mapping_project/
├── data/                     # Sample videos and models
│   ├── videos/               
│   └── models/               # Put your player detection model within models folder
├── src/                      # Core source code
│   ├── _init_.py             
│   ├── feature_extractor.py  
│   ├── player_detector.py   
│   ├── player_matcher.py     
│   ├── video_processor.py    
│   └── utils.py              
├── player_mapping.log        # Player mappings logs
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv11 team for object detection model
- OpenCV community for computer vision tools

## Contact

For questions or support, please open an issue on GitHub or contact [anunayminj@gmail.com].

---

**Note**: This system is designed for research and educational purposes. For production use, ensure compliance with relevant broadcasting and privacy regulations.
