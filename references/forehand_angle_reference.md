# Forehand Angle Reference

Angle ranges used to classify tennis forehand strokes from 2D video analysis.
All angles are in **degrees** and are measured as 2D projections from the camera plane.

## Angle Definitions

- **Swing path angle**: the angle of the racket head's trajectory from ~3 frames
  before contact to the contact frame, measured from horizontal.
  - Positive = upward (low-to-high), Negative = downward.
- **Racket face angle**: how closed the racket face is relative to the incoming
  ball trajectory at the contact frame.
  - Positive = face closed/tilted forward (topspin-producing orientation).
  - 0° = face aligned with ball direction (flat contact).

## Classification Ranges

Edit the YAML block below to update shot-type thresholds. The pipeline reads
this block at runtime, so changes here propagate automatically.

```yaml
classifications:
  flat_drive:
    label: "Flat Drive"
    swing_path_angle_deg:
      min: -5
      max: 20
    racket_face_angle_deg:
      min: -10
      max: 15
  standard_topspin:
    label: "Standard Topspin"
    swing_path_angle_deg:
      min: 18
      max: 42
    racket_face_angle_deg:
      min: 12
      max: 32
  heavy_topspin:
    label: "Heavy Topspin"
    swing_path_angle_deg:
      min: 38
      max: 75
    racket_face_angle_deg:
      min: 28
      max: 65
```

## Notes

- Ranges overlap intentionally: the classifier uses nearest-centroid assignment
  when a shot falls in an overlap region.
- Values are based on standard coaching literature for a right-handed player
  filmed from the side at roughly court-level height.
- Multi-camera triangulation would be required for true 3D angle measurement.
