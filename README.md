# jason-project

Parametric CAD base model for a compact autonomous litter-collection robot.

## Litter Robot CAD

The CAD source lives in `litter_robot_cad/build_litter_robot.py`. It builds a
four-wheel aluminum T-slot frame prototype with a front mechanical brush, scoop
ramp, internal waste bin, low battery box, electronics enclosure, and camera
mast.

Generated deliverables are included under `litter_robot_cad/exports/`:

- `step/litter_collection_robot_full_assembly.step`
- `stl/litter_collection_robot_full_assembly.stl`
- major grouped STL files under `stl/`
- `png/litter_collection_robot_preview.png`

Rebuild with:

```powershell
cd litter_robot_cad
python build_litter_robot.py
```
