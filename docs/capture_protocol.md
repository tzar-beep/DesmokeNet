# Capture protocol: stale-reference split (draft)

Follows the SmokeBench setup (fixed surveillance camera, smoke machine, sealed room) and adds
a reference recorded *before* the scene changes.

## Per scene
1. **Mount** the camera rigidly. Lock exposure, white balance and focus if the camera allows.
2. **Reference (T0):** record 10 s of the clean scene. Note the lighting.
3. **Change the scene.** Pick one change type per configuration and log it:
   - `move`: move 1–3 objects (chair, bag, box)
   - `add` / `remove`: add or remove an object
   - `light`: switch a lamp, change colour temperature (2700K ↔ 6000K), open blinds
   - `person`: a person stands still in the scene (mannequin if possible)
   - `none`: control, no change
4. **Clean GT (T1):** record 10 s of the changed, smoke-free scene.
5. **Smoke:** start the smoke machine. Record continuously until visibility is near zero,
   then through dissipation. Keep **all** frames (SmokeBench only kept one every 2 s;
   consecutive frames enable later temporal work).
6. **Change mask:** annotate the regions that differ between T0 and T1 (polygon tool,
   e.g. CVAT). Required for the test split.

## Targets
- 10–15 scenes × 2–3 change configurations each, plus `none` controls.
- Vary room size, clutter and lighting as SmokeBench does.

## Log per configuration
scene id · change type · lighting · camera model and settings · smoke machine setting ·
time stamps of T0, T1 and smoke start · notes on anything that moved accidentally.

## Safety
Use a fog/smoke machine rated for indoor use. Never use real combustion indoors. Ventilate
after each session and follow your institution's lab-safety rules.
