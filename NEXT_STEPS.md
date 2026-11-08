# Beyond the starter: build real AI and measurable haptic feedback

The starter code demonstrates the wiring between modules. It DOES NOT include a trained
classifier, a validated Kalman filter, personalized assistance, or tested servo haptics.
Do not claim these capabilities in a presentation until built and evaluated.

## Member 1: perception improvements
1. Add a timestamped CSV recorder for state fields.
2. Test with different people and lighting.
3. Measure tracking availability and one-way latency.
4. Make the camera preview optionally visible to the user.
5. Report `tracked=False` and stop grasp control on tracking loss.

## Member 2: machine-learning improvements
1. Record 30-50 short examples per class per available participant: open,
   closing, holding, releasing. Keep participant IDs separate.
2. Label segments manually, using visual inspection; do not assume labels.
3. Build features: pinch ratio, closing speed, x/y velocity, moving variance.
4. Split train/test by PERSON, not by random frame. Otherwise leakage can
   make performance look artificially strong.
5. Train simple logistic regression or random forest using scikit-learn.
6. Compare to threshold-only baseline on accuracy, per-class precision/recall,
   and detection delay.
7. Try a Kalman filter for 100-200 ms ahead; compare against constant velocity.
8. Add calibrated personal pinch thresholds, evaluated with on/off modes.
9. Report measured failures, not only successes.

## Member 3: simulation improvements
1. Define object properties in a table: pickup radius, compression factor,
   virtual fragility, and a minimum simulated hold condition.
2. Keep them explicitly simulated; no claim of physical force measurement.
3. Visualize observed pointer and forecast pointer separately.
4. Add safe, bounded "assist" snapping when forecast is confident.
5. Add 4 modes baseline / adaptive / cue / both with configurable sequence.
6. Count completed transfers as SUCCESS, not just release events.
7. Store trial data and calculate success rate and median completion time.

## Member 4: feedback improvements
1. Identify exact servo make/model and operating voltage/current from label
   and official datasheet; identify available power supply and wires.
2. Build and test a servo movement mechanism WITHOUT finger contact first.
3. Use external regulated supply if current demand requires it; share ground.
4. Ensure no pinch points, no high-force motion, and a hard stop.
5. Read Grove piezo digital events and accelerometer measurements with the
   corresponding official Seeed library, one sensor at a time.
6. Timestamp and log command, detected movement, and user-reported cues.
7. Do not call it validated haptic feedback without a user perception study.

## Integration order
1. Mouse simulation -> Arduino LED using main.py --port PORT.
2. Webcam -> simulation using main.py --camera.
3. Webcam -> simulation -> LED using main.py --camera --port PORT.
4. Replace simple prediction with evaluated AI model.
5. Replace LED with safe mechanical feedback AFTER power requirements verified.
6. Add Grove sensors, measurement logging, final comparisons.

## Final presentation evidence
- Video showing real input, simulation, and response in one continuous shot.
- Four modes with the actual measured results (no fake numbers).
- Examples of tracking/prediction errors and design limitations.
- Honest labeling: research prototype, not a medical device.
