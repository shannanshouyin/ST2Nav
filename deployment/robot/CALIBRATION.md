# Installation calibration

Fill `calibration.example.json` with measurements from the installed cameras before acquisition. Keep the camera brackets and waist posture fixed after calibration.

1. Obtain each camera's serial, color intrinsics `[fx, fy, cx, cy]`, distortion coefficients and depth scale at **1280 × 720, 6 Hz**. Depth is aligned to color; the present capture code requires zero color distortion coefficients. Save physical portrait orientation in the extrinsic rotation, not by rotating the stored arrays.
2. Estimate each rigid transform `T_pelvis_from_color_at_capture_waist` from independent target observations. It maps **optical x-right/y-down/z-forward** coordinates (meters) into **pelvis x-forward/y-left/z-up** coordinates. Record measured waist angles in `body_reference.capture_states[*].waist_q_rad`.
3. Use a measured planar target of known dimensions. Measure its center distance, height and lateral offset relative to the pelvis, and its orientation. Prefer several target poses. Check held-out target points and RGBD overlap at multiple distances before accepting the result.
4. Populate the template, validate rigid transforms and intrinsics, and only after independent checks set `status` to `STATIC_LOCAL_CHECKS_PASSED`. Save as `robot/dual-camera-static-calibration.json`; copy the **identical file bytes** to `server/dual-camera-static-calibration.json`. Server capture verification compares the file hash.

Reuse the verified calibration for the same installation and recalibrate after either camera moves. Keep measured calibration files and camera serials in local configuration.

Optionally set `NAV_CALIBRATION` on each computer to an external private calibration file instead of storing it in the repository. Both files must have identical bytes.
