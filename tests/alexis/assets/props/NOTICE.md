# assets/props — sources and licences

- `camera_mount_v2.usd` (mesh `plate`): wrist camera mount for the SO-101, derived from
  "Wrist camera mount for SO-ARM101 robotic arm (fits IMX335 5MP USB Camera)" by **therobotguy**,
  Thingiverse thing 7390547 (https://www.thingiverse.com/thing:7390547), licensed **CC BY-NC**.
  Changes: plate rebuilt thicker with CadQuery (3 mm, counterbored M2 holes, chamfered opening, fillets); the original
  tab is kept. Non-commercial use only. The black camera board and lens in the same file are simplified boxes
  (Waveshare IMX335 5MP USB Camera (B) outline, 25 x 24 mm).
- `camera_mount_v2_fit.json`: pose of that mount on the SO-101 gripper link, computed by `scripts/sim/fit_wrist_mount.py`
  from the robot USD (`assets/robots/so101/`, see its NOTICE) and the mount mesh.
- `table_wood.png` / `table_wood.usd`: texture cropped from a photo of the team's own wooden table (mirrored to tile).
