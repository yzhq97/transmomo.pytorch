# Limb Ordering

Mid_hip is the root.

| Index | Name       | Joints   | Dependents (Joints) |
| ----- | ---------- | -------- | ------------------- |
| 0     | neck       | (1, 0)   | 0                   |
| 1     | r_shoulder | (1, 2)   | 2, 3, 4             |
| 2     | r_arm      | (2, 3)   | 3, 4                |
| 3     | r_forearm  | (3, 4)   | 4                   |
| 4     | l_shoulder | (1, 5)   | 5, 6, 7             |
| 5     | l_arm      | (5, 6)   | 6, 7                |
| 6     | l_forearm  | (6, 7)   | 7                   |
| 7     | spine      | (8, 1)   | 1, 2, 3, 4, 5, 6, 7 |
| 8     | r_pelvis   | (8, 9)   | 9, 10, 11           |
| 9     | r_thigh    | (9, 10)  | 10, 11              |
| 10    | r_shin     | (10, 11) | 11                  |
| 11    | l_pelvis   | (8, 12)  | 12, 13, 14          |
| 12    | l_thigh    | (12, 13) | 13, 14              |
| 13    | l_shin     | (13, 14) | 14                  |

