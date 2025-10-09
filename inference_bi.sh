#!/bin/bash

# Run the lerobot teleoperate command for bimanual setup
# python -m lerobot.teleoperate \
#     --robot.type=bi_so101_follower \
#     --robot.left_arm_port=/dev/ttyACM2 \
#     --robot.right_arm_port=/dev/ttyACM3 \
#     --robot.id=follower \
#     --robot.cameras='{
#         left: {"type": "opencv", "index_or_path": 14, "width": 640, "height": 480, "fps": 15},
#         right: {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 15},
#         top: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 15},
#     }' \
#     --teleop.type=bi_so101_leader \
#     --teleop.left_arm_port=/dev/ttyACM1 \
#     --teleop.right_arm_port=/dev/ttyACM0 \
#     --teleop.id=leader \
#     --display_data=true \

# echo "Bimanual teleoperation session ended!"

python examples/SO-100/eval_lerobot.py \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM1 \
  --robot.right_arm_port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras='{
      left: {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 15},
      right: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 15},
      top: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 15},
  }' \
  --policy_host=172.17.0.2 \
  --lang_instruction="Pick up the block using the left arm and transfer!"