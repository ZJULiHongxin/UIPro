'''
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

import jax, ast
import jax.numpy as jnp
import numpy as np
from utils.data_utils.misc import find_smallest_box_containing_point
'''
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

import enum

action_id2text = {
  0: "SWIPE DOWN",
  1: "SWIPE UP",
  2: "SELECT",
  3: "TYPE",
  4: "CLICK",
  5: "PRESS_BACK",
  6: "PRESS_HOME",
  7: "PRESS_ENTER",
  8: "SWIPE LEFT",
  9: "SWIPE RIGHT",
  10: "STATUS_TASK_COMPLETE",
  11: "STATUS_TASK_IMPOSSIBLE",
  99: "OTHERS"
} 
class ActionType(enum.IntEnum):

  # Placeholders for unused enum values
  UNUSED_0 = 0
  UNUSED_1 = 1
  UNUSED_2 = 2
  UNUSED_8 = 8
  UNUSED_9 = 9

  ########### Agent actions ###########

  # A type action that sends text to the emulator. Note that this simply sends
  # text and does not perform any clicks for element focus or enter presses for
  # submitting text.
  TYPE = 3

  # The dual point action used to represent all gestures.
  DUAL_POINT = 4

  # These actions differentiate pressing the home and back button from touches.
  # They represent explicit presses of back and home performed using ADB.
  PRESS_BACK = 5
  PRESS_HOME = 6

  # An action representing that ADB command for hitting enter was performed.
  PRESS_ENTER = 7

  ########### Episode status actions ###########

  # An action used to indicate the desired task has been completed and resets
  # the environment. This action should also be used in the case that the task
  # has already been completed and there is nothing to do.
  # e.g. The task is to turn on the Wi-Fi when it is already on
  STATUS_TASK_COMPLETE = 10

  # An action used to indicate that desired task is impossible to complete and
  # resets the environment. This can be a result of many different things
  # including UI changes, Android version differences, etc.
  STATUS_TASK_IMPOSSIBLE = 11


_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


def _yx_in_bounding_boxes(
    yx, bounding_boxes
):
  """Check if the (y,x) point is contained in each bounding box.

  Args:
    yx: The (y, x) coordinate in pixels of the point.
    bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
      represents a bounding box: (y_top_left, x_top_left, box_height,
      box_width). Note: containment is inclusive of the bounding box edges.

  Returns:
    is_inside: A 1D bool array where each element specifies if the point is
      contained within the respective box.
  """
  y, x = yx

  # `bounding_boxes` has shape (n_elements, 4); we extract each array along the
  # last axis into shape (n_elements, 1), then squeeze unneeded dimension.
  top, left, height, width = [
      jnp.squeeze(v, axis=-1) for v in jnp.split(bounding_boxes, 4, axis=-1)
  ]

  # The y-axis is inverted for AndroidEnv, so bottom = top + height.
  bottom, right = top + height, left + width

  return jnp.logical_and(y >= top, y <= bottom) & jnp.logical_and(
      x >= left, x <= right)


def _resize_annotation_bounding_boxes(
    annotation_positions, annotation_width_augment_fraction,
    annotation_height_augment_fraction):
  """Resize the bounding boxes by the given fractions.

  Args:
    annotation_positions: Array of shape (N, 4), where each row represents the
      (y, x, height, width) of the bounding boxes.
    annotation_width_augment_fraction: The fraction to augment the box widths,
      E.g., 1.4 == 240% total increase.
    annotation_height_augment_fraction: Same as described for width, but for box
      height.

  Returns:
    Resized bounding box.

  """
  height_change = (
      annotation_height_augment_fraction * annotation_positions[:, 2])
  width_change = (
      annotation_width_augment_fraction * annotation_positions[:, 3])

  # Limit bounding box positions to the screen.
  resized_annotations = jnp.stack([
      jnp.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
      jnp.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
      jnp.minimum(1, annotation_positions[:, 2] + height_change),
      jnp.minimum(1, annotation_positions[:, 3] + width_change),
  ],
                                  axis=1)
  return resized_annotations


def is_tap_action(normalized_start_yx,
                  normalized_end_yx):
  distance = jnp.linalg.norm(
      jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
  return jnp.not_equal(action_type, ActionType.DUAL_POINT)


def _check_tap_actions_match(
    tap_pred_yx,
    tap_ref_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
    correct_if_in_the_same_row
):
  """
  Determines if two tap actions are the same.
  annotation_positions: (y, x, height, width)
  """
  
  both_in_box = False
  if len(annotation_positions):
    resized_annotation_positions = _resize_annotation_bounding_boxes(
        annotation_positions,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )

    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap_pred_in_box = _yx_in_bounding_boxes(tap_pred_yx, resized_annotation_positions)
    tap_ref_in_box = _yx_in_bounding_boxes(tap_ref_yx, resized_annotation_positions)
    both_in_box = jnp.max(tap_pred_in_box & tap_ref_in_box)

    if correct_if_in_the_same_row:
        # find the gt box
        new_boxes = np.stack([
            annotation_positions[:,1],
            annotation_positions[:,0],
            annotation_positions[:,1] + annotation_positions[:,3],
            annotation_positions[:,0] + annotation_positions[:,2],
        ], axis=1)
        box, box_idx = find_smallest_box_containing_point(tap_ref_yx[::-1], new_boxes)
        
        if box_idx is not None:
            if box[1] <= tap_pred_yx[0] <= box[3] and box[0] <= tap_pred_yx[1] <= 1 - box[0]:
                both_in_box = True
        else:
            if tap_ref_yx[0] - 0.1 <= tap_pred_yx[0] <= tap_ref_yx[0] + 0.1:
                both_in_box = True

  # If the ground-truth tap action falls outside any of the annotation
  # bounding boxes or one of the actions is inside a bounding box and the other
  # is outside bounding box or vice versa, compare the points using Euclidean
  # distance.
  within_threshold = (
      jnp.linalg.norm(jnp.array(tap_pred_yx) - jnp.array(tap_ref_yx))
      <= matching_tap_distance_threshold_screen_percentage
  )
  return jnp.logical_or(both_in_box, within_threshold)


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
  """Determines if two drag actions are the same."""
  # Store drag deltas (the change in the y and x coordinates from touch to
  # lift), magnitudes, and the index of the main axis, which is the axis with
  # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
  # ending at (0.3, 0.5) has a main axis index of 1).
  drag_1_deltas = drag_1_lift_yx - drag_1_touch_yx
  drag_1_magnitudes = jnp.abs(drag_1_deltas)
  drag_1_main_axis = np.argmax(drag_1_magnitudes)
  drag_2_deltas = drag_2_lift_yx - drag_2_touch_yx
  drag_2_magnitudes = jnp.abs(drag_2_deltas)
  drag_2_main_axis = np.argmax(drag_2_magnitudes)

  return jnp.equal(drag_1_main_axis, drag_2_main_axis)


def check_actions_match(
    action_pred_touch_yx,
    action_pred_lift_yx,
    action_pred_action_type,
    action_ref_touch_yx,
    action_ref_lift_yx,
    action_ref_action_type,
    annotation_positions,
    tap_distance_threshold = _TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
    correct_if_in_the_same_row = False
):
    """Determines if two actions are considered to be the same.

    Two actions being "the same" is defined here as two actions that would result
    in a similar screen state.

    Args:
        action_pred_touch_yx: The (y, x) coordinates of the first action's touch.
        action_pred_lift_yx: The (y, x) coordinates of the first action's lift.
        action_pred_action_type: The action type of the first action.
        action_ref_touch_yx: The (y, x) coordinates of the second action's touch.
        action_ref_lift_yx: The (y, x) coordinates of the second action's lift.
        action_ref_action_type: The action type of the second action.
        annotation_positions: The positions of the UI annotations for the screen. It
        is A 2D int array of shape (num_bboxes, 4), where each row represents a
        bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
        containment is inclusive of the bounding box edges.
        tap_distance_threshold: The threshold that determines if two taps result in
        a matching screen state if they don't fall the same bounding boxes.
        annotation_width_augment_fraction: The fraction to increase the width of the
        bounding box by.
        annotation_height_augment_fraction: The fraction to increase the height of
        of the bounding box by.

    Returns:
        A boolean representing whether the two given actions are the same or not.
    """
    action_pred_touch_yx = jnp.asarray(action_pred_touch_yx)
    action_pred_lift_yx = jnp.asarray(action_pred_lift_yx)
    action_ref_touch_yx = jnp.asarray(action_ref_touch_yx)
    action_ref_lift_yx = jnp.asarray(action_ref_lift_yx)

    # Checks if at least one of the actions is global (i.e. not DUAL_POINT),
    # because if that is the case, only the actions' types need to be compared.
    has_non_dual_point_action = jnp.logical_or(
        _is_non_dual_point_action(action_pred_action_type),
        _is_non_dual_point_action(action_ref_action_type),
    )
    #print("non dual point: "+str(has_non_dual_point_action))

    different_dual_point_types = jnp.logical_xor(
        is_tap_action(action_pred_touch_yx, action_pred_lift_yx),
        is_tap_action(action_ref_touch_yx, action_ref_lift_yx),
    )
    #print("different dual type: "+str(different_dual_point_types))

    is_tap = jnp.logical_and(
        is_tap_action(action_pred_touch_yx, action_pred_lift_yx),
        is_tap_action(action_ref_touch_yx, action_ref_lift_yx),
    )
    #print("is tap: "+str(is_tap))

    taps_match = _check_tap_actions_match(
        action_pred_touch_yx,
        action_ref_touch_yx,
        annotation_positions,
        tap_distance_threshold,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
        correct_if_in_the_same_row
    ) if is_tap else False
    #print("tap match: "+str(taps_match))

    taps_match = jnp.logical_and(is_tap, taps_match)
    #print("tap match: "+str(taps_match))

    drags_match = _check_drag_actions_match(
        action_pred_touch_yx, action_pred_lift_yx, action_ref_touch_yx, action_ref_lift_yx
    )
    drags_match = jnp.where(is_tap, False, drags_match)
    #print("drag match: "+str(drags_match))

    return jnp.where(
        has_non_dual_point_action,
        jnp.equal(action_pred_action_type, action_ref_action_type),
        jnp.where(
            different_dual_point_types,
            False,
            jnp.logical_or(taps_match, drags_match),
        ),
    )


def action_2_format(step_data):
    # 把test数据集中的动作格式转换为计算matching score的格式
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # 点击
            touch_point = step_data["touch"][::-1]
            lift_point = step_data["lift"][::-1]
        else:  # 上下左右滑动
            if step_data["action_type_text"] == 'scroll down':
                touch_point = [0.5, 0.8]
                lift_point = [0.5, 0.2]
            elif step_data["action_type_text"] == 'scroll up':
                touch_point = [0.5, 0.2]
                lift_point = [0.5, 0.8]
            elif step_data["action_type_text"] == 'scroll left':
                touch_point = [0.2, 0.5]
                lift_point = [0.8, 0.5]
            elif step_data["action_type_text"] == 'scroll right':
                touch_point = [0.8, 0.5]
                lift_point = [0.2, 0.5]
    else:
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]

    if action_type == 3:
        typed_text = step_data["type_text"]
    else:
        typed_text = ""

    action = {"action_type": action_type, "touch_point": touch_point, "lift_point": lift_point,
              "typed_text": typed_text}

    # action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    # action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    action["typed_text"] = action["typed_text"].lower()

    return action


def pred_2_format(step_data, scale=1):
    if isinstance(step_data, str):
        step_data = ast.literal_eval(step_data)
    scale = int(float(scale))
    # 把模型输出的内容转换为计算action_matching的格式
    action_type = step_data["action_type"]

    touch_point = [-1.0, -1.0]
    lift_point = [-1.0, -1.0]
    typed_text = ""
        
    if action_type == 'click':  # 点击
        action_type_new = 4
        target = step_data["target"][::-1]
        if scale > 1:
            target = list(map(lambda x: x/scale, target))
        touch_point = target
        lift_point = target
    elif action_type == 'dual_point_gesture':
        action_type_new = 4
        touch_point, lift_point = list(map(lambda x: x/scale, step_data['start'])), list(map(lambda x: x/scale, step_data['end']))
    elif action_type in ['input_text', 'type']:
        action_type_new = 3
        if 'text' in step_data:
            typed_text = step_data['text']
        else: typed_text = step_data['typed_text']
    elif action_type == 'swipe': # swipe up/down/left/right are assigned the ids 1, 0, 8, and 9 respectively.
        action_type_new, direction = 4, step_data['direction']
        if direction == 'up':
          touch_point = [0.5, 0.8] # y, x
          lift_point = [0.5, 0.2]
        elif action_type == 'down':
            touch_point = [0.5, 0.2]
            lift_point = [0.5, 0.8]
        elif action_type == 'right':
            touch_point = [0.2, 0.5]
            lift_point = [0.8, 0.5]
        elif action_type == 'left':
            touch_point = [0.8, 0.5]
            lift_point = [0.2, 0.5]
    elif action_type in ['navigate_back', 'press_back', 'back']:
        action_type_new = 5
    elif action_type in ['navigate_home', 'press_home', 'home']:
        action_type_new = 6
    elif action_type in ['enter', 'press_enter']:
        action_type_new = 7
    elif action_type == 'press_key' and step_data['key'].lower() == 'enter':
        action_type_new = 7
    elif action_type == 'status':
        if step_data['goal_status'] == 'successful':
            action_type_new = 10
        else: action_type_new = 11
    elif action_type == 'status_complete':
        action_type_new = 10
    elif action_type == 'status_impossible':
        action_type_new = 11
    else:
        action_type_new = 99

    action = {"action_type": action_type_new, "action_name": action_id2text[action_type_new], "touch_point": touch_point, "lift_point": lift_point,
              "typed_text": typed_text}

    # action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    # action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    action["typed_text"] = action["typed_text"].lower()

    return action