# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, deque
from typing import Optional
import numpy as np

from framework.Logger import logger

"""
Score Validation

While a score model may perform flawlessly in a simulated environment, it can struggle under real-world conditions
like inconsistent lighting, resolution differences, misalignment, or camera artifacts.

Atari scores and lives are rendered directly as part of the game screen using low-res pixel tiles.
They're often blurry, smudged, or affected by alignment and scale—leading to errors like:
 - extra leading zeros
 - dropped digits
 - overreliance on position instead of digit shape

To harden model output, we enforce constraints:
 - Per-digit confidence must be high
 - Score must increase monotonically (unless reset)
 - Jumps between scores must be from a valid set of increments
 - Score drops must be justified (e.g., episode reset)

Goals:
- Use temporal history to reject invalid predictions
- Handle false positives and recover cleanly
- Enforce consistency over time
- Reduce risk of locking into bad scores

Validation methods:
- Temporal debounce: accept a new score only if seen consistently across frames
- Increment validation: only allow known good score jumps
- Oscillation detection: detect back-and-forth errors
- Per-digit stability checks
- Dead-man switch: force revalidation if score doesn't change for too long

Some additional mechanisms (e.g. stabilization windows, debounce counters, cooldown timers) are stubbed
but currently disabled. These may be reactivated or refined.

The validator is intended to reduce false positives, improve robustness, and ensure more accurate tracking of score and lives over time.
"""


class ScoreValidator:
    def __init__(self, env_name, valid_jumps, displayed_lives, entropy_threshold, entropy_ceiling):
        self.env_name = env_name
        self.valid_jumps = set(valid_jumps)
        assert self.valid_jumps
        self.min_jump = min(x for x in self.valid_jumps)  # if x != 0)
        self.max_jump = max(self.valid_jumps)
        self.displayed_lives = displayed_lives

        self.frame_id = -1

        self.max_history_length = 5

        # for the games we support, initial score = 0
        # and initial lives = displayed_lives
        self.history = []
        self.prev_lives = self.displayed_lives

        self.recovery_period = 6
        if self.env_name == "battle_zone":
            self.recovery_period = 16
        elif self.env_name == 'centipede' or self.env_name == 'krull':
            self.recovery_period = 8
        self.recovery_mode = False
        self.recovery_history = []

        self.zero_stabilization_period = 3
        self.zero_buffer = deque(maxlen=self.zero_stabilization_period)
        self.last_reset_frame = None
        self.cooldown_frames = 16
        self.zero_accept_delay = 5
        self.reset_stabilizer = []
        self.reset_stabilizer_period = 4
        self.debounce_frames = 3

        self.entropy_allow_hot = 2
        if self.env_name == 'battle_zone':
            self.entropy_allow_hot = 3
        self.entropy_threshold_ceiling = entropy_ceiling
        self.entropy_threshold = entropy_threshold

        # adjust for domain shift in physical (this can be removed when
        # fine-tuning or training with samples from physical)
        # REVIEW: test with other games before enabling across the board
        if self.env_name == 'battle_zone':
            if entropy_threshold < 0.15:
                delta = max(0.02, 0.4 * entropy_threshold)
            else:
                delta = 0.10 if entropy_threshold < 0.4 else 0.15
            self.entropy_threshold = min(entropy_threshold + delta, entropy_ceiling - 0.2)
            logger.debug(f"entropy_threshold={entropy_threshold:4f} -> {self.entropy_threshold:4f}")

        # validation stats
        self.validation_warnings = defaultdict(int)

    def _log_validation_warning(
        self, lives, predicted_score, prev_score, score_entropy, maybe_valid, reason, extra=None
    ):
        entropy_str = " ".join(f"{x:>5.3f}" for x in score_entropy)
        reason_verbose = reason
        dt = (predicted_score - prev_score) if maybe_valid and predicted_score is not None else 0
        lives_str = f"{lives}" if lives is not None else "n/a"
        if extra is not None:
            reason_verbose += f":{extra}"
        logger.debug(
            "score_validation: "
            f"lives: {lives_str} | "
            f"pred: {str(predicted_score):<6} "
            f"accepted: {str(predicted_score if maybe_valid else prev_score):<6} | "
            f"entropy: {entropy_str} | "
            f"dt: {dt:<4} | "
            f"{'VALID(?)' if maybe_valid else 'INVALID'} |"
            f"Reason: {reason_verbose}"
        )
        self.validation_warnings[reason] += 1

    def _is_lives_valid(self, lives, lives_confidences) -> bool:
        # A prediction of None can be interpreted as undetectable, or unavailable for this game.
        if lives is None:
            return False

        # or len(lives_confidences) == 0 or (len(lives_confidences) & 1) != 0
        lives_uncertain = any(entropy > self.entropy_threshold for entropy in lives_confidences)
        if lives_uncertain:
            # logger.debug(f"score_validation: lives={lives} uncertain={lives_confidences}")
            # return False
            pass
        return True

    def _is_score_uncertain(self, score_confidence):
        if len(score_confidence) == 0:
            return True, "blank_entropy"

        mean_entropy = np.mean(score_confidence)
        num_high = sum(e > self.entropy_threshold for e in score_confidence)

        score_uncertain = False
        reason = None

        # score_uncertain = (
        #    any(entropy > self.entropy_threshold for entropy in score_confidence) or len(score_confidence) == 0
        # )

        if mean_entropy > self.entropy_threshold:
            score_uncertain = True
            reason = f"mean_entropy {mean_entropy:.4f} > threshold {self.entropy_threshold:.4f}"
        elif num_high >= self.entropy_allow_hot:
            score_uncertain = True
            reason = f"{num_high} digits > entropy threshold {self.entropy_threshold:.4f}"

        return score_uncertain, reason

    # use the history buffer to determine if the predicted score is valid
    # based on temporal consistency, and is within expected min,max increment range.
    def _is_score_valid(self, score, score_confidence, lives=None) -> bool:
        # get the last valid score
        prev_score = self.history[-1] if len(self.history) > 0 else 0
        prev_lives = self.prev_lives

        # A prediction of None can be interpreted as undetectable.
        if score is None:
            self._log_validation_warning(lives, score, prev_score, score_confidence, False, "blank_prediction")
            return False

        # check confidence levels first
        score_uncertain, reason = self._is_score_uncertain(score_confidence)
        if score_uncertain:
            self._log_validation_warning(lives, score, prev_score, score_confidence, False, reason)
            return False

        # reset cooldown to prevent duplicate reset events
        if False:  # self.last_reset_frame is not None and self.frame_id is not None:
            if self.frame_id - self.last_reset_frame < self.cooldown_frames:
                self._log_validation_warning(lives, score, prev_score, score_confidence, False, "reset_cooldown")
                return False

        # check for episode reset
        score_reset = score == 0 and prev_score > 0
        lives_reset = False if lives is None else (lives > prev_lives)

        # if score or lives, stabilize over multiple frames
        if False:  # score_reset:
            self.zero_buffer.append(score)
            if len(self.zero_buffer) >= self.zero_stabilization_period or lives_reset:
                self._log_validation_warning(
                    lives,
                    score,
                    prev_score,
                    score_confidence,
                    True,
                    "zero_stabilized_general",
                    extra=f"{self.zero_buffer} or lives_reset={lives_reset}",
                )
                self.zero_buffer.clear()
                # fall through to reset detection logic
            else:
                self._log_validation_warning(
                    lives,
                    score,
                    prev_score,
                    score_confidence,
                    False,
                    "zero_stabilization_wait",
                    extra=f"{self.zero_buffer}",
                )
                return False

        if score_reset or lives_reset:
            logger.debug(
                f"Predict score: prev_score:{prev_score} curr_score:{score} prev_lives:{prev_lives} lives:{lives}: potential game reset detected"
            )
            self.history = []
            self.prev_lives = self.displayed_lives
            self.zero_buffer.clear()
            self.recovery_mode = False
            self.recovery_history.clear()
            self.last_reset_frame = self.frame_id
            self.reset_stabilizer = [score]
            return True

        # stabilization post-reset
        # if self.last_reset_frame is not None and self.frame_id is not None:
        #     if self.frame_id - self.last_reset_frame < self.zero_accept_delay:
        #         self.reset_stabilizer.append(score)
        #         if len(self.reset_stabilizer) >= self.reset_stabilizer_period:
        #             if all(s == 0 for s in self.reset_stabilizer):
        #                 self._log_validation_warning(lives, score, prev_score, score_confidence, True, "zero_stabilized", extra=f"{self.reset_stabilizer}")
        #                 return True
        #             else:
        #                 self._log_validation_warning(lives, score, prev_score, score_confidence, False, "zero_stabilizer_failed", extra=f"{self.reset_stabilizer}")
        #                 return False
        #         else:
        #             self._log_validation_warning(lives, score, prev_score, score_confidence, False, "waiting_zero_stabilizer", extra=f"{self.reset_stabilizer}")
        #             return False

        # if in recovery mode, check for stable score
        if self.recovery_mode:
            self.recovery_history.append(score)

            if len(self.recovery_history) >= self.recovery_period:
                # check for potential reset caught in recovery
                if all(s == 0 for s in self.recovery_history):
                    logger.debug(
                        f"recovery: potential game reset missed: {self.recovery_history}: lives={lives if lives is not None else 'None'} > {prev_lives}"
                    )
                    self.recovery_mode = False
                    self.recovery_history.clear()
                    return True

                # determine if the predicted score is stable
                if len(set(self.recovery_history)) == 1:
                    if score < prev_score:
                        self._log_validation_warning(
                            lives,
                            score,
                            prev_score,
                            score_confidence,
                            True,
                            "recovery_mode",
                            extra=f"accept: prev_score={prev_score} was likely incorrect",
                        )
                    else:
                        self._log_validation_warning(
                            lives,
                            score,
                            prev_score,
                            score_confidence,
                            True,
                            "recovery_mode",
                            extra=f"accept:{self.recovery_history}",
                        )
                    self.recovery_mode = False
                    self.recovery_history.clear()
                    return True

                # determine if increments have been stable
                deltas = [b - a for a, b in zip(self.recovery_history[:-1], self.recovery_history[1:])]
                # include how the current pred would affect stability
                if self.recovery_history and score is not None:
                    deltas.append(int(score) - int(self.recovery_history[-1]))

                if all(0 <= delta <= self.max_jump for delta in deltas):
                    self._log_validation_warning(
                        lives, score, prev_score, score_confidence, True, "stable_increments", extra=f"deltas={deltas}"
                    )
                    self.recovery_mode = False
                    self.recovery_history.clear()
                    return True

                # drop oldest to maintain sliding window
                self.recovery_history.pop(0)

            # self._log_validation_warning(lives, score, prev_score, score_confidence,
            #                             False, "recovery_mode", extra=f"waiting:{self.recovery_history}")
            return False

        # NOTE: we will need to revisit the following consistency checks if we support games which
        # allow negative rewards.
        if score < prev_score:
            self.recovery_mode = True
            self.recovery_history = [score]
            self._log_validation_warning(
                lives, score, prev_score, score_confidence, False, "invalid_decrease", extra=f"[{score} < {prev_score}]"
            )
            return False

        # test for valid increments
        if len(self.history) == 0:
            if score == 0:
                return True

            # make sure the first score is not too large or small
            if score < self.min_jump or score > self.max_jump:
                self.recovery_mode = True
                self.recovery_history = [score]
                self._log_validation_warning(
                    lives,
                    score,
                    prev_score,
                    score_confidence,
                    False,
                    "invalid_increment",
                    extra=f"allowed_range=[{self.min_jump},{self.max_jump}]",
                )
                return False
            return True

        # Test for an invalid score jump
        if True:  # len(self.history) < 2:
            score_change = abs(score - prev_score)
        else:
            ave_score = np.mean(self.history)
            score_change = abs(score - ave_score)

        if score_change != 0 and (score_change < self.min_jump or score_change > self.max_jump):
            self.recovery_mode = True
            self.recovery_history = [score]
            self._log_validation_warning(
                lives,
                score,
                prev_score,
                score_confidence,
                False,
                "invalid_increment",
                extra=f"allowed_range=[{self.min_jump},{self.max_jump}]",
            )
            return False

        return True

    def validate(self, pred_score, score_confidence, pred_lives=None, lives_confidences=None) -> tuple[int, Optional[int]]:
        self.frame_id += 1

        lives_valid = self._is_lives_valid(pred_lives, lives_confidences)
        if lives_valid:
            valid_lives = pred_lives
        else:
            valid_lives = self.prev_lives

        score_valid = self._is_score_valid(pred_score, score_confidence, lives=valid_lives)
        if score_valid:
            valid_score = pred_score
        else:
            # get the last valid score
            valid_score = self.history[-1] if len(self.history) > 0 else 0

        # after all validity checks have been run, update history
        if lives_valid:
            self.prev_lives = valid_lives

        if score_valid:
            # add the score to the valid prediction history
            self.history.append(valid_score)
            if len(self.history) > self.max_history_length:
                self.history.pop(0)

        return valid_score, valid_lives
