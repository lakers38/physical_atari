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


import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.Logger import logger

# Based on the CRNN described here: https://arxiv.org/pdf/1507.05717
# CTC-based (need to collapse repeats and filter blank tokens) where
# the output is decoded as a sequence of class predictions up to a
# max length (or until padding is encountered)


class CRNN(nn.Module):
    def __init__(self, num_classes=11, hidden_size=128, cnn_dropout_rate=0.3, rnn_dropout_rate=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout_rate),
        )

        self.rnn = nn.GRU(input_size=128, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(rnn_dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W)
        x = self.dropout(x)

        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, S, C)

        x, _ = self.rnn(x)  # (B, S, 2*hidden)
        x = self.fc(x)  # (B, S, num_classes)
        return x.permute(0, 2, 1)  # (B, num_classes, S)


def greedy_decode_ctc(
    logits: torch.Tensor,
    max_length: int,
    padding_index=-1,
    blank_index=1,
    temperature=1.0,
    decoded_buf=None,
    confidences_buf=None,
):

    # Apply temperature scaling before softmax
    logits = logits / temperature

    # logits: [B, num_classes, S]
    logprobs = F.log_softmax(logits, dim=1)
    preds = torch.argmax(logprobs, dim=1)  # [B, S]

    # remove repeats and blanks
    prev_preds = torch.cat(
        [torch.full((preds.size(0), 1), fill_value=-1, dtype=preds.dtype, device=preds.device), preds[:, :-1]], dim=1
    )
    mask = (preds != prev_preds) & (preds != blank_index)  # [B, S]

    if decoded_buf is not None:
        decoded = decoded_buf
    else:
        decoded = torch.full((preds.shape[0], max_length), padding_index, dtype=torch.int32, device=logprobs.device)

    probs = F.softmax(logits, dim=1)  # [B, num_classes, S]
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)  # [B, S]

    if confidences_buf is not None:
        confidences = confidences_buf
    else:
        confidences = torch.zeros((preds.shape[0], max_length), dtype=torch.float32, device=logprobs.device)

    for b in range(preds.size(0)):
        valid_indices = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            continue
        seq = preds[b, valid_indices]
        n = min(seq.numel(), max_length)
        decoded[b, :n] = seq[:n]
        confidences[b, :n] = entropy[b, valid_indices[:n]]

    return decoded, confidences


class CRNNDecoder(CRNN):
    def __init__(
        self,
        num_classes=12,
        hidden_size=128,
        max_digits=6,
        score_mean=0.1307,
        score_std=0.113,
        lives_mean=0.1307,
        lives_std=0.113,
        blank_index=11,
        padding_index=-1,
        life_symbol=10,
        image_size=32,
        entropy_threshold=1.0,
        entropy_ceiling=1.0,
        temperature=1.0,
        device='cpu',
        use_mixed_precision=False,
    ):
        super().__init__(num_classes=num_classes, hidden_size=hidden_size)

        self.use_mixed_precision = use_mixed_precision
        self.device = device

        self.score_mean = torch.tensor([score_mean], dtype=torch.float32, device=self.device)
        self.score_std = torch.tensor([score_std], dtype=torch.float32, device=self.device)
        self.lives_mean = torch.tensor([lives_mean], dtype=torch.float32, device=self.device)
        self.lives_std = torch.tensor([lives_std], dtype=torch.float32, device=self.device)

        self.blank_idx = blank_index
        self.padding_idx = padding_index
        self.life_symbol = life_symbol
        self.life_symbol_str = str(self.life_symbol)

        self.entropy_threshold = entropy_threshold
        self.entropy_ceiling = entropy_ceiling
        self.temperature = temperature

        self.image_size = image_size
        self.decode_max_length = max_digits
        self.input_dims = (self.image_size, self.image_size * self.decode_max_length)

        self._decoded_buf = None
        self._confidences_buf = None

    def preprocess(self, x: np.ndarray, is_lives=False):
        crop_t = torch.from_numpy(x).permute(2, 0, 1).float().div(255.0)

        if crop_t.shape[0] == 3:
            crop_t = 0.2989 * crop_t[0] + 0.5870 * crop_t[1] + 0.1140 * crop_t[2]
            crop_t = crop_t.unsqueeze(0)

        crop_t = F.interpolate(crop_t.unsqueeze(0), size=self.input_dims, mode='bilinear', align_corners=False).squeeze(
            0
        )

        mean = self.score_mean if not is_lives else self.lives_mean
        std = self.score_std if not is_lives else self.lives_std
        crop_t = crop_t.to(self.device)
        crop_t = (crop_t - mean) / std

        if self.use_mixed_precision:
            crop_t = crop_t.half()

        return crop_t

    def decode_ctc(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = self.decode_max_length
        padding_index = self.padding_idx
        blank_index = self.blank_idx

        if self._decoded_buf is None:
            self._decoded_buf = torch.empty((logits.shape[0], max_length), dtype=torch.int32, device=logits.device)

        decoded = self._decoded_buf
        decoded.fill_(padding_index)

        if self._confidences_buf is None:
            self._confidences_buf = torch.empty(
                (logits.shape[0], max_length), dtype=torch.float32, device=logits.device
            )

        confidences = self._confidences_buf
        confidences.zero_()

        return greedy_decode_ctc(
            logits,
            max_length,
            padding_index,
            blank_index,
            decoded_buf=decoded,
            confidences_buf=confidences,
            temperature=self.temperature,
        )

    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        return self.decode_ctc(logits)

    def convert(self, score_pred, score_confidence, lives_pred=None, lives_confidence=None):
        """
        Some games may not display score or lives for periods of time, so we need to distinguish
        between true-zero and not-visible states. True zero is mostly an issue with lives, where
        the absence of life symbols means no lives remain in several supported games.
        This distinction is more complex in games like Q*bert, where both cases occur.

        With CTC decoding, a blank prediction should indicate no score or lives displayed. However,
        the model may sometimes be overconfident and predict noise instead.

        Game-specific quirks:
            Centipede: The score strobe intermittently; when off, a blank prediction means not-visible.
            Q*bert: Both lives and score flash off together for extended periods; blank predictions for both indicate not-visible.
            Q*bert (and others): Absence of life symbols means true zero; if the score prediction is not blank, return zero.
            Atlantis: Lives detection is non-trivial; lives prediction is expected to be None.

        To handle both cases for lives, return None for not-visible and zero for true zero.
        """
        score_pred = score_pred[score_pred != self.padding_idx]
        blank_score = len(score_pred) == 0
        score = None if blank_score else int(''.join(map(str, score_pred)))
        score_conf = score_confidence[score_confidence != self.padding_idx]

        lives = None
        if lives_pred is not None:
            lives_pred = lives_pred[lives_pred != self.padding_idx]
            blank_lives = len(lives_pred) == 0

            if blank_lives and blank_score:
                lives = None
            else:
                # lives prediction will be 1 symbol = 10; 2 symbols = 1010; and so on.
                # find the number of times the symbol is repeated
                lives_str = ''.join(lives_pred.astype(str))
                lives = lives_str.count(self.life_symbol_str)

        lives_conf = None
        if lives_confidence is not None:
            lives_conf = lives_confidence[lives_confidence != self.padding_idx]
        return score, score_conf, lives, lives_conf


def fuse_cnn_layers(model):
    fused = []
    for i in range(0, len(model.cnn) - 1):
        m1 = model.cnn[i]
        m2 = model.cnn[i + 1]
        if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.ReLU):
            fused.append([str(i), str(i + 1)])

    torch.quantization.fuse_modules(model.cnn, fused, inplace=True)
    return model


def load_model(
    checkpoint: str,
    env_name: str,
    mixed_precision: bool = True,
    device: str = 'cpu',
    memory_format=torch.contiguous_format,
) -> CRNNDecoder:

    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = checkpoint_data['state_dict']
    model_config = checkpoint_data['model_config']
    game_config = checkpoint_data['game_config']
    train_config = checkpoint_data['train_config']

    score_mean, score_std = game_config["score_mean_std"]
    lives_mean, lives_std = game_config["lives_mean_std"]
    max_digits = game_config["max_digits"]

    num_classes = model_config["num_classes"]
    hidden_size = model_config["hidden_size"]
    blank_idx = model_config["blank_idx"]
    padding_idx = model_config["padding_idx"]
    life_symbol = model_config["life_symbol"]
    image_size = model_config["image_size"]
    # entropy_threshold = model_config["balanced_entropy_threshold"]
    entropy_threshold = model_config["entropy_threshold"]
    entropy_ceiling = (
        model_config["entropy_ceiling"] if "entropy_ceiling" in model_config else math.log(num_classes) * 0.75
    )
    temperature = train_config["temperature"]

    model = CRNNDecoder(
        num_classes=num_classes,
        hidden_size=hidden_size,
        max_digits=max_digits,
        blank_index=blank_idx,
        padding_index=padding_idx,
        image_size=image_size,
        life_symbol=life_symbol,
        score_mean=score_mean,
        score_std=score_std,
        lives_mean=lives_mean,
        lives_std=lives_std,
        entropy_threshold=entropy_threshold,
        entropy_ceiling=entropy_ceiling,
        temperature=temperature,
        device=device,
        use_mixed_precision=mixed_precision,
    )

    model.load_state_dict(weights)
    model = model.to(memory_format=memory_format).to(device)
    model = fuse_cnn_layers(model)
    model.eval()
    if mixed_precision:
        model = model.half()
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False, dynamic=False)
    # print(model)

    model_param_count = 0
    for p in model.parameters():
        model_param_count += p.numel()
    logger.debug(f"CRNN_CTC: model param count = {model_param_count}")

    return model
