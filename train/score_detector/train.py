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

# pip install -r train/score_detector/requirements.txt
import argparse
import json
import math
import os
import random
import shutil
import time
from typing import Union

import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .augmentations import AugmentationAnalyzer, AugmentationBuilder, save_augmented_batch
from .dataset import MultiDigitDataset, get_class_weights
from .generate_dataset import generate_data
from .preprocess_dataset import compute_train_split_stats, preprocess_data
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from framework.models.score_detector.crnn_ctc import CRNN, greedy_decode_ctc

IMAGE_SIZE = 32
# digits are classified as (0-9)
# life symbol is classified as 10
# Since we process score and lives separately:
# 0-9 is returned for the score.
# for lives with LIFE_DIGIT = 10, 0=life = 0 ,1 head = 10, 2 heads = 1010.
# the model should return 10, or 1010. then we count the number of 10s to get the number of life symbols.
LIFE_DIGIT = 10


class CropInfo:
    def __init__(self, x, y, w, h, num_digits):
        self.x_start = x
        self.y_start = y
        self.x_offset = w
        self.y_offset = h
        self.num_digits = num_digits


def get_entropy_threshold(correct_entropies, percentile=95.0, num_classes=12):
    if len(correct_entropies) >= 2:
        return np.percentile(correct_entropies, percentile)
    elif len(correct_entropies) == 1:
        return correct_entropies[0]
    else:
        print("get_entropy_threshold: not enough data, defaulting")
        return math.log(num_classes) * 0.6


def find_balanced_entropy_threshold(correct_entropies, incorrect_entropies):
    all_entropies = np.concatenate([correct_entropies, incorrect_entropies])
    thresholds = np.linspace(min(all_entropies), max(all_entropies), 100)

    best_thresh = None
    min_diff = float('inf')

    for t in thresholds:
        false_reject = np.mean(correct_entropies > t)
        false_accept = np.mean(incorrect_entropies <= t)
        diff = abs(false_reject - false_accept)
        if diff < min_diff:
            min_diff = diff
            best_thresh = t

    return best_thresh


# find the first occurrence of padding along a specific dimension
def find_first_occurrence_of_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    # no occurrence, set index to size of tensor along the specified dim
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices


def train(args, model, device, train_loader, criterion, optimizer, scheduler, epoch, padding_index=-1):
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)  # (B, num_classes, S)
        logprobs = torch.log_softmax(logits, dim=1)

        B, _, S = logprobs.shape
        logprobs_for_loss = logprobs.permute(2, 0, 1)

        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
        target_lengths = find_first_occurrence_of_element(y, padding_index).to(device)
        loss = criterion(logprobs_for_loss, y, input_lengths, target_lengths)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch:<2} [{batch_idx * len(x):>4}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):>3.0f}%)]  Loss: {loss.item():>9.6f}'
            )
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)


def test(model, device, criterion, test_loader, padding_index=-1, blank=10, temperature=1.0, debug=0):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    correct_confidences = []
    incorrect_confidences = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            logprobs = torch.log_softmax(logits, dim=1)
            B, _, S = logprobs.shape
            logprobs_for_loss = logprobs.permute(2, 0, 1)

            input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S
            target_lengths = find_first_occurrence_of_element(y, padding_index).to(device)

            loss = criterion(logprobs_for_loss, y, input_lengths, target_lengths)
            test_loss += loss.item()

            decoded, confidences = greedy_decode_ctc(
                logits, y.shape[1], padding_index=padding_index, blank_index=blank, temperature=temperature
            )

            incorrect_indices = []
            correct_indices = []
            for i in range(B):
                seq = decoded[i].cpu().numpy()
                target = y[i].cpu().numpy()
                conf = confidences[i].cpu().numpy()

                valid_pred = seq[seq != padding_index]
                valid_target = target[target != padding_index]
                valid_conf = conf[seq != padding_index]

                if len(valid_pred) == len(valid_target) and np.array_equal(valid_pred, valid_target):
                    correct += 1
                    correct_confidences.extend(valid_conf.tolist())
                    correct_indices.append(i)
                else:
                    incorrect_confidences.extend(valid_conf.tolist())
                    incorrect_indices.append(i)

                total += 1

                if len(valid_pred) < len(valid_target):
                    valid_pred = np.pad(valid_pred, (0, len(valid_target) - len(valid_pred)), constant_values=-1)

                all_preds.append(valid_pred)
                all_targets.append(valid_target)

            if debug:
                if False:  # len(correct_indices) > 0:
                    t_indices = torch.tensor(correct_indices)
                    # incorrect_x = torch.index_select(x.cpu(), 0, t_indices)
                    incorrect_y = torch.index_select(y.cpu(), 0, t_indices)
                    incorrect_decoded = torch.index_select(decoded.cpu(), 0, t_indices)
                    incorrect_confidences = torch.index_select(confidences.cpu(), 0, t_indices)
                    print(incorrect_confidences)
                    # RenderImages((np.clip(incorrect_x.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8), incorrect_y.cpu().numpy(), incorrect_decoded.cpu().numpy(), cmap="gray")

                if len(incorrect_indices) > 0:
                    t_indices = torch.tensor(incorrect_indices)
                    # incorrect_x = torch.index_select(x.cpu(), 0, t_indices)
                    incorrect_y = torch.index_select(y.cpu(), 0, t_indices)
                    incorrect_decoded = torch.index_select(decoded.cpu(), 0, t_indices)
                    # incorrect_confidences = torch.index_select(confidences.cpu(), 0, t_indices)
                    # print(incorrect_confidences)
                    # RenderImages((np.clip(incorrect_x.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8),
                    # incorrect_y.cpu().numpy(), incorrect_decoded.cpu().numpy(), cmap="gray")
                    # print(f"Incorrect: {incorrect_y.cpu().numpy()}")
                    # print(f"Decoded: {incorrect_decoded}")
                    print("Incorrect predictions:")
                    for idx in range(len(incorrect_y)):
                        target_seq = [d.item() for d in incorrect_y[idx] if d.item() != padding_index]
                        pred_seq = [d.item() for d in incorrect_decoded[idx] if d.item() != padding_index]
                        target_str = ''.join(str(d) for d in target_seq)
                        pred_str = ''.join(str(d) for d in pred_seq)
                        print(
                            f"  Target: {target_str:<10} -> Predicted: {pred_str:<10} | target_len={len(target_seq)}, pred_len={len(pred_seq)}"
                        )

                if False:  # confidences is not None:
                    t_indices = torch.tensor(correct_indices)
                    correct_confidences = torch.index_select(confidences.cpu(), 0, t_indices)
                    for i, c in enumerate(correct_confidences):
                        valid_conf = c[decoded[i].cpu() != padding_index]
                        low_conf = valid_conf < 0.89
                        if low_conf.any():
                            value = decoded[i].cpu()
                            print(f"CORRECT has low_conf: {valid_conf} for value {value} ")

    num_batches = len(test_loader)
    test_loss /= num_batches

    # compute metrics
    cer_list = []
    total_chars = 0
    correct_chars = 0
    for pred, target in zip(all_preds, all_targets):
        pred_str = ''.join(map(str, pred))
        target_str = ''.join(map(str, target))
        cer = editdistance.eval(pred_str, target_str) / max(1, len(target_str))
        cer_list.append(cer)

        # char accuracy
        match_len = min(len(pred), len(target))
        correct_chars += sum(p == t for p, t in zip(pred[:match_len], target[:match_len]))
        total_chars += len(target)

    avg_cer = sum(cer_list) / len(cer_list)
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    mean_conf = (
        np.mean(correct_confidences + incorrect_confidences) if (correct_confidences or incorrect_confidences) else 0.0
    )

    print(
        f'\nTest set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / total:.2f}%)'
    )
    print(f"CER: {avg_cer:.4f}, Char Acc: {char_acc:.4f}, Mean Conf: {mean_conf:.4f}")

    return test_loss, correct / total, avg_cer, char_acc, mean_conf, correct_confidences, incorrect_confidences


def get_game_config(game):
    config = {
        'use_class_balancing': False,
        'cnn_drop_out_rate': 0.3,
        'rnn_drop_out_rate': 0.3,
        'weight_decay': 1e-5,
        'patience': 5,
        'factor': 0.1,
        'min_lr': 1e-5,
        'temperature': 1.0,
        'early_stop_patience': 5,
        'score_aug_scale': 0.3,
        'lives_aug_scale': 0.3,
        'epochs': 100,
        'batch_size': 32,
        'test_batch_size': 128,
        'lr': 1e-4,
    }

    if game == 'atlantis':
        config.update(
            {
                'use_class_balancing': True,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.3,
                'batch_size': 8,
                'lr': 0.0003,
                'cnn_drop_out_rate': 0.2,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-8,
                'temperature': 1.0,
                'early_stop_patience': 10,
            }
        )
    elif game == 'battle_zone':
        """
        {
            'use_class_balancing': False,
            'score_aug_scale': 0.15,
            'lives_aug_scale': 0.075,
            'batch_size': 8,
            'lr': 0.0003,
            'cnn_drop_out_rate': 0.0,  # skinny fonts
            'rnn_drop_out_rate': 0.2,
            'weight_decay': 1e-4,
            'patience': 5,
            'factor': 0.3,
            'min_lr': 1e-6,
            'temperature': 1.5,
            'early_stop_patience': 8,
        }
        """
        config.update(
            {
                'use_class_balancing': False,
                'score_aug_scale': 0.2,  # .15
                'lives_aug_scale': 0.075,
                'batch_size': 8,
                'lr': 0.0003,
                'cnn_drop_out_rate': 0.0,  # skinny fonts
                'rnn_drop_out_rate': 0.2,
                'weight_decay': 1e-4,
                'patience': 5,
                'factor': 0.3,
                'min_lr': 1e-6,
                'temperature': 1.5,
                'early_stop_patience': 10,
            }
        )
    elif game == 'centipede':
        config.update(
            {
                'use_class_balancing': False,
                'cnn_drop_out_rate': 0.1,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 5,
                'factor': 0.3,
                'min_lr': 1e-5,
                'temperature': 1.0,
                'early_stop_patience': 10,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.3,
                'batch_size': 16,
                'lr': 0.0003,
            }
        )
    elif game == 'defender':
        config.update(
            {
                'use_class_balancing': False,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.15,
                'batch_size': 16,
                'cnn_drop_out_rate': 0.15,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 5,
                'factor': 0.3,
                'min_lr': 1e-5,
                'temperature': 1.3,
                'early_stop_patience': 8,
                'lr': 0.0003,
            }
        )
    elif game == 'krull':
        config.update(
            {
                'use_class_balancing': False,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.3,
                'batch_size': 8,
                'lr': 0.0003,
                'cnn_drop_out_rate': 0.2,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 3e-4,
                'patience': 5,
                'factor': 0.3,
                'min_lr': 1e-5,
                'temperature': 1.5,
                'early_stop_patience': 5,
            }
        )
    elif game == 'ms_pacman':
        config.update(
            {
                'use_class_balancing': False,
                'cnn_drop_out_rate': 0.1,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 3,
                'factor': 0.3,
                'min_lr': 1e-5,
                'temperature': 1.0,
                'early_stop_patience': 8,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.3,
                'batch_size': 16,
                'lr': 0.0003,
            }
        )
    elif game == 'qbert':
        config.update(
            {
                'use_class_balancing': False,
                'cnn_drop_out_rate': 0.1,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 3,
                'factor': 0.3,
                'min_lr': 1e-5,
                'temperature': 1.45,
                'early_stop_patience': 5,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.3,
                'batch_size': 16,
            }
        )
    elif game == 'up_n_down':
        config.update(
            {
                'use_class_balancing': True,
                'score_aug_scale': 0.1,
                'lives_aug_scale': 0.1,
                'batch_size': 8,
                'lr': 0.0001,
                'cnn_drop_out_rate': 0.1,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-8,
                'temperature': 1.0,
                'early_stop_patience': 10,
            }
        )
    else:
        config.update(
            {
                'use_class_balancing': True,
                'score_aug_scale': 0.3,
                'lives_aug_scale': 0.3,
                'batch_size': 8,
                'lr': 0.0003,
                'cnn_drop_out_rate': 0.2,
                'rnn_drop_out_rate': 0.3,
                'weight_decay': 1e-4,
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-8,
                'temperature': 1.0,
                'early_stop_patience': 10,
            }
        )

    return config


# if training data already generated: specify --data_dir '{data}/{game}' otherwise training data will be auto-generated.
# python train/score_detector/train.py --game_config 'configs/games/ms_pacman.json' --output_dir 'assets/models'
def main():
    parser = argparse.ArgumentParser(description='Atari multi-digit training')
    parser.add_argument('--game_config', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--log_interval', type=int, default=10)
    args, _ = parser.parse_known_args()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
    # but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case,
    # you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(False)
    # ctc_decode is not supported deterministically on the gpu
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(args.game_config) as gf:
        game_data = gf.read()

    game_config = json.loads(game_data)["game_config"]
    game = game_config["name"]
    total_lives = game_config["lives"]
    score_config = json.loads(game_data)["score_config"]

    score_crop_info = score_config.get("score_crop_info")
    lives_crop_info = score_config.get("lives_crop_info")

    score_crop_info = CropInfo(**score_crop_info) if score_crop_info else None
    lives_crop_info = CropInfo(**lives_crop_info) if lives_crop_info else None

    capture_lives = lives_crop_info is not None

    if capture_lives:
        model_name = os.path.join(args.output_dir, f"{game}_score_lives.pt")
    else:
        model_name = os.path.join(args.output_dir, f"{game}_score.pt")

    # since the life-digit is actually 2 digits '10', we need to multiply num_lives * 2
    max_digits = max(score_crop_info.num_digits, 0 if lives_crop_info is None else lives_crop_info.num_digits * 2)
    image_size = IMAGE_SIZE

    config = get_game_config(game)

    use_class_balancing = config['use_class_balancing']
    cnn_drop_out_rate = config['cnn_drop_out_rate']
    rnn_drop_out_rate = config['rnn_drop_out_rate']
    weight_decay = config['weight_decay']
    patience = config['patience']
    factor = config['factor']
    min_lr = config['min_lr']
    temperature = config['temperature']
    early_stop_patience = config['early_stop_patience']
    score_aug_scale = config['score_aug_scale']
    lives_aug_scale = config['lives_aug_scale']

    epochs = config['epochs']
    lr = config['lr']
    batch_size = config['batch_size']
    test_batch_size = config['test_batch_size']

    print("\n[Train Parameters]")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  lr: {lr}")

    print(f"  cnn_drop_out_rate: {cnn_drop_out_rate}")
    print(f"  rnn_drop_out_rate: {rnn_drop_out_rate}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  patience: {patience}")
    print(f"  early_stop_patience: {early_stop_patience}")
    print(f"  factpr: {factor}")
    print(f"  min_lr: {min_lr}")
    print(f"  temperature: {temperature}")
    print(f"  use_class_balancing: {use_class_balancing}")

    num_classes = 12  # 10 digits (0-9) + 1 life symbol (LIFE_DIGIT) + 1 blank for CTC
    blank_index = 11  # CTC loss - set blank to the number of classes
    assert blank_index != LIFE_DIGIT
    hidden_size = 128
    padding_index = -1

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': min(4, os.cpu_count() or 1),
            'pin_memory': True,
            'shuffle': True if not use_class_balancing else False,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.Resize((image_size, image_size * max_digits)), transforms.Grayscale()])

    # process the data
    if args.data_dir is not None:
        assert os.path.exists(args.data_dir)
        train_dir = f"{args.data_dir}/train"
        test_dir = f"{args.data_dir}/test"
    else:
        game_dir = os.path.join(os.getcwd(), "train_data", f"{game}")
        print(f"generating data at: {game_dir}")
        frame_dir = os.path.join(game_dir, "frames")
        generate_data(game, frame_dir)
        train_dir, test_dir = preprocess_data(
            frame_dir, game_dir, game, total_lives, score_config, auto_balance=(not use_class_balancing)
        )

    # create dir to store debug augmentation images
    augmentation_dir = os.path.join(os.getcwd(), "train_data", f"{game}", "augmentations")
    if os.path.exists(augmentation_dir):
        shutil.rmtree(augmentation_dir)
    os.makedirs(augmentation_dir)

    score_stats, lives_stats = compute_train_split_stats(
        train_dir, image_width=IMAGE_SIZE, image_height=IMAGE_SIZE * max_digits
    )

    # If real-world (camera-captured) images are available, a lightweight analysis of brightness,
    # contrast, blur, and noise is performed to adapt the augmentation pipeline for better domain alignment.
    analyzer = AugmentationAnalyzer(train_dir=train_dir, camera_dir=None)
    results = analyzer.analyze()

    builder = AugmentationBuilder(game, results)
    aug_transforms = builder.build_transforms(score_aug_scale, lives_aug_scale)

    score_transform = aug_transforms['score']
    lives_transform = aug_transforms['lives']

    # create test/train datasets
    train_dataset = MultiDigitDataset(
        train_dir,
        max_digits,
        transform=transform,
        transform_score=score_transform,
        transform_lives=lives_transform,
        score_meanstd=score_stats,
        lives_meanstd=lives_stats,
        padding_value=padding_index,
    )
    test_dataset = MultiDigitDataset(
        test_dir,
        max_digits,
        transform=transform,
        score_meanstd=score_stats,
        lives_meanstd=lives_stats,
        padding_value=padding_index,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    if use_class_balancing:
        train_weights = get_class_weights(train_dataset, num_classes)
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True, generator=g)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, generator=g, **train_kwargs)

        # sampled_labels = [train_dataset.labels[i] for i in list(sampler)[:1000]]
        # label_counts = Counter(sampled_labels)
        # print("Sampled label distribution (first 1000):")
        # for k in sorted(label_counts, key=lambda x: int(x)):
        #    print(f"Class {k}: {label_counts[k]}")
        # visualize_sampler_distribution(train_dataset, sampler, num_samples=2000)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, generator=g, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = CRNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
        cnn_dropout_rate=cnn_drop_out_rate,
        rnn_dropout_rate=rnn_drop_out_rate,
    ).to(device)
    criterion = nn.CTCLoss(blank=blank_index, reduction='mean', zero_infinity=True)

    print("Starting Training...")
    start_time = time.time()
    train_losses = []
    test_losses = []
    accuracy = []
    lrs = []
    entropy_vals = []
    cer = []
    char_acc = []
    mean_confidences = []

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", min_lr=min_lr, factor=factor, patience=patience
    )

    best_cer = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        if epoch % 4 == 0:
            save_augmented_batch(train_dataset, data_dir=augmentation_dir, epoch=epoch, num_samples=batch_size)

        train_loss = train(
            args, model, device, train_loader, criterion, optimizer, None, epoch, padding_index=padding_index
        )
        test_loss, test_acc, test_cer, test_char_acc, mean_conf, correct_confidences, incorrect_confidences = test(
            model,
            device,
            criterion,
            test_loader,
            padding_index=padding_index,
            blank=blank_index,
            temperature=temperature,
            debug=False,
        )
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"scheduler: lr reduced from {prev_lr:.2e} to {new_lr:.2e}")

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracy.append(test_acc)
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        cer.append(test_cer)
        char_acc.append(test_char_acc)
        mean_confidences.append(mean_conf)

        """
        cer: starts high, decreases steadily, plateaus as learning converges (inverse to accuracy)
            - stuck high: model not learning or ctc decoding failing
            - fluctuations in later epoch: overfitting or unstable learning
        cer_acc: starts low, consistent increase over time
            - smooths or plateaus near 0.95-1 if training successful
            - stuck or dropping: overfitting, bad label alignment, or poor confidence calibration
            - doesn't rise, potential issue in label lengths, padding, or input formatting
        mean confidence:
            - start low 0.2-0.4 and rise steadily as model becomes more certain
            - plateaus near 0.9+ ideally not pegged at 1 unless overconfident

        high confidence + low accuracy: model is confidently wrong, possible overfitting or bad decoding.
        flat confidence = issue in how confidence is computed or bad activations
        """

        if epoch > 1:
            if train_losses[-1] < train_losses[-2] and test_losses[-1] > test_losses[-2]:
                print(
                    f"[OVERFITTING WARNING] Epoch {epoch}: Train loss decreasing ({train_losses[-2]:.4f} -> {train_losses[-1]:.4f}), "
                    f"Test loss increasing ({test_losses[-2]:.4f} -> {test_losses[-1]:.4f})"
                )

            if accuracy[-1] < accuracy[-2] and train_losses[-1] < train_losses[-2]:
                print(
                    f"[OVERFITTING WARNING] Epoch {epoch}: Accuracy decreasing ({accuracy[-2] * 100:.2f}% -> {accuracy[-1] * 100:.2f}%), "
                    f"Train loss decreasing ({train_losses[-2]:.4f} -> {train_losses[-1]:.4f})"
                )

        if test_cer < best_cer:
            best_cer = test_cer
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping: CER hasn't improved for {early_stop_patience} epochs.")
            break

    print(f"Train end: {(time.time() - start_time) / 60:.2f}m")

    if entropy_vals:
        percent_high_entropy = np.mean(np.array(entropy_vals) > 1.0)
    else:
        percent_high_entropy = 0.0

    # if it's low (< 5%), then the model is accurate and confident
    print(f"{percent_high_entropy * 100:.2f}% of predictions have entropy > 1.0")

    # use a high percentile since these values are based on a clean test set (less noise than runtime)
    threshold_percentile = 98
    threshold = get_entropy_threshold(correct_confidences, percentile=threshold_percentile, num_classes=num_classes)

    # only use balanced threshold if incorrect_confidences has enough data
    if len(incorrect_confidences) >= 10:
        balanced_threshold = find_balanced_entropy_threshold(
            correct_confidences,
            incorrect_confidences,
        )
        # fallback
        if balanced_threshold is None:
            balanced_threshold = threshold
    else:
        # fallback if incorrect_confidences too sparse
        balanced_threshold = threshold

    base_ceiling = get_entropy_threshold(correct_confidences, percentile=99.5, num_classes=num_classes) + 0.05
    entropy_ceiling = max(base_ceiling, threshold + 0.3, math.log(num_classes) * 0.6)
    # plot_confidence_histograms(correct_confidences, incorrect_confidences, threshold=threshold)

    print("[Entropy Thresholds]")
    print(f"  {threshold_percentile}% percentile (correct only): {threshold:.4f}")
    print(f"  Balanced threshold: {balanced_threshold:.4f}")
    print(f"  Ceiling threshold: {entropy_ceiling}")

    if args.save_model:
        game_stats = {
            "score_mean_std": score_stats,
            "lives_mean_std": lives_stats,
            "max_digits": max_digits,
        }

        model_config = {
            "name": "crnn_ctc",
            "image_size": IMAGE_SIZE,
            "max_decode_length": max_digits,
            "num_classes": num_classes,
            "hidden_size": hidden_size,
            "cnn_dropout": cnn_drop_out_rate,
            "rnn_dropout": rnn_drop_out_rate,
            "blank_idx": blank_index,
            "padding_idx": padding_index,
            "life_symbol": LIFE_DIGIT,
            "entropy_threshold": threshold,
            "balanced_entropy_threshold": balanced_threshold,
            "entropy_ceiling": entropy_ceiling,
        }

        train_config = {
            "seed": args.seed,
            "class_balancing": use_class_balancing,
            "weight_decay": weight_decay,
            "patience": patience,
            "early_stop_patience": early_stop_patience,
            "factor": factor,
            "min_lr": min_lr,
            "temperature": temperature,
            "lr": lr,
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "score_aug_scale": score_aug_scale,
            "lives_aug_scale": lives_aug_scale,
        }

        train_history = {
            "epochs": epoch,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracy": accuracy,
            "cer": cer,
            "char_acc": char_acc,
            "lrs": lrs,
            "mean_confidences": mean_confidences,
        }

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "game_config": game_stats,
            "train_config": train_config,
            "train_summary": train_history,
            "final_cer": cer[-1],
            "final_acc": accuracy[-1],
            "num_train_samples": len(train_loader.dataset),
            "num_test_samples": len(test_loader.dataset),
        }

        def archive_existing_model(model_name):
            if not os.path.exists(model_name):
                return

            rev = 1
            while True:
                archived_name = f"{model_name}{rev}"
                if not os.path.exists(archived_name):
                    os.rename(model_name, archived_name)
                    print(f"archived previous model as: {archived_name}")
                    break
                rev += 1

        archive_existing_model(model_name)
        print(f"Writing model to {model_name}")
        torch.save(checkpoint, model_name)

    print("Complete")


if __name__ == '__main__':
    main()
