"""Generate CO3D evaluation indices compatible with ``Co3dEvalDataset``.

Each selected sequence records context and target view indices chosen with the
same spacing heuristics used during training (RealEstate10K-style). The output
JSON can then be supplied to ``Co3dEvalDataset`` to control which frames are
loaded during evaluation.

Example usage:
python nvs/gen_index_file_co3d.py

python nvs/gen_index_file_co3d.py \
--output /home/yuwu3/prope/assets/co3d_test_context4_full.json \
--mode random --split test --categories full --context-views 4 --target-views 4
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Sequence

from co3d_dataset import (
	CO3D_ANNOTATION_DIR,
	TRAINING_CATEGORIES,
	TEST_CATEGORIES,
	_Co3dSequenceStore,
)


def resolve_categories(raw: Sequence[str]) -> List[str]:
	resolved = set()
	for item in raw:
		if item == "seen":
			resolved.update(TRAINING_CATEGORIES)
		elif item == "unseen":
			resolved.update(TEST_CATEGORIES)
		elif item == "full":
			resolved.update(TRAINING_CATEGORIES)
			resolved.update(TEST_CATEGORIES)
		else:
			resolved.add(item)
	return sorted(resolved)


def select_views(
	n_frames: int,
	context_views: int,
	target_views: int,
	min_frame_dist: int,
	max_frame_dist: int,
	mode: str = "range",
) -> List[int] | None:
	total_needed = context_views + target_views
	if n_frames < total_needed:
		return None

	if mode == "random":
		# Randomly sample all needed frames without temporal constraints
		return random.sample(range(n_frames), total_needed)

	# Range-based sampling with temporal constraints
	max_dist = min(n_frames - 1, max_frame_dist)
	min_dist = min(max(1, min_frame_dist), max_dist)

	if max_dist <= 1:
		return random.sample(range(n_frames), total_needed)

	frame_dist = random.randint(min_dist, max_dist)
	if n_frames <= frame_dist:
		return None

	start_index = random.randint(0, n_frames - frame_dist - 1)
	end_index = start_index + frame_dist

	supervise_pool = list(range(start_index + 1, end_index))
	if len(supervise_pool) < target_views:
		supervise_pool = [idx for idx in range(n_frames) if idx not in (start_index, end_index)]
		if len(supervise_pool) < target_views:
			return None

	supervise_indices = random.sample(supervise_pool, target_views)
	return [start_index, end_index] + supervise_indices


def build_index(
	categories: Sequence[str],
	split: str,
	annotation_dir: str,
	context_views: int,
	target_views: int,
	min_frame_dist: int,
	max_frame_dist: int,
	attempts: int,
	mode: str = "range",
) -> Dict[str, Dict[str, List[int]]]:
	store = _Co3dSequenceStore(categories, split, annotation_dir, min_frames=context_views + target_views)

	index: Dict[str, Dict[str, List[int]]] = {}
	skipped = 0

	for sequence in store.sequence_list:
		frames = store.rotations[sequence]
		selected: List[int] | None = None
		for _ in range(attempts):
			selected = select_views(
				len(frames),
				context_views=context_views,
				target_views=target_views,
				min_frame_dist=min_frame_dist,
				max_frame_dist=max_frame_dist,
				mode=mode,
			)
			if selected is not None:
				break
		if selected is None:
			skipped += 1
			continue

		context_indices = selected[:context_views]
		target_indices = selected[context_views : context_views + target_views]
		index[sequence] = {
			"context_view_indices": context_indices,
			"target_view_indices": target_indices,
		}

	total = len(store.sequence_list)
	print(f"Prepared {len(index)} sequences (skipped {skipped} / {total}).")
	return index


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate CO3D evaluation index JSON.")
	parser.add_argument(
		"--output", 
		default="/home/yuwu3/prope/assets/co3d_test_context2_full_10-30.json",
		help="Path to the output JSON file.")
	parser.add_argument(
		"--split",
		default="test",
		choices=["train", "val", "test"],
		help="Dataset split to index (default: test).",
	)
	parser.add_argument(
		"--categories",
		nargs="*",
		default=["full"],
		help="Category list or shortcuts: seen, unseen, full.",
	)
	parser.add_argument("--context-views", type=int, default=2, help="Number of context views to select (default: 2).")
	parser.add_argument("--target-views", type=int, default=3, help="Number of target views to select (default: 3).")
	parser.add_argument("--min-frame-dist", type=int, default=10, help="Minimum frame distance when sampling (default: 25).")
	parser.add_argument("--max-frame-dist", type=int, default=30, help="Maximum frame distance when sampling (default: 100).")
	parser.add_argument("--attempts", type=int, default=20, help="Retry count per sequence before giving up (default: 20).")
	parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible sampling (default: 0).")
	parser.add_argument("--mode", type=str, default="range", choices=["range", "random"], help="View selection mode: 'range' uses temporal constraints, 'random' samples uniformly (default: range).")
	parser.add_argument("--annotation-dir", default=CO3D_ANNOTATION_DIR, help="Override CO3D annotation directory.")

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.annotation_dir is None:
		raise RuntimeError("Annotation directory must be provided via --annotation-dir.")

	categories = resolve_categories(args.categories)
	if not categories:
		raise ValueError("No categories resolved from the provided input.")

	random.seed(args.seed)

	index = build_index(
		categories=categories,
		split=args.split,
		annotation_dir=args.annotation_dir,
		context_views=args.context_views,
		target_views=args.target_views,
		min_frame_dist=args.min_frame_dist,
		max_frame_dist=args.max_frame_dist,
		attempts=args.attempts,
		mode=args.mode,
	)

	with open(args.output, "w", encoding="utf-8") as handle:
		json.dump(index, handle, indent=2)

	print(f"Wrote index with {len(index)} sequences to {args.output}.")


if __name__ == "__main__":
	main()
