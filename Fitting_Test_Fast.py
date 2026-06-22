import argparse
import csv
import os
import re
import time
from collections import OrderedDict

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image


def get_building_maps(building_path, order, height, shape=(256, 256)):
    img_filename = f"{order}_{height}.png"
    img_path = os.path.join(building_path, img_filename)

    if not os.path.exists(img_path):
        return np.zeros(shape, dtype=np.uint8)

    img = Image.open(img_path)
    img_array = np.array(img)

    if img_array.ndim == 3:
        building_mask = (
            (img_array[:, :, 0] == 255)
            & (img_array[:, :, 1] == 255)
            & (img_array[:, :, 2] == 255)
        )
    else:
        building_mask = img_array == 255

    return building_mask.astype(np.float64)


def find_valid_tx_locations_all_maps(building_maps_all, min_height, max_height, num_tx, num_realizations):
    """
    This is intentionally equivalent to Fitting_Test.py.
    The coordinate order and np.random.choice calls must not change.
    """
    num_maps, num_heights, _, _ = building_maps_all.shape
    tx_locs_all = []
    for map_idx in range(num_maps):
        valid_coords_3d = []
        for h_idx in range(min_height - 1, max_height):
            if h_idx < num_heights:
                building_map_at_h = building_maps_all[map_idx, h_idx]
                coords_at_h = np.argwhere(building_map_at_h == 0)
                for x, y in coords_at_h:
                    valid_coords_3d.append((int(x), int(y), int(h_idx + 1)))

        if len(valid_coords_3d) < num_tx:
            raise ValueError(f"Map {map_idx} 中可用位置不足")

        map_tx_locs = []
        for _ in range(num_realizations):
            chosen_idxs = np.random.choice(len(valid_coords_3d), num_tx, replace=False)
            chosen_coords = [valid_coords_3d[i] for i in chosen_idxs]
            map_tx_locs.append(chosen_coords)

        tx_locs_all.append(map_tx_locs)
    return np.array(tx_locs_all)


def save_tx_locations_to_csv(tx_locs_all, minimum_orders, save_path):
    num_maps = len(tx_locs_all)
    num_realizations = len(tx_locs_all[0])

    with open(save_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [f"Map_{i + minimum_orders}" for i in range(num_maps)]
        writer.writerow(["Realization"] + header)

        for r in range(num_realizations):
            row = [f"R{r + 1}"]
            for m in range(num_maps):
                tx_coords = tx_locs_all[m][r]
                coord_str = ",".join([f"[{x},{y},{h}]" for x, y, h in tx_coords])
                row.append(coord_str)
            writer.writerow(row)


def load_tx_locations_from_csv(save_path):
    with open(save_path, mode="r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        map_cols = header[1:]
        maps = len(map_cols)
        rows = list(reader)
        num_realizations = len(rows)
        pattern = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+)\]")

        first_matches = pattern.findall(rows[0][1]) if rows else []
        num_tx = len(first_matches)

        tx_locs_arr = np.zeros((maps, num_realizations, num_tx, 3), dtype=int)
        for r_idx, row in enumerate(rows):
            for m_idx in range(maps):
                matches = pattern.findall(row[m_idx + 1])
                for t_idx, (x, y, h) in enumerate(matches):
                    tx_locs_arr[m_idx, r_idx, t_idx] = [int(x), int(y), int(h)]
        return tx_locs_arr


def shift_and_lock_noise_floor(
    new_output_all,
    building_maps_all,
    map_idx,
    heights,
    xySize,
    p3,
    p2,
    minimum_orders,
    realization_idx,
    noise_floor_dbm=-120.0,
    target_min_dbm=-100.0,
    target_max_dbm=30.0,
    shift_eps=1e-6,
    outlier_min_count=50,
    quiet=False,
):
    mask_free = building_maps_all[map_idx, : len(heights), :xySize, :xySize] == 0

    vals = new_output_all[mask_free]
    finite_vals = vals[np.isfinite(vals)]

    if finite_vals.size > 0:
        outlier_mask = (finite_vals < target_min_dbm) | (finite_vals > target_max_dbm)
        outlier_count = int(np.count_nonzero(outlier_mask))

        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))

        out = new_output_all.copy()
        out[~mask_free] = noise_floor_dbm

        if outlier_count < outlier_min_count:
            if not quiet:
                print(
                    f"[Shift] map {map_idx + minimum_orders}, realization {realization_idx + 1}, "
                    f"3d factor {p3:.1f}, 2d factor {p2:.1f}: "
                    f"skip shift (outliers {outlier_count} < {outlier_min_count}), "
                    f"free-space before=[{vmin:.3f},{vmax:.3f}] dBm"
                )
            return out

        c_low = (target_min_dbm + shift_eps) - vmin
        c_high = (target_max_dbm - shift_eps) - vmax

        if c_low <= c_high:
            c = float(np.clip(0.0, c_low, c_high))
        else:
            cur_center = 0.5 * (vmin + vmax)
            tar_center = 0.5 * (target_min_dbm + target_max_dbm)
            c = tar_center - cur_center

        out[mask_free] += c

        vals2 = out[mask_free]
        finite_vals2 = vals2[np.isfinite(vals2)]
        vmin2 = float(np.min(finite_vals2)) if finite_vals2.size > 0 else float("nan")
        vmax2 = float(np.max(finite_vals2)) if finite_vals2.size > 0 else float("nan")

        if not quiet:
            print(
                f"[Shift] map {map_idx + minimum_orders}, realization {realization_idx + 1}, "
                f"3d factor {p3:.1f}, 2d factor {p2:.1f}: "
                f"c={c:.6f}, before=[{vmin:.3f},{vmax:.3f}] dBm, after=[{vmin2:.3f},{vmax2:.3f}] dBm"
            )
        return out

    out = new_output_all.copy()
    out[~mask_free] = noise_floor_dbm
    vmin = float(np.nanmin(out))
    vmax = float(np.nanmax(out))
    if not quiet:
        print(
            f"[Shift] map {map_idx + minimum_orders}, realization {realization_idx + 1}, "
            f"3d factor {p3:.1f}, 2d factor {p2:.1f}: no finite free-space values, "
            f"after=[{vmin:.3f},{vmax:.3f}] dBm (buildings locked to {noise_floor_dbm} dBm)"
        )
    return out


class LruRawOutputCache:
    def __init__(self, max_items):
        self.max_items = int(max_items)
        self._items = OrderedDict()

    def get(self, key):
        if self.max_items <= 0:
            return None
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key, value):
        if self.max_items <= 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


def predict_one_raw_output(model, building_map, tx_x, tx_y, xy_size):
    input_image = np.zeros((1, 2, xy_size, xy_size), dtype=np.float64)
    input_image[0, 0] = building_map
    input_image[0, 1, int(tx_x), int(tx_y)] = 1.0
    _, out = model.predict(input_image)
    return out.squeeze().astype(np.uint8, copy=False)


def predict_raw_outputs_for_realization(model, building_maps_for_map, tx_locs, xy_size, raw_cache):
    tx_count = tx_locs.shape[0]
    num_heights = building_maps_for_map.shape[0]
    raw_outputs = np.empty((tx_count, num_heights, xy_size, xy_size), dtype=np.uint8)

    for tx_idx, tx in enumerate(tx_locs):
        tx_x = int(tx[0])
        tx_y = int(tx[1])
        for height_idx in range(num_heights):
            key = (height_idx, tx_x, tx_y)
            cached = raw_cache.get(key)
            if cached is None:
                cached = predict_one_raw_output(
                    model=model,
                    building_map=building_maps_for_map[height_idx],
                    tx_x=tx_x,
                    tx_y=tx_y,
                    xy_size=xy_size,
                ).copy()
                raw_cache.put(key, cached)
            raw_outputs[tx_idx, height_idx] = cached

    return raw_outputs


def predict_raw_outputs_original_call_order(model, building_maps_for_map, tx_locs, xy_size):
    tx_count = tx_locs.shape[0]
    num_heights = building_maps_for_map.shape[0]
    raw_outputs = np.empty((tx_count, num_heights, xy_size, xy_size), dtype=np.uint8)

    for height_idx in range(num_heights):
        for tx_idx, tx in enumerate(tx_locs):
            raw_outputs[tx_idx, height_idx] = predict_one_raw_output(
                model=model,
                building_map=building_maps_for_map[height_idx],
                tx_x=int(tx[0]),
                tx_y=int(tx[1]),
                xy_size=xy_size,
            )

    return raw_outputs


def precompute_distance_logs(tx_locs, heights, xy_size):
    tx_count = tx_locs.shape[0]
    num_heights = len(heights)
    grid_x = np.arange(xy_size, dtype=np.float64)[:, None]
    grid_y = np.arange(xy_size, dtype=np.float64)[None, :]

    log2 = np.empty((tx_count, xy_size, xy_size), dtype=np.float64)
    log3 = np.empty((tx_count, num_heights, xy_size, xy_size), dtype=np.float64)

    for tx_idx, tx in enumerate(tx_locs):
        dx = grid_x - float(tx[0])
        dy = grid_y - float(tx[1])
        d2_sq = dx * dx + dy * dy

        d2 = np.sqrt(d2_sq)
        this_log2 = np.zeros_like(d2, dtype=np.float64)
        mask2 = d2 != 0
        this_log2[mask2] = np.log10(d2[mask2])
        log2[tx_idx] = this_log2

        for height_idx, height in enumerate(heights):
            dz = float(tx[2] - height)
            d3 = np.sqrt(d2_sq + dz * dz)
            this_log3 = np.zeros_like(d3, dtype=np.float64)
            mask3 = d3 != 0
            this_log3[mask3] = np.log10(d3[mask3])
            log3[tx_idx, height_idx] = this_log3

    return log3, log2


def synthesize_radio_map_vectorized(raw_terms, log3, log2, coef3, coef2):
    tx_count, num_heights, xy_size, _ = raw_terms.shape
    output = np.empty((num_heights, xy_size, xy_size), dtype=np.float64)

    for height_idx in range(num_heights):
        total_power_mw = np.zeros((xy_size, xy_size), dtype=np.float64)
        for tx_idx in range(tx_count):
            power = raw_terms[tx_idx, height_idx] + coef3 * log3[tx_idx, height_idx]
            power = power + coef2 * log2[tx_idx]
            total_power_mw += 10.0 ** (power / 10.0)
        output[height_idx] = 10.0 * np.log10(total_power_mw)

    return output


def calculate_point_power_reference(tx, raw_output, height_idx, fitting_parameters_good, heights, xySize, total_power_mW):
    for i in range(xySize):
        for j in range(xySize):
            q = np.array([i, j, heights[height_idx]])
            diff = tx - q
            dis_3d = np.linalg.norm(diff)
            dis_2d = np.linalg.norm(diff[0:2])

            power = fitting_parameters_good[0] + fitting_parameters_good[1] * raw_output[i, j]
            if dis_3d != 0:
                power += fitting_parameters_good[2] * np.log10(dis_3d)
            if dis_2d != 0:
                power += fitting_parameters_good[3] * np.log10(dis_2d)

            point_power_mw = 10 ** (power / 10)
            total_power_mW[i, j] += point_power_mw

    return total_power_mW


def synthesize_radio_map_reference(raw_outputs, tx_locs, fitting_parameters_good, heights, xy_size):
    tx_count = tx_locs.shape[0]
    num_heights = len(heights)
    output = np.empty((num_heights, xy_size, xy_size), dtype=np.float64)

    for height_idx in range(num_heights):
        total_power_mw = np.zeros((xy_size, xy_size), dtype=np.float64)
        for tx_idx in range(tx_count):
            total_power_mw = calculate_point_power_reference(
                tx=tx_locs[tx_idx],
                raw_output=raw_outputs[tx_idx, height_idx],
                height_idx=height_idx,
                fitting_parameters_good=fitting_parameters_good,
                heights=heights,
                xySize=xy_size,
                total_power_mW=total_power_mw,
            )
        output[height_idx] = 10.0 * np.log10(total_power_mw)

    return output


def verify_vectorized_math():
    rng = np.random.default_rng(123)
    xy_size = 32
    heights = list(range(1, 6))
    tx_locs = np.array([[4, 7, 3], [17, 9, 5], [31, 31, 1]], dtype=int)
    raw_outputs = rng.integers(0, 256, size=(tx_locs.shape[0], len(heights), xy_size, xy_size), dtype=np.uint8)
    fitting_parameters_good = [
        -116.39514779824822,
        0.5609487210289004,
        -36.93654925277295 * 2.3,
        52.181020798935606 * 2.1,
    ]

    ref = synthesize_radio_map_reference(raw_outputs, tx_locs, fitting_parameters_good, heights, xy_size)
    log3, log2 = precompute_distance_logs(tx_locs, heights, xy_size)
    raw_terms = fitting_parameters_good[0] + fitting_parameters_good[1] * raw_outputs.astype(np.float64)
    vec = synthesize_radio_map_vectorized(
        raw_terms=raw_terms,
        log3=log3,
        log2=log2,
        coef3=fitting_parameters_good[2],
        coef2=fitting_parameters_good[3],
    )

    abs_diff = np.abs(ref - vec)
    same_float32 = np.array_equal(ref.astype(np.float32), vec.astype(np.float32))
    print(f"float64 max abs diff: {abs_diff.max():.12g}")
    print(f"stored float32 exactly equal: {same_float32}")
    if not same_float32:
        mismatch = np.argwhere(ref.astype(np.float32) != vec.astype(np.float32))[0]
        idx = tuple(int(x) for x in mismatch)
        print(f"first float32 mismatch at {idx}: ref={ref[idx]}, vec={vec[idx]}")
        raise SystemExit(1)


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fast, output-compatible generator for Fitting_Test.py 3D radio maps."
    )
    parser.add_argument("--verify-vectorized", action="store_true", help="run a small math equivalence check and exit")
    parser.add_argument("--num-tx-list", default="2")
    parser.add_argument("--maximum-heights", type=int, default=20)
    parser.add_argument("--minimum-orders", type=int, default=1)
    parser.add_argument("--maximum-orders", type=int, default=200)
    parser.add_argument("--min-height", type=int, default=1)
    parser.add_argument("--max-height", type=int, default=20)
    parser.add_argument("--num-realizations", type=int, default=10)
    parser.add_argument("--xy-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--p2-list", default="2.0,2.1,2.2,2.3,2.4")
    parser.add_argument("--delta-list", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--building-path", default="data/Beijing_Dataset/BJ_256_png")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--raw-cache-mb", type=int, default=512)
    parser.add_argument("--no-cache-raw", dest="cache_raw", action="store_false")
    parser.add_argument("--no-deterministic-model", dest="deterministic_model", action="store_false")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--quiet-shift", action="store_true")
    parser.set_defaults(cache_raw=True, deterministic_model=True)
    return parser


def output_index(combo_idx, map_idx, realization_idx, num_maps, num_realizations):
    return (combo_idx * num_maps + map_idx) * num_realizations + realization_idx


def configure_torch_runtime(args):
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.deterministic_model:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def main():
    args = build_arg_parser().parse_args()

    if args.verify_vectorized:
        verify_vectorized_math()
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    configure_torch_runtime(args)
    from RadioUNet_Model import RadioWNetModel

    np.random.seed(args.seed)

    num_tx_list = parse_int_list(args.num_tx_list)
    p2_list = parse_float_list(args.p2_list)
    delta_list = parse_float_list(args.delta_list)
    heights = list(range(1, args.maximum_heights + 1))
    orders = list(range(args.minimum_orders, args.maximum_orders + 1))
    num_maps = len(orders)
    num_combos = len(p2_list) * len(delta_list)

    base_fitting = [
        -116.39514779824822,
        0.5609487210289004,
        -36.93654925277295,
        52.181020798935606,
    ]
    noise_floor_dbm = -120.0

    building_path = args.building_path
    results_root = args.results_root or os.path.join(script_dir, "results")
    os.makedirs(results_root, exist_ok=True)

    print("Loading building maps...")
    building_maps_all = []
    for predicting_order in orders:
        building_maps = []
        for predicting_height in heights:
            building_maps_raw = get_building_maps(
                building_path,
                order=predicting_order,
                height=predicting_height,
                shape=(args.xy_size, args.xy_size),
            )
            building_maps.append(building_maps_raw)
        building_maps_all.append(building_maps)
    building_maps_all = np.array(building_maps_all)

    model = RadioWNetModel()

    if args.cache_raw:
        print("Generation mode: cached raw model outputs + vectorized power synthesis")
    else:
        print("Generation mode: original model-call order + vectorized power synthesis")
    if args.cache_raw and not args.deterministic_model and not args.force_cpu:
        print(
            "[Warn] --no-deterministic-model with cached raw outputs can differ from Fitting_Test.py on CUDA, "
            "because repeated single-sample inference is not bitwise stable."
        )

    combos = []
    for p2 in p2_list:
        for delta in delta_list:
            p3 = p2 + delta
            combos.append((p2, p3, base_fitting[2] * p3, base_fitting[3] * p2))

    raw_item_bytes = args.xy_size * args.xy_size
    raw_cache_items = max(0, (args.raw_cache_mb * 1024 * 1024) // raw_item_bytes)

    for num_tx in num_tx_list:
        tx_csv_path = os.path.join(
            results_root,
            f"tx_locations_c{num_combos}"
            f"_m{num_maps}"
            f"_r{args.num_realizations}"
            f"_tx{num_tx}"
            f"_h{args.maximum_heights}.csv",
        )
        signal_map_npy_path = os.path.join(
            results_root,
            f"signal_map_c{num_combos}"
            f"_m{num_maps}"
            f"_r{args.num_realizations}"
            f"_tx{num_tx}"
            f"_h{args.maximum_heights}.npy",
        )
        n_samples = num_combos * num_maps * args.num_realizations

        print(f"\nPreparing Tx={num_tx}: {n_samples} samples -> {signal_map_npy_path}")
        all_mm = open_memmap(
            signal_map_npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, 1, args.maximum_heights, args.xy_size, args.xy_size),
        )

        if os.path.exists(tx_csv_path):
            tx_locs_all = load_tx_locations_from_csv(tx_csv_path)
            if tx_locs_all.shape[2] != num_tx:
                print(f"[Warn] {tx_csv_path} has {tx_locs_all.shape[2]} Tx, expected {num_tx}. Regenerating...")
                tx_locs_all = find_valid_tx_locations_all_maps(
                    building_maps_all,
                    args.min_height,
                    args.max_height,
                    num_tx,
                    args.num_realizations,
                )
                save_tx_locations_to_csv(tx_locs_all, args.minimum_orders, tx_csv_path)
        else:
            tx_locs_all = find_valid_tx_locations_all_maps(
                building_maps_all,
                args.min_height,
                args.max_height,
                num_tx,
                args.num_realizations,
            )
            save_tx_locations_to_csv(tx_locs_all, args.minimum_orders, tx_csv_path)

        started = time.perf_counter()
        if args.cache_raw:
            for map_idx in range(num_maps):
                map_started = time.perf_counter()
                raw_cache = LruRawOutputCache(raw_cache_items)
                building_maps_for_map = building_maps_all[map_idx]

                for realization_idx in range(args.num_realizations):
                    tx_locs = tx_locs_all[map_idx, realization_idx]
                    raw_outputs = predict_raw_outputs_for_realization(
                        model=model,
                        building_maps_for_map=building_maps_for_map,
                        tx_locs=tx_locs,
                        xy_size=args.xy_size,
                        raw_cache=raw_cache,
                    )
                    raw_terms = base_fitting[0] + base_fitting[1] * raw_outputs.astype(np.float64)
                    log3, log2 = precompute_distance_logs(tx_locs, heights, args.xy_size)

                    for combo_idx, (p2, p3, coef3, coef2) in enumerate(combos):
                        new_output_all = synthesize_radio_map_vectorized(
                            raw_terms=raw_terms,
                            log3=log3,
                            log2=log2,
                            coef3=coef3,
                            coef2=coef2,
                        )
                        new_output_all = shift_and_lock_noise_floor(
                            new_output_all=new_output_all,
                            building_maps_all=building_maps_all,
                            map_idx=map_idx,
                            heights=heights,
                            xySize=args.xy_size,
                            p3=p3,
                            p2=p2,
                            minimum_orders=args.minimum_orders,
                            realization_idx=realization_idx,
                            noise_floor_dbm=noise_floor_dbm,
                            target_min_dbm=-150.0,
                            target_max_dbm=50.0,
                            quiet=args.quiet_shift,
                        )
                        n_idx = output_index(combo_idx, map_idx, realization_idx, num_maps, args.num_realizations)
                        all_mm[n_idx, 0] = new_output_all

                elapsed = time.perf_counter() - map_started
                total_elapsed = time.perf_counter() - started
                print(
                    f"[Progress] Tx={num_tx}, map {map_idx + 1}/{num_maps} finished "
                    f"in {elapsed:.1f}s, total {total_elapsed / 60:.1f} min"
                )
        else:
            completed = 0
            for combo_idx, (p2, p3, coef3, coef2) in enumerate(combos):
                combo_started = time.perf_counter()
                for map_idx in range(num_maps):
                    building_maps_for_map = building_maps_all[map_idx]
                    for realization_idx in range(args.num_realizations):
                        tx_locs = tx_locs_all[map_idx, realization_idx]
                        raw_outputs = predict_raw_outputs_original_call_order(
                            model=model,
                            building_maps_for_map=building_maps_for_map,
                            tx_locs=tx_locs,
                            xy_size=args.xy_size,
                        )
                        raw_terms = base_fitting[0] + base_fitting[1] * raw_outputs.astype(np.float64)
                        log3, log2 = precompute_distance_logs(tx_locs, heights, args.xy_size)
                        new_output_all = synthesize_radio_map_vectorized(
                            raw_terms=raw_terms,
                            log3=log3,
                            log2=log2,
                            coef3=coef3,
                            coef2=coef2,
                        )
                        new_output_all = shift_and_lock_noise_floor(
                            new_output_all=new_output_all,
                            building_maps_all=building_maps_all,
                            map_idx=map_idx,
                            heights=heights,
                            xySize=args.xy_size,
                            p3=p3,
                            p2=p2,
                            minimum_orders=args.minimum_orders,
                            realization_idx=realization_idx,
                            noise_floor_dbm=noise_floor_dbm,
                            target_min_dbm=-150.0,
                            target_max_dbm=50.0,
                            quiet=args.quiet_shift,
                        )
                        n_idx = output_index(combo_idx, map_idx, realization_idx, num_maps, args.num_realizations)
                        all_mm[n_idx, 0] = new_output_all
                        completed += 1

                combo_elapsed = time.perf_counter() - combo_started
                total_elapsed = time.perf_counter() - started
                print(
                    f"[Progress] Tx={num_tx}, combo {combo_idx + 1}/{num_combos} finished "
                    f"in {combo_elapsed / 60:.1f} min, total {total_elapsed / 60:.1f} min, samples {completed}"
                )

        del all_mm


if __name__ == "__main__":
    main()
