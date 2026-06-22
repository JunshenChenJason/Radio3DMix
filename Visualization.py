import numpy as np
import os
import re
import csv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def get_building_maps(building_path, order, height, shape=(256, 256)):
    """
    读取指定高度层的建筑地图 PNG 文件并转换为二值数组
    """
    img_filename = f"{order}_{height}.png"
    img_path = os.path.join(building_path, img_filename)

    if not os.path.exists(img_path):
        return np.zeros(shape, dtype=np.uint8)

    img = Image.open(img_path)
    img_array = np.array(img)

    if img_array.ndim == 3:  # RGB 图像
        building_mask = (img_array[:, :, 0] == 255) & \
                        (img_array[:, :, 1] == 255) & \
                        (img_array[:, :, 2] == 255)
    else:  # 灰度图像
        building_mask = (img_array == 255)

    return building_mask.astype(np.float64)


def load_tx_locations_from_csv(save_path):
    """从 CSV 文件加载 Tx 位置"""
    with open(save_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        map_cols = header[1:]
        maps = len(map_cols)
        rows = list(reader)
        num_realizations = len(rows)
        pattern = re.compile(r'\[(\d+),\s*(\d+),\s*(\d+)\]')

        first_matches = pattern.findall(rows[0][1]) if rows else []
        num_tx = len(first_matches)

        Tx_locs_arr = np.zeros((maps, num_realizations, num_tx, 3), dtype=int)
        for r_idx, row in enumerate(rows):
            for m_idx in range(maps):
                matches = pattern.findall(row[m_idx + 1])
                for t_idx, (x, y, h) in enumerate(matches):
                    Tx_locs_arr[m_idx, r_idx, t_idx] = [int(x), int(y), int(h)]
        return Tx_locs_arr


def save_image_with_mask(img, mask, save_path, output="gray", mode="per_image", other=None, percentile_clip=None,
                         global_min=None, global_max=None, cmap_name="viridis"):
    """
    保存带掩膜的图像,支持灰度和彩色输出,以及多种归一化模式
    args:
        img: np.ndarray or PIL.Image, 输入图像数据
        mask: np.ndarray or PIL.Image, 掩膜数据（非零值表示掩盖区域）
        save_path: str, 保存路径
        output: str, "gray" 或 "color"
        mode: str, "per_image", "pair", 或 "global"
        other: np.ndarray or PIL.Image, 仅在 mode="pair" 时使用的另一张图像
        percentile_clip: tuple(float, float), 可选的百分位裁剪范围 (lo_p, hi_p)
        global_min: float, 仅在 mode="global" 时使用的全局最小值
        global_max: float, 仅在 mode="global" 时使用的全局最大值
        cmap_name: str, 颜色映射名称，仅在 output="color" 时使用
    """
    img = np.asarray(img, dtype=np.float32)

    if mode not in ("per_image", "pair", "global"):
        raise ValueError("mode 必须是 'per_image', 'pair' 或 'global'")
    if output not in ("gray", "color"):
        raise ValueError("output 必须是 'gray' 或 'color'")

    # 计算 vmin/vmax
    if mode == "global":
        if global_min is None or global_max is None:
            raise ValueError("mode='global' 时必须提供 global_min 和 global_max")
        vmin = float(global_min)
        vmax = float(global_max)
    elif mode == "pair":
        if other is None:
            raise ValueError("mode='pair' 时必须传入 other 参数")
        other = np.asarray(other, dtype=np.float32)
        vmin = float(min(np.nanmin(img), np.nanmin(other)))
        vmax = float(max(np.nanmax(img), np.nanmax(other)))
    else:  # per_image
        vmin = float(np.nanmin(img))
        vmax = float(np.nanmax(img))

    # 可选百分位裁剪(非 global)
    if percentile_clip is not None and mode != "global":
        lo_p, hi_p = percentile_clip
        lo = np.percentile(img, lo_p)
        hi = np.percentile(img, hi_p)
        if mode == "pair":
            other = np.asarray(other, dtype=np.float32)
            lo = min(lo, np.percentile(other, lo_p))
            hi = max(hi, np.percentile(other, hi_p))
        if hi > lo:
            vmin, vmax = float(lo), float(hi)

    # 归一化
    if vmax - vmin < 1e-12:
        norm = np.zeros_like(img, dtype=np.float32)
    else:
        norm = (img - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    if output == "gray":
        arr8 = (norm * 255.0).round().astype(np.uint8)
        Image.fromarray(arr8, mode="L").save(save_path)
        print(f"[Saved gray] {save_path} (mode={mode}, vmin={vmin:.3f}, vmax={vmax:.3f})")
        # 覆盖为掩膜后的图
        if mask is not None:
            overlay = arr8.copy()
            overlay[np.asarray(mask) != 0] = 0
            Image.fromarray(overlay, mode="L").save(save_path)
    else:  # output == "color"
        cmap = plt.colormaps[cmap_name]
        rgba = cmap(norm)
        rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
        if mask is not None:
            overlay = rgb.copy()
            overlay[np.asarray(mask) != 0] = 0
            rgb = overlay
        Image.fromarray(rgb, mode="RGB").save(save_path)
        print(f"[Saved color] {save_path} (mode={mode}, cmap={cmap_name}, vmin={vmin:.3f}, vmax={vmax:.3f})")


def plot_and_save_3d_grayscale_layers_optimized(data, save_path, realmax, realmin, building_mask, heights,
                                                elev=45, azim=45, downsample_factor=1, dpi=300, alpha=0.05):
    """
    将多层二维灰度图像堆叠成三维图像并保存为文件(优化性能版)
    args:
        data (np.ndarray): 三维数组，形状为 (layers, height, width)，表示多层灰度图像。
        save_path (str): 保存图像的路径（包括文件名和扩展名，如 'output.png'）。
        elev (float): 三维图像的俯仰角（默认30度）。
        azim (float): 三维图像的方位角（默认45度）。
        downsample_factor (int): 降采样因子，用于减少数据量（默认4，即每4个点取1个）。
        dpi (int): 保存图像的分辨率（默认300）。
        alpha (float): 透明度参数，用于设置图像的透明度（默认0.5）。
    """
    if data.ndim != 3:
        raise ValueError("输入数据必须是三维数组,形状为 (layers, height, width)")

    # 对数据进行归一化处理
    data = (data - realmin) / (realmax - realmin)

    # 对建筑物进行掩码处理
    for index in range(heights):
        data[index, building_mask[index] != 0] = 0

    # 降采样数据
    if downsample_factor > 1:
        data = data[:, ::downsample_factor, ::downsample_factor]

    layers, height, width = data.shape

    # 创建三维图像
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    z_positions = np.arange(layers)
    y, x = np.meshgrid(np.arange(height), np.arange(width))

    cmap = plt.cm.coolwarm  # plt.cm.gray

    norm = Normalize(vmin=realmin, vmax=realmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 一次性绘制所有层
    for z in z_positions:
        layer = data[z]
        rgba_colors = cmap(layer)
        rgba_colors[..., -1] = alpha

        ax.plot_surface(
            x, y,
            np.full_like(x, z),
            rstride=1,
            cstride=1,
            facecolors=rgba_colors,
            shade=False,
        )

    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Intensity')

    ax.set_xlabel('Width')
    ax.set_ylabel('Length')
    ax.set_zlabel('Height')
    ax.view_init(elev=elev, azim=azim)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def visualize_from_npy(signal_map_npy_path, tx_csv_path, building_path,
                       minimum_orders=1, maximum_orders=200, num_realizations=10, maximum_heights=20,
                       xySize=256, p2_list=[2.0, 2.1, 2.2, 2.3, 2.4], delta_list=[0.0, 0.1, 0.2, 0.3, 0.4],
                       output_folder='visualization_output'):
    """
    从已保存的 .npy 和 .csv 文件中读取数据并进行可视化

    参数:
        signal_map_npy_path: 信号地图 .npy 文件路径
        tx_csv_path: 发射机位置 .csv 文件路径
        building_path: 建筑地图文件夹路径
        其他参数: 与原脚本保持一致
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    slices_folder = os.path.join(output_folder, 'slices')
    img_3d_folder = os.path.join(output_folder, '3d_images')
    os.makedirs(slices_folder, exist_ok=True)
    os.makedirs(img_3d_folder, exist_ok=True)

    # 加载数据
    print(f"加载信号地图数据: {signal_map_npy_path}")
    signal_maps = np.load(signal_map_npy_path, mmap_mode='r')
    print(f"数据形状: {signal_maps.shape}")

    print(f"加载发射机位置: {tx_csv_path}")
    Tx_locs_all = load_tx_locations_from_csv(tx_csv_path)
    print(f"发射机位置形状: {Tx_locs_all.shape}")

    # 加载建筑地图
    print("加载建筑地图...")
    building_maps_all = []
    heights = list(range(1, maximum_heights + 1))
    orders = list(range(minimum_orders, maximum_orders + 1))
    for predicting_order in orders:
        building_maps = []
        for predicting_height in heights:
            building_maps_raw = get_building_maps(building_path, order=predicting_order, height=predicting_height)
            building_maps.append(building_maps_raw)
        building_maps_all.append(building_maps)
    building_maps_all = np.array(building_maps_all)

    # 开始可视化
    n_idx = 0
    num_maps = maximum_orders - minimum_orders + 1

    for p2 in p2_list:
        for delta in delta_list:
            p3 = p2 + delta
            print(f"\n处理参数组合: p2={p2:.1f}, p3={p3:.1f}")

            for map_idx in range(num_maps):
                for realization_idx in range(num_realizations):
                    print(f"  可视化地图 {map_idx + minimum_orders}, realization {realization_idx + 1}")

                    # 获取当前数据
                    new_output_all = signal_maps[n_idx, 0]  # 形状: (H, W, D)
                    Tx_loc = Tx_locs_all[map_idx][realization_idx]

                    # 文件名基础
                    base_filename = f"map{map_idx + minimum_orders}_r{realization_idx + 1}_p3({p3:.1f})_p2({p2:.1f})"

                    # 保存多层灰度图
                    save_path_bottom = os.path.join(slices_folder, f"{base_filename}_bottom.png")
                    # save_path_mid = os.path.join(slices_folder, f"{base_filename}_mid.png")
                    # save_path_top = os.path.join(slices_folder, f"{base_filename}_top.png")
                    #
                    bottom_img = new_output_all[0, :xySize, :xySize]
                    # mid_img = new_output_all[9, :xySize, :xySize]
                    # top_img = new_output_all[19, :xySize, :xySize]
                    #
                    bottom_mask = building_maps_all[map_idx, 0, :xySize, :xySize]
                    # mid_mask = building_maps_all[map_idx, 9, :xySize, :xySize]
                    # top_mask = building_maps_all[map_idx, 19, :xySize, :xySize]
                    #
                    save_image_with_mask(bottom_img, bottom_mask, save_path_bottom, output="color", mode="per_image")
                    # save_image_with_mask(mid_img, mid_mask, save_path_mid, output="color", mode="per_image")
                    # save_image_with_mask(top_img, top_mask, save_path_top, output="color", mode="per_image")

                    # 保存发射机位置切片
                    for t_id, (ty, tx, th) in enumerate(Tx_loc, start=1):
                        ty = int(np.clip(ty, 0, xySize - 1))
                        tx = int(np.clip(tx, 0, xySize - 1))

                        slice_xh = new_output_all[:, ty, :][::-1, :]
                        slice_yh = new_output_all[:, :, tx][::-1, :]

                        mask_xh = building_maps_all[map_idx, :, ty, :][::-1, :]
                        mask_yh = building_maps_all[map_idx, :, :, tx][::-1, :]

                        save_y_slice = os.path.join(slices_folder, f"{base_filename}_tx{t_id}_Y{ty}.png")
                        save_x_slice = os.path.join(slices_folder, f"{base_filename}_tx{t_id}_X{tx}.png")
                        save_image_with_mask(slice_yh, mask_yh, save_y_slice, output='color', mode="per_image")
                        save_image_with_mask(slice_xh, mask_xh, save_x_slice, output='color', mode="per_image")

                    # 保存 3D 图像
                    # save_path_img = os.path.join(img_3d_folder, f"{base_filename}.png")
                    # plot_and_save_3d_grayscale_layers_optimized(
                    #     new_output_all[:, :xySize, :xySize],
                    #     save_path_img,
                    #     realmax=np.max(new_output_all[:, :xySize, :xySize]),
                    #     realmin=np.min(
                    #         new_output_all[:, :xySize, :xySize][building_maps_all[map_idx, :, :xySize, :xySize] == 0]),
                    #     building_mask=building_maps_all[map_idx, :, :xySize, :xySize],
                    #     heights=maximum_heights,
                    #     alpha=0.5
                    # )

                    # 建筑物可视化
                    # save_path_building_3d = os.path.join(img_3d_folder, f"{base_filename}_building_3d.png")
                    # building_data = building_maps_all[map_idx, :, :xySize, :xySize].astype(np.float32)
                    # building_data = 1.0 - building_data  # 反转:建筑物变为0,空间变为1
                    # empty_mask = np.zeros_like(building_data)  # 创建一个全零的掩码(不需要掩盖任何部分)
                    # plot_and_save_3d_grayscale_layers_optimized(
                    #     building_data,
                    #     save_path_building_3d,
                    #     realmax=1.0,  # 建筑物二值图的最大值
                    #     realmin=0.0,  # 建筑物二值图的最小值
                    #     building_mask=empty_mask,  # 不需要掩码
                    #     heights=maximum_heights,
                    #     elev=45,
                    #     azim=45,
                    #     downsample_factor=1,
                    #     alpha=0.05  # 建筑物使用较高透明度以便观察
                    # )

                    n_idx += 1

    print("\n可视化完成!")


if __name__ == "__main__":
    # 配置参数(与生成数据时保持一致)
    minimum_orders = 1
    maximum_orders = 5  # 200
    num_realizations = 1  # 2
    Num_Tx = 1
    maximum_heights = 20
    xySize = 256
    p2_list = [1.0]  # [2.0, 2.1, 2.2, 2.3, 2.4]
    delta_list = [0.0]  # [0.0, 0.1, 0.2, 0.3, 0.4]

    # 数据文件路径
    results_root = os.path.join(os.path.dirname(__file__), 'results')
    signal_map_npy_path = os.path.join(results_root, f"signal_map_c{len(p2_list) * len(delta_list)}"
                                                     f"_m{(maximum_orders - minimum_orders + 1)}"
                                                     f"_r{num_realizations}"
                                                     f"_tx{Num_Tx}"
                                                     f"_h{maximum_heights}.npy")
    tx_csv_path = os.path.join(results_root, f"tx_locations_c{len(p2_list) * len(delta_list)}"
                                             f"_m{(maximum_orders - minimum_orders + 1)}"
                                             f"_r{num_realizations}"
                                             f"_tx{Num_Tx}"
                                             f"_h{maximum_heights}.csv")
    building_path = os.path.join(os.path.dirname(__file__), 'data', 'Beijing_Dataset', 'BJ_256_png')
    output_folder = os.path.join(results_root, 'visualization_output')

    # 执行可视化
    visualize_from_npy(
        signal_map_npy_path=signal_map_npy_path,
        tx_csv_path=tx_csv_path,
        building_path=building_path,
        minimum_orders=minimum_orders,
        maximum_orders=maximum_orders,
        num_realizations=num_realizations,
        maximum_heights=maximum_heights,
        xySize=xySize,
        p2_list=p2_list,
        delta_list=delta_list,
        output_folder=output_folder
    )
