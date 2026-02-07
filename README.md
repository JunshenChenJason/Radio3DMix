# Radio3DMix
## 🏙️ The Radio3DMix Dataset

**Radio3DMix** is a large-scale, high-fidelity 3D radio map dataset designed for complex urban environments. Unlike existing datasets restricted to 2D slices or pure ray-tracing, Radio3DMix explicitly models spatial correlations across height slices and incorporates antenna polarization characteristics.

### Key Features
* **Scale**: 50,000 3D Radio Maps.
* **Environment**: Based on 200 distinct real-world urban regions (Beijing).
* **Resolution**: High-precision $1 \text{ m} \times 1 \text{ m} \times 1 \text{ m}$ voxel grid.
* **Dimensions**: $256 \times 256 \times 20 \text{ m}^3$ per map.
* **Propagation Physics**: Captures 3D channel fading and transmitter antenna polarization effects.

### Dataset Parameters

| Parameter | Value |
| :--- | :--- |
| **Radio Map Size** | $256 \times 256 \times 20 \text{ m}^3$ |
| **Resolution** | $1 \text{ m}$ |
| **Total Samples** | 50,000 |
| **Building Maps** | 200 |
| **Transmitters per Sample** | 2 |
