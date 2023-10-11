# LIDARpy: A Python Library for LIDAR Data Analysis

LIDARpy is a comprehensive Python library tailored for the analysis, manipulation, and interpretation of LIDAR data. This library provides a set of tools for background noise removal, data grouping, bin adjustments, uncertainty computations, and advanced data inversion using both the Klett and Raman methods.

## Installation:

```python
pip install lidarpy
```

## Features:

- **Cloud Identification**: 
  - The `CloudFinder` class has been designed to scrutinize LIDAR signals and pinpoint cloud layers based on set conditions and statistical measures.
  
- **Klett Inversion Application**:
  - Employ the `Klett` class for the execution of the Klett inversion algorithm specific to LIDAR inversion.
  
- **Raman Inversion Technique**:
  - The `Raman` class assists in applying the Raman inversion algorithm, extracting information on aerosol extinction and backscatter profiles from LIDAR inversions.
  
- **Multi-Scattering Corrections**:
  - Harness the power of the `multiscatter` function to perform comprehensive multiple scattering calculations for radar or lidar, inspired by Hogan's 2008 model on fast lidar and radar multiple-scattering.

- **Cloud Optical Depth Calculation**:
  - Utilize the `GetCod` class to compute Cloud Optical Depth (COD) via methods elaborated by Young in 1995. The class capitalizes on molecular scattering principles and radiative transfer theory to present both standard fitting and Monte Carlo techniques.

- **Lidar Ratio Computation**:
  - The upcoming `LidarRatioCalculator` class is anticipated to offer essential tools and algorithms for calculating the lidar ratio, crucial for many LIDAR applications.

## Usage:

For hands-on examples and better understanding:

- **Klett Inversion**: 
  - A practical example of the Klett inversion can be accessed [here](https://colab.research.google.com/drive/1adUcYvsfHEO-ncbU-AaVtlIaqZSRMHv4?usp=sharing).
  
- **Raman Inversion**:
  - For a detailed example of the Raman inversion, click [here](https://colab.research.google.com/drive/1JdSv8H25krw-dEjKL9COnPCeiDDV4mIp?usp=sharing).
 
- **Transmittance Method**:
  - For a detailed example of the tansmittance method, click [here](https://colab.research.google.com/drive/14ERNR1mqINw04KMRrZXKHyl8zDfg34eg?usp=sharing).

- **Cloud Detection Tool**
  - For a detailed example of the cloud detection, click [here](https://colab.research.google.com/drive/1dt13jyj_OV7bhYIAJYf8KtD79Ucn2p_M?usp=sharing).
 
- **Real Inversion** 
  - For a detailed example of a inversion, click [here](https://colab.research.google.com/drive/1BT6kEFeMs11ESPUFd3vxR38pphnRUAvn?usp=sharing).

## License:

This project is licensed under the MIT License.
