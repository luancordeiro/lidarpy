# LIDARpy: A Python Library for LIDAR Data Analysis

LIDARpy is a comprehensive Python library tailored for the analysis, manipulation, and interpretation of LIDAR data. This library provides a set of tools for background noise removal, data grouping, bin adjustments, uncertainty computations, and advanced data inversion using both the Klett and Raman methods.

DOI: 10.5281/zenodo.15644175

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

## License:

This project is licensed under the MIT License.
