# [Designing Specialized Two-Dimensional Graph Spectral Filters for Spatial-Temporal Graph Modeling]

By: Yuxin Chen, Fangru Lin, Jingyi Huo, Hui yan

## Our paper has been accepted for publication of AAAI 2025 (acceptance rate 23.4%).

## Requirements

Our code is based on Python version 3.9.7 and PyTorch version 1.10.1. Please make sure you have installed Python and PyTorch correctly.

## Data 

Traffic: We utilize four widely studied traffic forecasting datasets: PeMS03, PeMS04, PeMS07, and PeMS08. PeMS stands for the Caltrans Performance Measurement System, which measures highway traffic in California in real-time every 30 seconds. All datasets are aggregated into 5-minute intervals, resulting in 288 data points per day.

Climate: KnowAir documents weather observations every 3 hours, encompassing a dataset that spans from 2015 to 2018 and covers 184 major cities in China. We use three meteorological characteristics from the KnowAir dataset, including U wind components, urban temperature recordings, and PM2.5 concentrations, which we denote as KA-UW, KA-TEMP, and KA-PM$_{2.5}$.
