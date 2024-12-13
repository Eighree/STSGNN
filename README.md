# [Designing Specialized Two-Dimensional Graph Spectral Filters for Spatial-Temporal Graph Modeling]

By: Yuxin Chen, Fangru Lin, Jingyi Huo, Hui yan

## Our paper has been accepted for publication of AAAI 2025 (acceptance rate 23.4%).

## Requirements

Our code is based on Python version 3.9.7 and PyTorch version 1.10.1. Please make sure you have installed Python and PyTorch correctly.

## Data 

Traffic: We utilize four widely studied traffic forecasting datasets: PeMS03, PeMS04, PeMS07, and PeMS08. PeMS stands for the Caltrans Performance Measurement System, which measures highway traffic in California in real-time every 30 seconds. All datasets are aggregated into 5-minute intervals, resulting in 288 data points per day. Download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with code: `p72z`. Since the spatial graphs for these datasets are constructed based on geographical distances, we maintain a low proportion of node connections by tuning the parameters $\sigma^{2}$ and $\epsilon$ (we set $\sigma^{2}=10$ and $\epsilon=0.1$ in this work). 

Climate: KnowAir documents weather observations every 3 hours, encompassing a dataset that spans from 2015 to 2018 and covers 184 major cities in China. We use three meteorological characteristics from the KnowAir dataset, including U wind components, urban temperature recordings, and PM2.5 concentrations, which we denote as KA-UW, KA-TEMP, and KA-PM2.5. Download dataset **KnowAir** from [Google Drive](https://drive.google.com/open?id=1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L) or [Baiduyun](https://pan.baidu.com/s/18D6Etl5Lm1E4vOLVrX0ZAw) with code `t82d`.  We directly use the adjacency matrix provided in the dataset.


| Datasets   | Node | Interval | Time Range       | Time Steps |
|------------|------|----------|------------------|------------|
| PeMS03     | 358  | 5min     | 9/1/2018-11/30/2018 | 26208     |
| PeMS04     | 307  | 5min     | 1/1/2018-2/28/2018  | 16992     |
| PeMS07     | 883  | 5min     | 5/1/2017-8/31/2017  | 28224     |
| PeMS08     | 170  | 5min     | 7/1/2016-8/31/2016  | 17856     |
| KA-UW      | 184  | 3h       | 1/1/2015-12/31/2018 | 11688     |
| KA-TEMP    | 184  | 3h       | 1/1/2015-12/31/2018 | 11688     |
| KA-PM2.5   | 184  | 3h       | 1/1/2015-12/31/2018 | 11688     |

## Assortative Property Investigation
To further illustrate the characteristics of spatial-temporal signals across various datasets, we use feature similarity between nodes as a homophily measure to statistically analyze their assortative properties based on the constructed graphs. Specifically, we have defined the following indicators:

### Intra-homophily rate: probabilities of node pairs exhibiting similar observations at each time step.
