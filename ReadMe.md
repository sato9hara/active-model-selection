# ActiveModelSelection

* This is an implementation of **Active Model Selection: A Variance Minimization Approach** to appear in ACML 2024.

## Reproducing Experiments in Section 5
* Before running the experiments, please edit `config.json` and set the directories.
1. Prepare data.
```
cd ./Sec5/data
./download.sh
```
2. Train models.
```
cd ../
./train.sh
```
3. For reproducing the results in Section 5.2.
```
./Sec52.sh
```
4. For reproducing the results in Section 5.3.
```
./Sec53.sh
```
5. The result figures are available by running `Sec5/PlotFigures.ipynb`

## Reproducing Experiments in Section 6
* Before running the experiments, please edit `config.json` and set the directories.
1. Prepare data.
```
cd ./Sec6/data
./download.sh
```
2. Load models and evaluate them. This requires GPU.
```
cd ../model
./eval.sh mat 0
./eval.sh thr 0
./eval.sh top 0
```
3. For reproducing the results.
```
./test.sh
```
4. The result figures are available by running `Sec6/PlotFigures.ipynb`

