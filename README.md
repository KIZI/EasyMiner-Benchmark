# marcbench
Benchmark suite for EasyMiner

##Preparing data
Prerequisites include Python 2 with sci-kit learn and pandas.

 ```
 prepare_data.sh
 ```
The benchmark uses standard open datasets from the UCI repository. To ensure that  algorithm implementations in all platforms (Weka, R) operate on exactly the same folds, the folds are materialized. Two versions of the folds are created, one without discretization of numerical attributes and one with hit.  Missing values are treated in both versions.

The output is saved into 
```
data/folds
data/folds_nodiscr
```

The process also creates a temporary folder
```
data/output
```

###Running benchmarks - WEKA

Weka implementations of reference algorithms  of the two benchmark suites are executed using
```
./run_WEKA_Bench_Acc.sh
./run_WEKA_Bench_Scaling.sh
```
The output is stored into
```
/result
```
###Running benchmarks  - Python
Sci-Kit decision tree benchmarks are run with  

```
python PDT.py
```

The output is stored into
```
/result
```

###Running benchmarks - EasyMiner

First, it is necessary to input valid API_KEY and API_URL into `easyminercenter_api_config.py`

The default benchmark of the rCBA implementation in EasyMiner is run with 

```
python em_api.py
```
By default, the benchmark runs in five parallel threads. This can be changed by modifying `PARALLEL_THREADS` variable in `em_api.py`.

###Generating won-tie-loss matrix
All benchmarks are saved into:
```
/result
```

The won-tie-loss matrix and Wilcoxon signed rank test are executed using `wontieloss.py`.
The result is saved into the same folder. 