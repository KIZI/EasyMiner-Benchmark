# marcbench
Benchmark suite for EasyMiner

## Preparing data
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

Since this proces takes long, precomputed folds are shipped zipped in 
```
prepared_data
```
and can be unzipped with

```
unzipfolds.sh
```




### Running benchmarks - WEKA

Weka implementations of PART, J48 and RIPPER with grid-based metaparameter optimiziation  are executed using
```
./run_WEKA_Bench_Acc.sh
```

All benchmarks use raw, undiscretized data.

If interrupted, running the file again will compute the missing results.

### Running benchmarks  - Python
Sci-Kit decision tree benchmarks are run with  

```
python PDT.py
```

Uses raw, undiscretized data.


### Running benchmarks - EasyMiner

First, it is necessary to input valid API_KEY and API_URL into `easyminercenter_api_config.py`

The default benchmark (cba_d) of the rCBA implementation in EasyMiner is run with 

```
./cba_d.sh
```
The benchmark of auto-tuned CBA (cba_a) can be run with
```
./cba_a.sh
```


By default, the benchmarks run in five parallel threads. This can be changed by passing `PARALLEL_THREADS` command line option to `cba_d.sh` or `cba_a.sh`.

Uses discretized data.

If interrupted, running the file again will compute the missing results.


Note that `cba_a.sh` returns slightly different results in each execution due to time limits used in the optimization algorithm.

### Generating won-tie-loss matrix
The won-tie-loss matrix and Wilcoxon signed rank test are executed using:
```
python wontieloss.py
```

All benchmarks are saved into:
```
/result
```
