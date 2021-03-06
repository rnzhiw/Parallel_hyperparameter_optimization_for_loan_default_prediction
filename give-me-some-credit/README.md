# Parallel hyper-parameter optimization for loan default prediction
Official Python implementation for our Parallel hyper-parameter optimization for loan default prediction framework.

Hyper-parameter selection can significantly impact
the performance of the machine learning model. Due to the large scale of data, parallel hyper-parameter selection is necessary for practical applications. Compared with the widely-used grid search and random search, Bayesian optimization is a global wise method proposed in recent years with fewer iterations. We consider these three methods for hyper-parameter selection in their parallel implementations. In many real-world applications such as Internet financial lending, delayed loan review often hurts business efficiency, thus faster processing is required. 



## Environment

* python3.7, pillow, tqdm, torchfile, pytorch1.1+ (for inference)

  ```
  pip install pillow
  pip install tqdm
  pip install torchfile
  conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
  ```

Then, clone the repository locally:

```
git clone https://github.com/rnzhiw/Parallel_hyperparameter_optimization_for_loan_default_prediction.git
```



## Test

**Step 1: Prepare images**

* All images are in the givemesomecredit/ Data folder

## Results

* Result1: Use GridSearch for parallel operations

   <img src="img/grid.png" width = "350" alt="图片名称" align=center />

* Result2: Use RandomSearch for parallel operations

   <img src="img/random.png" width = "350" alt="图片名称" align=center />

* Result3: Use Bayesian Optimiza for parallel operations

  <img src="img/Bayesia.png" width = "350" alt="图片名称"/>  













