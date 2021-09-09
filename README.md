# causal-machine-learning-models

In this code, I investigate the finite sample properties of three estimators for my pa- rameter of interest which is the average treatment effect. This paper covers the OLS, inverse probability weighting and "naive" double machine learning estimation approaches. I vary several dimensions of the design that are of practical importance, like the underlying data generation process, the sample size and aspects of the selection process for the treatment. Moreover, I decided to examine several research designs.


# "Naive" double machine learning approach: 
This subchapter covers a comparison between the OLS and the "naive" double machine learning approach. The latter is a very general toolbox for estimation and inference of econometric parameters with machine learning methods. The double machine learning framework deals with the problem of biased parameter estimates (e.g. treatment effect) due to high dimensionality.

I therefore assumed to be in the following framework:

<a href="https://www.codecogs.com/eqnedit.php?latex=y_i=d_i\theta&space;&plus;g_0({x}_i)&plus;u_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i=d_i\theta&space;&plus;g_0({x}_i)&plus;u_i" title="y_i=d_i\theta +g_0({x}_i)+u_i" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=d_i=m_0({x}_i)&plus;v_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_i=m_0({x}_i)&plus;v_i" title="d_i=m_0({x}_i)+v_i" /></a>

Where Theta is my parameter of interest (average treatment effect), xi are a set of control variables and ui as well as vi are error terms. Next to the OLS approach, I want to estimate the average treatment effect in this setting with the "Naive" double machine learning estimator stated above. I retrieve this approach:

1: Estimate <a href="https://www.codecogs.com/eqnedit.php?latex=d_i=\hat{m}_0({x}_i)&plus;\hat{v}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_i=\hat{m}_0({x}_i)&plus;\hat{v}_i" title="d_i=\hat{m}_0({x}_i)+\hat{v}_i" /></a>

2: Estimate <a href="https://www.codecogs.com/eqnedit.php?latex=y_i=\hat{g}_0({x}_i)&plus;\hat{u}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i=\hat{g}_0({x}_i)&plus;\hat{u}_i" title="y_i=\hat{g}_0({x}_i)+\hat{u}_i" /></a>

3: Estimate <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\theta}=(\sum_{i=1}^N&space;\hat{v}_id_i)^{-1}\sum_{i=1}^N&space;\hat{v}_i&space;(y_i-\hat{g}_0({x}_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{\theta}=(\sum_{i=1}^N&space;\hat{v}_id_i)^{-1}\sum_{i=1}^N&space;\hat{v}_i&space;(y_i-\hat{g}_0({x}_i))" title="\hat{\theta}=(\sum_{i=1}^N \hat{v}_id_i)^{-1}\sum_{i=1}^N \hat{v}_i (y_i-\hat{g}_0({x}_i))" /></a>


Result:

![dml_estimator_distribution](https://user-images.githubusercontent.com/32592350/132741529-e83758cd-4258-4fe0-b476-a31121099923.png)

