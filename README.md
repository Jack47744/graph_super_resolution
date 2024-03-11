# DGL2024 Brain Graph Super-Resolution Challenge

## Contributors

- Metis Sotangkur (ms922@ic.ac.uk)
- Wutikorn Ratanapan (wr323@ic.ac.uk)
- Carlos Brat (cb1223@ic.ac.uk)
- Marios Charalambides (mc1122@ic.ac.uk)
- Aryan Agrawal (aa6923@ic.ac.uk)

## Problem Description

Conventional equipment designed for detailed brain connectivity capture is often expensive and may not be readily accessible in various regions. However, devices that provide a rough approximation of brain connectivity are more widely accessible. This discrepancy poses a significant question: why can't we infer higher-resolution brain connectivity using more accessible devices currently available? In response to this challenge, our team, Super Idol, has developed an enhanced version of the AGSR-Net, Super-AGSR-Net. Our Super-AGSR-Net recognizes the importance of complex structural patterns in brain connectivity analysis and aims to better capture these intricacies for improved high-resolution predictions.

## Super Idol - Methodology

Our Super-AGSR-Net is the enhance version of the AGSR-Net. Inspired by the transformer model, we employ the attention network with the residual connection in Graph U-Net to capture varying node importance and long-range dependencies between non-neighbours. On the other hand, the discriminator’s residual connection helps counteract the vanishing gradients of the deep discriminator, allowing more efficient training. With an improved discriminator, Super-AGSR-Net’s generator receives better training feedback, encouraging it to produce a more accurate graph.


- Figure of your model.

## Used External Libraries

To set up your environment for the project, you will need to install `networkx`, `optuna`, and `torch`.

```bash
pip install -q networkx optuna torch  
```

## Results
![AGR-Net pipeline](/imgs/bar_plot.png)
From the bar chart in 3-fold cross validation.


## References
Isallari, M., Rekik, I.: Brain graph super-resolution using adversarial graph neural network with application to functional brain connectivity. Medical Image Analysis 71 (2021) 102084. Elsevier.
