# Towards Green Recommender Systems: Investigating the Impact of Data Reduction on Carbon Footprint and Algorithm Performances

Source code for the paper "Towards Green Recommender Systems: Investigating the Impact of Data Reduction on Carbon Footprint and Algorithm Performances", submitted at ACM RecSys 2024 - Short Paper Track.

## Abstract

This work investigates the path toward green recommender systems by examining the impact of data reduction on both algorithm performance and carbon footprint. In the pursuit of developing energyefficient recommender systems, we specifically investigated whether and how a reduction of the training data impacts the performances of several representative recommendation algorithms. In order to obtain a fair comparison, all the algorithms were run based on the implementations available in a popular recommendation library, i.e., RecBole, and used the same experimental settings. Results indicate that: (a) data reduction can be a promising strategy to make recommender systems more sustainable at the cost of a reduction in the predictive accuracy; (b) training recommender systems with less data makes the suggestions more diverse and less biased. Overall, this study contributes to the ongoing discourse on the development of recommendation algorithms that meets the principles of SDGs, laying the groundwork for the adoption of more sustainable practices in the field.

In this repository, we show how to reproduce the results we obtained.

## Requirements 

The requirements needed to reproduce our experiments are related to the two libraries we used, [RecBole](https://recbole.io/docs/) and [CodeCarbon](https://mlco2.github.io/codecarbon/).
Please refer to the original repositories to set proper virtual environments (they mainly require PyTorch).

## Executing the experiments

This repo contains 2 python files, which are used to execute the experiments:
- `exec_model.py`
- `run.py`

`run.py` is used to set the experimental setting, in terms of recommendation models to be trained, the dataset, and the data reduction to be applied; `exec_model.py` is called by `run.py` and executes the experiments, using the libraries RecBole and CodeCarbon to train the recommendation models and track the emissions, respectively.

In addition, we report, in the `dataset` folder, data we used to carry out our experiments. For each dataset (`Movielens1M` and `Amazon-Books`) we report the original version, the data-reduced versions (10 splits, but in our experiments we used only 5 splits), and the scripts to generate them, as `.ipynb` python notebooks.

As output, you will have, for each dataset and for each folder, the results related to each split (quantity of training data); in each folder, there will be two files, one related to the recommendation metrics, and another file related to the emission metrics. These files are used in our python notebooks (described in the next section) to evaluate the trade-off between emission, recommendation metrics, and data reduction.

## Plot the results

Finally, in the `results` folder, we put the results we obtained as anonymized `.tsv` files (we anonymized and globalized to guarantee double-blind the results as discussed in the paper), with the `.ipynb` python notebooks we used to process them and generate the plots in the paper.

These files are `movielens_analysis.ipynb` and `amazon_books_analysis.ipynb`, that read the output of the experiments (folder for each dataset, model, and spli), or, alternatively, the aggregated files `results_movielens.tsv` and `amazonbooks.tsv` (that are generated by the same notebooks in a smart way).