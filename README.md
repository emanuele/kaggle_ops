kaggle_ops
==========

My best submission to the Kaggle competition "Online Product Sales",
ranked 21th over 366 teams (score: 0.57885).

http://www.kaggle.com/c/online-sales/leaderboard

----

Requirements:

0) NumPy, scikit-learn

1) Pandas, http://pandas.pydata.org/ , just to create the initial
dataset or to explore it.

2) joblib, http://packages.python.org/joblib/ , if you want to run
blender_parallel.py , i.e. to use all your cores with
GradientBoostingRegressor().

3) The "Online Product Sales" trainset/testset files to be put in the
subdirectory "data/".

----

Usage:

- "python explore.py" to have a look to the quantitative variables and
to decided which of them to put in the logscale.

- "python create_dataset.py" , creates and save the dataset from the
initial trainset/testset files.

- "python blender.py" computes the actual submission (simple blending
of GradientBoosting).

- "python blender_parallel.py" computes the actual submission
splitting the computation on as many cores as you like.

