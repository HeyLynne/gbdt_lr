# gbdt_lr
implement gbdt_lr with sklern
### Intro
A simple implementation of gbdt + lr. It was proposed by Facebook, you can get the paper from https://www.semanticscholar.org/paper/Practical-Lessons-from-Predicting-Clicks-on-Ads-at-He-Pan/daf9ed5dc6c6bad5367d7fd8561527da30e9b8dd.

It's now widely used in CTR prediction
In our case we use it to predict whether two shops are same store.

### Requirements
- python 2.7
- sklearn
- numpy

### Usage
python gbdt_lr_train.py --file_path PATH --test_path PATH
