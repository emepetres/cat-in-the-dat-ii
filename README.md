# Categorical Feature Encoding Challenge II

Binary classification, with every feature a categorical (and interactions!).
Second challenge of [cat-in-the-dat](https://www.kaggle.com/competitions/cat-in-the-dat-ii/submit) from kaggle.

NOTES:

* Logistic Regression works much better than Random Forest

## Train

```bash
conda activate ml
python -W ignore train.py --model=[lr|rf|svd|xgb]
```
