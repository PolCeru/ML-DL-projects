# Google Play App Popularity Prediction

This project explores predicting the number of downloads for Google Play Store apps using different machine learning models. The main goal was to see whether a simple algorithm could perform well with a huge dataset, or if more advanced algorithms would do better on smaller subsets of the data.

---

## Overview

We worked with a large dataset of Google Play Store apps to build predictive models. The goal was to see if having lots of data could make a simple model competitive, or if more advanced models would still outperform simpler ones with less data. The workflow included thorough data cleaning, preprocessing, and then training and evaluating several classification algorithms.

---

## Methodology & Tools

* **Dataset**: A large collection of apps with multiple features.  
* **Data Preprocessing**: Done using Knime, including removing irrelevant columns, dropping rows with missing data, scaling numerical features, and converting text to numerical values.  
* **Algorithms Tested**:
    * CART (Classification and Regression Tree)  
    * Random Forest  
    * XGBoost  
    * Classifium  
* **Implementation**: Models were built and tuned using Scikit-learn. Hyperparameters were optimized using `RandomizedSearchCV` and Grid Search Cross-Validation.

---

## Results

Predicting app downloads turned out to be more challenging than expected, with all models struggling to achieve high accuracy.

* **CART**: The initial model overfitted the data, performing very well on training data but poorly on unseen data. After tuning, performance improved but remained limited.  
* **Random Forest**: This model also struggled, particularly with unevenly distributed download classes. The feature "Rating Count" was the most influential in predictions.  
* **XGBoost**: Showed signs of overfitting and had difficulty distinguishing between different download ranges.  
* **Classifium**: Tested on a smaller sample, this model produced a high error rate and did not improve overall performance.

---

## Challenges

* **Data Cleaning**: Handling such a large dataset and engineering useful features was a major hurdle.  
* **Hardware Limits**: Training some models on the full dataset required a lot of memory, which forced us to work with smaller subsets for tuning.

---

## Conclusion

The main takeaway is that more data doesn’t automatically lead to better predictions—especially if the dataset is noisy or lacks strong predictors. Apart from "rating" and "rating count," the existing features weren’t enough to accurately predict downloads. Future improvements would benefit from adding more predictive features like app usage, advertising data, or update frequency.
