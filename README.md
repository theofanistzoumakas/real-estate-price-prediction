# 🏠Real estate median price prediction using machine learning algorithms
The program implements machine learning models for houses' price prediction, based on houses' characteristics, and shows the classifiers according to the machine learning model. The program calculates mean and absolute square error and uses  10 fold cross validation for each machine learning model.

## ⚡Features
 - Pre-proccess data:
    - One Hot Vector
    - Scaling data
    - Filling null values with the column's mean value
 - Threshold calculation (threshold is the mean value)
 - Histgram construction
 - Diagrams construction with values (values are divided into four diagramms for visual clarity reasons):
    - 1st: Scaled Longitude, scaled latitude and near ocean columns
    - 2nd: Scaled total rooms, scaled total bedrooms and less than 1h ocean columns
    - 3rd: Scaled households, scaled median income and inland columns
    - 4th: Scaled median house value and island
 - Classifier construction using all data through perceptron algorithm
 - Classifier construction using all data through least squares algorithm
 - Ten fold cross validation:
    - Using perceptron and least squares algorithm and
    - Mean absolute error calculation
    - Mean square error calculation
 - Ten fold cross validation:
    - Using multilayer network
    - Mean absolute error calculation
    - Mean square error calculation

**This application is developed solely for academic and research purposes.**
## 🔒 Code Ownership & Usage Terms
This project was created and maintained by:

Theofanis Tzoumakas (@theofanistzoumakas)
Konstantinos Pavlis (@kpavlis)
Michael-Panagiotis Kapetanios (@KapetaniosMP)

🚫 Unauthorized use is strictly prohibited.
No part of this codebase may be copied, reproduced, modified, distributed, or used in any form without explicit written permission from the owners.

Any attempt to use, republish, or incorporate this code into other projects—whether commercial or non-commercial—without prior consent may result in legal action.

For licensing inquiries or collaboration requests, please contact via email: theftzoumi _at_ gmail _dot_ com .

© 2025 Theofanis Tzoumakas, Konstantinos Pavlis, Michael-Panagiotis Kapetanios. All rights reserved.
