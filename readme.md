# Bayes Hack  
## Reduce Accidents in Mines  
Data provided by the US Department of Labor  

## Challenge  
Can we leverage mine inspections reports to determine the likelihood that a mine will 
have an accident in the future?  

### Datasets and details:
http://www.msha.gov/OpenGovernmentData/OGIMSHA.asp

#### Notes
For this challenge, we chose to focus on the inspections, violations, and site-employment data. Due to the time constraints, our goal was to build a simple prioritization web-app that ranked sites by their likelihood of having an accident.

## Labels
We chose to model the risk of a site having an accident as the probabilty of an accident following an inspection. For the binary classification problem, positive labels were given to sites having accidents in the previous or current calendar year. 

### Logistic Regression
For simplicity and explainabilty, we chose logistic regression to determine the probability a site will have an accident.

$$
\begin{eqnarray}
P(A|I) & = \frac{1}{1 + \exp(-t)}  \\
t1 & = \beta_{0} + \sum{\beta_{i} \cdot x_{i}} \\
\end{eqnarray}
$$

where $P(A|I)$ is the probability a site will have an accident following an inspection, $\beta$ are the coefficients determined from training, and $x_{i}$ are the features derived from the data. 

#### Simplicity
This model is a simple transformation of a weighted average. It's easy to train off-line and can be deployed easily in most standard database systems.

#### Explainability
The coefficients represent the impact of each feature. Therefore, you can easily assess the impact to the probability when one of the features is perturbed. In our app, we graph the raw metric along with the average metric for sites without accidents. This allows an inspector to see which feature triggered the high risk assessment.

## Features
To generate our risk score, we calculated several features based on previous inpsections and site employment data. Although we considered the violations data set, we did not find much predictive power due to an imbalance in the distribution of violations over sites.
  
### Previous Inspections
The following features are derived from previous inspections.
The activity codes are binarized dummy variables derived from
the __`ACTIVITY_CODE`__ column in the inspections dataset.
The total number of occurrences are then aggregated over all
previous inspections.

* __`ACTIVITY_CODE_E03`__ Written Hazard Complaint  
* __`ACTIVITY_CODE_E04`__ Verbal Hazard Complaint 
* __`ACTIVITY_CODE_E28`__ Mine Idle Activity  

### Mine Employment
The following features characterize the employment at the mines. They
reflect the total time the mines are in operation as well as the size of the
crews employed in parts of the mines.  
* __`MILL_OPERATIONPREPARATION_PLANT`__ Operation crew size  
* __`INDEPENDENT_SHOPS_OR_YARDS`__ Independent shop crew size  
* __`DREDGE`__ Dredge crew size  
* __`OFFICE_WORKERS_AT_MINE_SITE_ANN_HRS`__ Total time workers are on site  
* __`STRIP_QUARY_OPEN_PIT_ANN_HRS`__ Total hours mine is operational  
* __`UNDERGROUND`__ Underground crew size  

## Training the Model
Each sample represented a unique `MINE_ID, EVENT_NO` pair. Features were selected through an iterative greedy search while optimizing the area under the ROC curve (AUC). The datasets were split using a 80/20 randomized split. The AUC observed was around 0.8.

## Implementation Details
### Features
Most features were aggregates calculated in python using the pandas library. Any relational database system should be capable of performing the calculations. 

### Model
To train the logistic regression model, we used the scikit-learn library in python. Cross-validation and feature selection were also performed using the scikit-learn library.

### Deployment Considerations
Due to the size of the data, it's quite simple to train a model offline and then hard-code the coefficients in a script to calculate the scores. For example, in MS SQL Server, it's possible to create a stored procedure that uses features in a table derived from the raw data, generates a risk score, and then inserts the scores into the table.

## Final Thoughts
### Domain Expertise
We designed an application to help inspectors decide which sites to visit for further inspection. We made assumptions when developing the model that may not be applicable. However, we believe that this approach which combined with some domain expertise can produce a very effective tool for preventing accidents.

### Overfitting
Due to time constraints, we focused on building a prototype application that highlighted the potential of having a tool that predicted accidents. Although we applied the standard tests for assessing the generality of our model, we did not have the chance to assess the validity of our features. For example, you might imagine a scenario where our model works really well for large mines, but because of the features we chose, we are not able to accurately capture the risk at smaller mines. 


