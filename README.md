# NAS to Predict Recidivism
1st Place Submission for 2022 MLH AI Hacks 4 Good Hackathon 

## Purpose
Recidivism algorithms are used by probation departments, where defendants classified for a higher risk of recidivism are more likely to be detained. One of the most widely used assessment tool was developed by Northpointe. Their model was trained upon many classifications such as age, previous sentences, gender, but not race. This was a method of data blindness, however, it was found that when all other features are held constant, black defendants were predicted an almost doubled rate of recidivism compared to white defendants. These results are further propagated in their treatment of being detained more. Thus, the AI used had a direct negative impact on black defendants.

## Model Summary
- **Purpose**: Build an AI model classifying risk of recidivism with minimal bias.
- **Dataset:** COMPAS
- **Features:** Sex, age, race, time in jail, number of juvenile felonies, juvenile misdemeanors, and other juvenile counts, prior convictions, charge degree
- **Split**: Temporally split data, 80:10:10 training-validation-test
- **Metric:** counterfactual fairness - as an example of one of the non-differentiable fairness metrics that would otherwise be hard to incorporate in an end-to-end fashion

## Neural Architecture Search (NAS)
Many fairness metrics are useful only to guide model design and cannot be directly incorporated into a loss function due to non-differentiability (Kusner et al., 2017). To address this, we use Neural Architecture Search (NAS) with the fairness metric as a reward signal.
<img style="float: right;" src="https://github.com/JuliaLWang8/julialwang8.github.io/blob/master/src/media/NAS.png" width="700">

The Controller NN uses the reward and counterfactual fairness metric provided by the Child NN validation to adjust the Child NN’s sensitive characteristic thresholds and hidden layer sizes.

**Hyperparameters:** Our method uses an additional recurrent controller network alongside the primary prediction neural net (child), in line with Zoph et al. (2016). The control network (RL agent) is able to learn several critical hyperparameters through the feedback from the child network.

**Methodology:** 
- Trained a “naive” classifier to get a baseline result to compare against, where we arbitrarily picked some hyperparamaters. This model converged quickly, achieving a counterfactual fairness metric of 0.043.
- Our controller network was run over 1500 episodes, where the best architecture it produced was taken. We then trained a new child model from scratch with those hyperparameters.

## NAS Key Results
- 3.5x reduction in counterfactual fairness (as measured by the standard deviation of positive classification rates between demographic categories)
- Does not harm accuracy significantly
- Provides a general method for incorporating non-differentiable fairness metrics into model

Naive          |  NAS
:-------------------------:|:-------------------------:
<img width="357" alt="image" src="https://user-images.githubusercontent.com/55002716/223922582-aa921c38-54fe-4686-97f7-5b8f6052d6ab.png"> | <img width="339" alt="image" src="https://user-images.githubusercontent.com/55002716/223922685-034981f0-d7a2-4625-87bb-ab84623cc104.png">

