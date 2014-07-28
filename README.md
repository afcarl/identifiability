Identifyability
===============

The potential for re-identification in an anonymized data set is a critical risk when preparing data for public release or for sharing with third parties. While a number of anonymization standards and practices exist (including k-anonymity, de-identification, aggregation, and introducing noise), evaluaing the anonymity of a data set remains a nontrivial task. 

In real-world social data, there are often latant dependancies between different variables (i.e., the probabilities of someone having both traits A and B are not independent). These relationships in the data are often non-intuitive or complex to model and account for when assessing the probability that an individual can be identified. 

Additionally, many strategies for evaluating the re-identifyability of a data set require joining the data with other data sets. Unfortunately, comparison data sets are often hard to come by, and linking data sets is often legally forbidden by terms of use.

This project offers re-identification risk assessment with the following features:
- Requires no outside comparison data sets.
- Makes no assumptions about distributions of or relationships between the variables in a data set.
- Returns actionable information about which variables and combinations of variables are most identifying.
- Returns a summary of how identifyable records in the data set are.
- Parallel implementation for multi-core computers.

This project makes the following assumptions about input data:
- The sample is representative (or exhastive) of some population.
- The sample has no columns for unique ID.
- The sample is in csv format. 

Example script calls:
`python identifyability.py -n 4 -c 1 -i data.csv -o summary.txt`
`python identifyability.py -h`
