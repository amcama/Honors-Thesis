# Teaching a neural network to replicate a rule-based approach for biomedical information extraction

## Description

This project involves training the BERT model using different approaches for . ...


## Configurations

1. Configuration 1: Classifies the sentences using the [CLS] embedding.
2. Configuration 2: Classifies the sentences using the [CLS] embedding and adds 4 entity markers (begin controller [E1], end controller [/E1], begin controlled [E2], end controlled [/E2]).
3. Configuration 3: Classifies the sentences using the [CLS] embedding and adds 4 entity markers. This configuration max pools the data in the forward function.

### Executing program

Run the script using the following command:

```bash
python training.py $configuration
```

where configuration is the approach to use when training the BERT model.



## Authors

Andrea Camarillo


## Acknowledgments
I would like to acknowledge the authors of the notebook [chap13_classification_bert.ipynb](https://github.com/clulab/gentlenlp/blob/main/notebooks/chap13_classification_bert.ipynb) from the [GentleNLP](https://github.com/clulab/gentlenlp) repository. Their work served as a valuable reference and inspiration for this project.
