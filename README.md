# Classification-of-12-lead-ECGs
https://physionetchallenges.github.io/2020/
The PhysioNet/Computing in Cardiology Challenge 2020
Automatic detection and classification of cardiac abnormalities can assist physicians in the diagnosis of the growing number of ECGs recorded.
The data source is the public and unused data from the China Physiological Signal Challenge in 2018 (CPSC2018), held during the 7th International Conference on Biomedical Engineering and Biotechnology in Nanjing, China.


# Results


Dataset| Loss | Accuracy |
--- | --- | 
Training  | 0.015 | 0.96 |
Validation  | 0.102 | 0.82 |


Dataset | f-2 score  | g-2 score | geometric mean |
--- | --- | --- | 
Training  | 0.367 | 0.115 |  0.205 | 
Testing  | 0.365 | 0.114 |  0.204 |


Actual value| AF | I-AVB | LBBB | Normal | PAC | PVC | RBBB | STD | STE |
277 | 1.0  |  0.0 |  0.0  | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 | 0.0
Predicted value | AF | I-AVB | LBBB | Normal | PAC | PVC | RBBB | STD | STE |
277 | 0.999 | 0.0 | 0.0  | 0.0 | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
