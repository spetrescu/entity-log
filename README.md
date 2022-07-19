# entity-log
Trained Bi-LSTM model that transforms raw logs into rich structured information.

## Pretrained model
Fetch the pretrained model by running: <br>
```
1. pip install gdown
2. gdown https://drive.google.com/uc?id=1oz8wl6MbXFFkpSh83sSMIM0dIQ37ZYre
```

## Usage
<img width="1264" alt="entity_log_workflow_diagram" src="https://user-images.githubusercontent.com/60047427/178634796-d7630fb2-c76f-4895-97c9-848d4bc2d1f1.png">

## Example output
```
EXAMPLE 1
Word   (True) : Pred
Progress      : CONSTANT
of            : CONSTANT
TaskAttempt   : CONSTANT
attempt_1445062781478_0013_m_000001_1000: ID
is            : CONSTANT
:             : CONSTANT
0.19211523    : GENERIC_VAR


EXAMPLE 2
Word   (True) : Pred
Reading       : CONSTANT
broadcast     : CONSTANT
variable      : CONSTANT
14400         : ID
took          : CONSTANT
3             : GENERIC_VAR
ms            : CONSTANT

EXAMPLE 3
Word   (True) : Pred
Deleting      : CONSTANT
instance      : CONSTANT
files         : CONSTANT
/var/lib/nova/instances/ad084e02-8ba7-4d63-a17a-6721587c38f6_del: PATH
```
