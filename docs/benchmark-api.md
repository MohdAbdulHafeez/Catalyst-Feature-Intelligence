\# Benchmark API



\## Endpoint



`POST /api/v1/benchmark`



Executes the CATALYST leakage-safe categorical encoding benchmark

against a stored dataset.



\## Request



```json

{

&#x20; "dataset\_id": "dataset-id",

&#x20; "target\_column": "target",

&#x20; "categorical\_columns": \[

&#x20;   "city",

&#x20;   "segment"

&#x20; ],

&#x20; "metric": "accuracy",

&#x20; "cv\_splits": 3,

&#x20; "shuffle": true,

&#x20; "random\_state": 42,

&#x20; "task": "classification"

}

