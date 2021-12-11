**************************************************CS685 Assignment #1********************************************************************************************************


How to run the code:
1. Run <bash assign1.sh>

Components of assignment1.sh:
For task1: question1.sh
For task2: case-generator.sh
For task3: edge-generator.sh
For task4: neighbor-generator.sh
For task5: state-generator.sh
For task6: zscore-generator.sh
For task7: method-spot-generator.sh
For task8: top-generator.sh

For some tasks, I have created some helper files such as hardcode.py, neighbor_modified.py etc.

Data Used:
Raw data from covid API in json format.

Assumptions Made:
1. The districts were extracted from Raw jsons. There were a few district entries such as "Airport Quarantine" that were ignored.
2. Only the currently Hospitalized Field was considered.
3. The mapping with neighbor-districts.json was made by word matching, Subsequence and Edit distance(Less than 3)
4. In the end, there were a few districts that was hard coded.
5. Due to some errors, there were some districts having negative cases and therefore they were removed.

System Requirements to run:
python3,jq,bash,json(python package),pandas,collections(python package)
