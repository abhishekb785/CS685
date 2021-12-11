printf "Executing question1..\n"
tail -n +13 articles.tsv >temp2
python3 question1.py

printf "Executing question2..\n"

tail -n +14 categories.tsv >temp
awk -F ' ' '{print $2}' temp > temp2
python3 question2.py

printf "Executing question3..\n"

tail -n +14 categories.tsv >temp
python3 question3.py

printf "Executing question4..\n"

tail -n +18 shortest-path-distance-matrix.txt > temp8
python3 question4.py

printf "Executing question5..\n"

python3 question5.py

printf "Executing question6..\n"

tail -n +17 paths_finished.tsv >temp3
awk -F ' ' '{print $4}' temp3 > temp4

printf "Executing question7..\n"

python3 question6_7.py

printf "Executing question8..\n"

python3 question8.py
printf "Executing question9..\n"

python3 question9.py
printf "Executing question10..\n"

tail -n +18 paths_unfinished.tsv >temp5
awk -F ' ' '{print $4,$5}' temp5 > temp6
python3 question10.py

printf "Executing question11..\n"

tail -n +18 shortest-path-distance-matrix.txt > temp10
python3 question11.py

rm temp temp2 temp8 temp3 temp4 temp5 temp6 temp10
