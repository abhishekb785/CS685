#!/bin/sh
python3 create.py

touch dis.txt
jq 'keys' neighbor-districts.json > dis.txt
cut -f 1 -d '/' dis.txt > dis2.txt
sed 's/"//' dis2.txt > dis.txt
cat dis.txt | tr '_' ' ' > dis2.txt
tail -n +2 dis2.txt > dis.txt
sed -i '$ d' dis.txt
sed 's/\<district\>//g' dis.txt > dis2.txt
sed 's/\<city\>//g' dis2.txt > dis.txt
rm dis2.txt

touch neighbor-district-modified.json
cat neighbor-districts.json | tr '_' ' ' > temp1.json
sed 's/\<district\>//g' temp1.json > temp2.json
sed 's/\<city\>//g' temp2.json > neighbor-district-temp.json

python3 neighbor_modified.py > mapping.txt
python3 hardcode.py > b.json
head -c -2 < b.json | tail -c +2 > a.json
sed "s/'//g" a.json > neighbor-districts-modified.json
rm temp1.json
rm a.json
rm b.json
rm temp2.json
