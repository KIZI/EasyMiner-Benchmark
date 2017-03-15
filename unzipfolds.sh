cp -r prepared_data/* data
cd data/folds
unzip train.zip
unzip test.zip
rm train.zip
rm test.zip
cd ../folds_nodiscr
unzip train.zip
unzip test.zip
rm train.zip
rm test.zip