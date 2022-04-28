# make sure to be in the data_helpers folder

mkdir ../data_2017-09/
wget https://files.pushshift.io/reddit/comments/RC_2017-09.bz2
mv RC_2017-09.bz2 ../data_2017-09/
bzip2 -d ../data_2017-09/RC_2017-09.bz2

mkdir ../data_2017-09/websites_train
mkdir ../data_2017-09/websites_val
mkdir ../data_2017-09/websites_test