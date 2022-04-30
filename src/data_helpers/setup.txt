# make sure to be in the data_helpers folder
mkdir ../data_2017-09/


# for downloading reddit September 2017 data set
wget https://files.pushshift.io/reddit/comments/RC_2017-09.bz2
mv RC_2017-09.bz2 ../data_2017-09/
bzip2 -d ../data_2017-09/RC_2017-09.bz2

# for setting up data folders
mkdir ../data_2017-09/webpages
mkdir ../data_2017-09/queries
mkdir ../data_2017-09/pyserini