#!/bin/sh
set -e
WIKI_DUMP_FILE_IN=data/enwiki-20220301-pages-articles-multistream1.xml-p1p41242
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."

python3 -m wikiextractor.WikiExtractor  $WIKI_DUMP_FILE_IN --processes 8 --json -l -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Succesfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"