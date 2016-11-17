#!/bin/bash

#Start tor and polipo in new terminal windows

echo -e 'Starting tor and polipo in new terminal windows...\n'

osascript -e 'tell application "Terminal" to do script "tor"'
osascript -e 'tell application "Terminal" to do script "polipo -c ~/.polipo"'

#Check tor status via tor check page
TRY=0
ONTOR=0
while [  $TRY -lt 10 ]; do
  sleep 1s
  if curl --socks5 localhost:9050 -s https://check.torproject.org/ | grep -q 'Congratulations'
  then
    # code if found
    let ONTOR=1
    let TRY=10
  else
    # code if not found
    let ONTOR=0
  fi
  let TRY=TRY+1
done
if [ $ONTOR -eq 1 ]
then
#Show caller tor check page
echo -e 'Reconfirming your running tor...\n'

SUCCESS=`curl --socks5 localhost:9050 -s https://check.torproject.org/ | grep -m 1 "Congratulations"`

echo -e $SUCCESS

echo -e '\n'

#Run scraper

echo -e 'Running scrapy scraper...\n'

osascript -e 'tell application "Terminal" to do script "cd other/ms_courses/capstone/DS-GA-1006-Project/streeteasy_scrapy/streeteasy_sales; scrapy crawl streeteasy -s JOBDIR=crawls/streeteasy-1"'

else
echo -e 'Sorry. You are not using Tor- not scraping streeteasy.'
fi


