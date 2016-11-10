# Scraping streeteasy
## 11-10-2016

### Running scraper

This scrapy project scrapes the streeteasy sales pages. It is run using the bash script:

```
./run_scrapy.sh
```

The scraped data is then zipped and uploaded to S3 by running the bash script:

```
./upload_zip_to_s3.sh
```

Note this script depends on configuration of the aws CLI, and is included solely for documentation of data provinance.

## Notes on design of scraper

This scraper scrapes the pages for streeteasy sales and saves them in the data directory as raw html pages. Note the html is not parsed, and features are not extracted, since obtaining the html data is anticipated to be a bottleneck in the project pipeline. Hence saving the entire html page avoids any potential future issues where the parser failed to capture potentially useful features from the raw html.

As a final note, the scraper is also set to run using Tor/polipo. For reference on this process, see this [blog](http://pkmishra.github.io/blog/2013/03/18/how-to-run-scrapy-with-TOR-and-multiple-browser-agents-part-1-mac/).
