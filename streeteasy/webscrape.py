from bs4 import BeautifulSoup
import re
import pprint
import sys
import urllib2
#import pandas as pd
#import pandas as np

FIELDS = [ 'address', 'price', 'neighborhood', 'num_beds', 'num_baths', 'num_sqft', 'type', 'url',
		   'amenities_list', 'transit_list', 'gps_coordinates',
		   'building_name', 'built_date', 'building_num_units', 'building_url']


def clean_string(string):
	return string.strip().replace(' ', '_').replace('-', '_').replace('/', '_').replace(',', '_').lower()

def clean_number(number):
	return number.replace(',', '').replace('$', '')


def find_first_value_or_none(haystack, key1, key2):
	search_result = haystack.find_all(key1, key2)
	return search_result[0].string if search_result else None


def parse_listing_url(url, data):
	try:
		r = urllib2.urlopen(url, timeout=5).read()
	except:
		return
	soup = BeautifulSoup(r, 'html.parser')

	listings_info = soup.find_all('div', {'class' : 'details'})[0].find_all('div', {'class' : 'details_info'})

	if len(listings_info) > 0:
		listing_details = listings_info[0]
		for detail in listing_details:
			if detail.string:
				if 'bed' in detail.string:
					data['num_beds'] = clean_number(detail.string.split()[0])
				elif 'studio' in detail.string:
					data['num_beds'] = 'studio'
				elif 'bath' in detail.string:
					data['num_baths'] = clean_number(detail.string.split()[0])
				elif 'ft' in detail.string and 'per ft' not in detail.string:
					data['num_sqft'] = clean_number(detail.string.split()[0])
	if len(listings_info) > 1 and 'in' in listings_info[1].get_text():
		sep_found = False
		listing_type = []
		neighborhood = []
		for item in listings_info[1].stripped_strings:
			if item == 'in':
				sep_found = True
				continue
			if item == ',':
				assert not sep_found, listings_info[1]
				listing_type = []
				continue
			if sep_found:
				neighborhood.append(item)
			else:
				listing_type.append(item)
		if sep_found:
			data['type'] = clean_string('_'.join(listing_type))
			data['neighborhood'] = clean_string('_'.join(neighborhood))

	building_info = soup.find_all('div', {'class' : 'in_this_building big_separator'})[0]
	for item in building_info.find_all('a'):
		building_url = item.get('href')
		if 'building' in building_url or 'property' in building_url:
			data['building_url'] = 'http://streeteasy.com' + item.get('href')
			data['building_name'] = clean_string(item.string)
			break

	for details in building_info.find_all('div', { 'class' : 'details_info'}):
		for item in details.stripped_strings:
			item = clean_string(item)
			if 'built_in' in item:
				data['built_date'] = clean_number(item.split('_')[-1])
			elif 'units' in item:
				data['building_num_units'] = clean_number(item.split('_')[0])

	data['amenities_list'] = []
	amenities = soup.find_all('div', {'class' : 'amenities big_separator'})	
	for sections in amenities:
		for li in sections.find_all('li'):
			for item in li.stripped_strings:
				item = clean_string(item)
				if 'googletag' not in item:
					data['amenities_list'].append(item)
					break

	data['transit_list'] = []
	transits = soup.find_all('div', {'class' : 'transportation'})	
	for section in transits:
		for p in section.find_all('p'):
			cluster = ''
			for item in p.stripped_strings:
				if 'mile' in item or 'feet' in item or 'at' in item:
					item = clean_string(item)
					cluster = cluster + '_'
				cluster = cluster + item
			data['transit_list'].append(cluster)
		for item in section.find_all('a'):
			google_maps_address = item.get('href')
			if 'www.google.com/maps' in google_maps_address:
				re_match = re.search('40.[0-9]+,-7[3-4].[0-9]+', google_maps_address)
				if re_match:
					data['gps_coordinates'] = re_match.group(0).replace(',', ' ')


processed_urls = set()
				
with open('sales_listings.csv', 'w') as fout:
	fout.write(','.join(FIELDS) + '\n')
	for pageno in range(1, 999999):
		page_url = 'http://streeteasy.com/for-sale/nyc?page={}'.format(pageno)
		try:
			r = urllib2.urlopen(page_url, timeout=5).read()
		except urllib2.HTTPError:
			break
		soup = BeautifulSoup(r, 'html.parser')
		listings = soup.find_all(lambda tag : tag.has_attr('data-id'))
		if not listings: break
		print 'processing {} listings on {}'.format(len(listings), page_url)
		for listing in listings:
			title = listing.find_all('div', {'class' : 'details-title'})[0].find_all('a')[0]
			url = 'http://streeteasy.com' + title.get('href')
			if url in processed_urls: continue
			details = listing.find_all('div', {'class' : 'details_info'})
			listing_data = {}
			listing_data['url'] = url
			listing_data['address'] = clean_string(title.string)
			listing_data['price'] = clean_number(listing.find_all('span', {'class' : 'price'})[0].string)
			parse_listing_url(url, listing_data)
			for field in FIELDS:
				if field in listing_data and type(listing_data[field]) == list:
					fout.write(' '.join(listing_data[field]))
				elif field in listing_data:
					fout.write(listing_data[field])
				if field != FIELDS[-1]:
					fout.write(',')
				else:
					fout.write('\n')
			processed_urls.add(url)