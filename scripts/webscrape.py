from bs4 import BeautifulSoup
from multiprocessing import Pool
import re
import gzip
import os
import string
import urllib2


def clean_string(string):
	return '_'.join(string.encode('ascii', 'ignore').lower().replace('-', ' ').replace('/', ' ').replace(',', ' ').strip().split())

def clean_number(number):
	return number.replace(',', '').replace('$', '')

PUNCTUATION_TO_SPACE = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

FIELDS = [ 'saleid', 'address', 'price', 'neighborhood', 'borough', 'status', 'date',
		   'num_beds', 'num_baths', 'num_sqft', 'type', 'url', 'monthly_cost',
		   'amenities_list', 'transit_list', 'gps_coordinates', 'school_district',
		   'building_name', 'built_date', 'building_num_units', 'building_url', 'description']

NYC_BOROUGHS = map(lambda b: clean_string(b), ['Manhattan', 'Queens', 'Brooklyn', 'Staten Island', 'Bronx'])


def parse_listings_html(html, data):
	soup = BeautifulSoup(html, 'html.parser')

	location_vector = soup.find('title').get_text().split('|')[0].strip().split(' in ')[-1].split(',')
	if len(location_vector) != 2:
		print '{} : unexpected location vector {}'.format(data['saleid'], location_vector)
		return False

	# parse location
	data['neighborhood'] = clean_string(location_vector[0].strip())
	data['borough'] = clean_string(location_vector[1].strip())

	if data['borough'] not in NYC_BOROUGHS:
		print '{} : invalid borough {}'.format(data['saleid'], data['borough'])
		return False

	# parse address
	data['address'] = clean_string(soup.find('h1', {'class' : 'building-title'}).get_text().strip())

	# parse price
	price_info = soup.find('div', {'class' : 'details'}).find('div', {'class' : 'details_info_price'})
	for item in price_info.find('div', {'class' : 'price'}).stripped_strings:
		if '$' in item:
			data['price'] = clean_number(item)

	# parse listing status
	price_status = price_info.find('div', {'class' : 'status'})
	if price_status:
		for item in price_status.stripped_strings:
			if 'Register to see what it closed for' in item:
				data['status'] = clean_string('SOLD')
			else:
				data['status'] = clean_string(item)
			break
		secondary_status = price_status.find('span', {'class' : 'secondary'})
		if secondary_status:
			data['date'] = secondary_status.get('title').encode('ascii', 'ignore')
		else:
			data['date'] = ''
	else:
		data['status'] = ''
		data['date'] = ''

	# parse listing information
	listings_info = soup.find('div', {'class' : 'details'}).find_all('div', {'class' : 'details_info'})

	if len(listings_info) > 0:
		listing_details = listings_info[0]
		for item in listing_details.stripped_strings:
			if 'bed' in item:
				data['num_beds'] = clean_number(item.split()[0])
			elif 'studio' in item:
				data['num_beds'] = 'studio'
			elif 'bath' in item:
				data['num_baths'] = clean_number(item.split()[0])
			elif 'ft' in item and 'per ft' not in item:
				data['num_sqft'] = clean_number(item.split()[0])

	if len(listings_info) > 1 and 'in' in listings_info[1].get_text():
		listing_details = listings_info[1]
		for item in listing_details.stripped_strings:
			if item != 'in': data['type'] = clean_string(item)
			break

	# parse additional monthly charges (taxes, common charges, maintainance, etc)
	additional_listing_info = soup.find('div', {'class' : 'vitals top_spacer'}).find_all('div', {'class' : 'details_info'})

	monthly_costs = 0
	for listing_details in additional_listing_info:
		if 'monthly_charges' in clean_string(listing_details.get_text()):
			for item in listing_details.stripped_strings:
				if '$' in item:
					monthly_costs += float(clean_number(item.split(':')[-1]))

	data['monthly_cost'] = str(monthly_costs)

	# parse building information
	building_info = soup.find('div', {'class' : 'in_this_building'})
	if building_info:
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


	# parse amenities list
	amenities = soup.find_all('div', {'class' : 'amenities'})
	if amenities:
		data['amenities_list'] = []
		for sections in amenities:
			for li in sections.find_all('li'):
				for item in li.stripped_strings:
					item = clean_string(item)
					if 'googletag' not in item:
						data['amenities_list'].append(item)
						break

	# parse nearby public transportation and education options
	nearby = soup.find('div', {'class' : 'nearby'})
	if nearby:
		data['transit_list'] = []
		transit = nearby.find('div', {'class' : 'transportation'})
		for p in transit.find_all('p'):
			cluster = ''
			for item in p.stripped_strings:
				if 'mile' in item or 'feet' in item or 'at' in item:
					cluster = cluster + '@' + clean_string(item)
				else:
					cluster = cluster + '%' + item if cluster else item
			data['transit_list'].append(cluster)

		data['school_district'] = ''
		schools = nearby.find('div', {'class' : 'schools'})
		for item in schools.stripped_strings:
			if 'district' in clean_string(item):
				district_number = clean_number(item.split()[-1])
				if district_number.isdigit():
					data['school_district'] = district_number


	# get gps coordinates
	map_half = soup.find('div', {'se:behavior' : 'mappable'})
	if map_half:
		data['gps_coordinates'] = map_half.get('se:map:point').encode('ascii', 'ignore').replace(',', ' ')

	# get listing text
	description_vector = []
	description = soup.find('blockquote', {'id' : re.compile('description_full_*')})
	if description:
		for item in description.stripped_strings:
			description_vector.append(item)
		data['description'] = ' '.join(description_vector).encode('ascii', 'ignore').lower().replace('[less]', '').translate(PUNCTUATION_TO_SPACE).strip()

	return True


def parse_listing_file(input_file, data):
	print 'processing file %s' % input_file

	try:
		with gzip.open(input_file, 'r') as fin:
			r = fin.read()
			return parse_listings_html(r, data)
	except:
		return False


def parse_listing_url(url, data):
	print 'processing url %s' % url

	try:
		r = urllib2.urlopen(url, timeout=5).read()
	except:
		return False

	with gzip.open(url.split('/')[-1] + '.html.gz', 'w') as fout:
		fout.write(r)

	return parse_listings_html(r, data)


def process_listing_files(args):
	output_path = args[0]
	input_files = args[1]

	with gzip.open(output_path, 'w') as fout:
		fout.write(','.join(FIELDS) + '\n')
		for input_file in input_files:
			saleid = input_file.split('/')[-1].split('.')[0].split('-')[-1]
			sale_url = 'http://streeteasy.com/sale/' + saleid
			listing_data = {}
			listing_data['url'] = sale_url
			listing_data['saleid'] = saleid
			# success = parse_listing_url(sale_url, listing_data)
			try:
				success = parse_listing_file(input_file, listing_data)
			except:
				print '{} : failed to parse'.format(saleid)
				success = False
			if not success: continue
			for field in FIELDS:
				if field in listing_data and type(listing_data[field]) == list:
					fout.write(' '.join(listing_data[field]))
				elif field in listing_data:
					fout.write(listing_data[field])
				if field != FIELDS[-1]:
					fout.write(',')
				else:
					fout.write('\n')

# parse_listing_url('http://streeteasy.com/sale/1214907', {})
# assert False

num_chunks = 4
file_chunks = []
for idx in range(num_chunks):
	file_chunks.append([])

# set html dir to be directory containing all downloaded HTML files
# this code assumes that the html files are gzipped
chunk_index = 0
num_files = 0
num_files_max = 9999999
html_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/html_files')
for f in os.listdir(html_dir):
	if f.endswith('.html.gz'):
		file_chunks[chunk_index].append(os.path.join(html_dir, f))
		chunk_index = (chunk_index + 1) % num_chunks
		num_files = num_files + 1
		if num_files >= num_files_max:
			break

p = Pool(num_chunks)
p.map(process_listing_files, [('sales_listings.1.csv.gz', file_chunks[0]),
							  ('sales_listings.2.csv.gz', file_chunks[1]),
							  ('sales_listings.3.csv.gz', file_chunks[2]),
							  ('sales_listings.4.csv.gz', file_chunks[3])])

print 'processed %d gzipped html files' % num_files
