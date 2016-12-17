import geojson
import os


def clean_string(string):
    return '_'.join(string.encode('ascii', 'ignore').lower().replace('-', ' ').replace('/', ' ').replace(',', ' ').strip().split())


def generate_geojson_for_subregion_with_model_evaluations(subregion_name, geojson_file_name, geojson_property_name):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'modeling')

    with open(os.path.join(data_dir, '%s.geojson' % geojson_file_name), 'r') as geojson_obj:
        geojson_obj = geojson.load(geojson_obj)
        for feature in geojson_obj['features']:
            subregion = clean_string(str(feature['properties'][geojson_property_name]))
            subregion_model_scores_file = os.path.join(model_dir, 'model_output/enet/submodels/enet_%s_%s_test_scores.txt' % (subregion_name, subregion))
            if os.path.exists(subregion_model_scores_file):
                with open(subregion_model_scores_file) as file_obj:
                    for line in file_obj:
                        tokens = [x.strip() for x in line.strip().split('=')]
                        if len(tokens) == 2:
                            feature['properties'].update({tokens[0] : float(tokens[1])})
            else:
                print 'cannot find submodel model accuracy file for %s_%s at %s' % (subregion_name, subregion, subregion_model_scores_file)

    with open(os.path.join(data_dir, '%s_with_enet_model_evaluations.geojson' % geojson_file_name), 'w') as outfile:
        geojson.dump(geojson_obj, outfile, sort_keys=True, indent=4)


generate_geojson_for_subregion_with_model_evaluations(
    subregion_name='borough', geojson_file_name='nyc_boroughs', geojson_property_name='borough')

generate_geojson_for_subregion_with_model_evaluations(
    subregion_name='community_district', geojson_file_name='nyc_communities', geojson_property_name='communityDistrict')

generate_geojson_for_subregion_with_model_evaluations(
    subregion_name='neighborhood', geojson_file_name='nyc_neighborhoods', geojson_property_name='neighborhood')