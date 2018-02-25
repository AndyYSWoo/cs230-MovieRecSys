import pickle
class MetadataExtractor(object):
   def load_genre_json(self):
       fp = open('../data/movieGenre', 'rb')
       genre_string_list = pickle.load(fp)
       print genre_string_list

   def load_keyword_json(self):
       fp = open('../data/movieKeyword', 'rb')
       keyword_string_list = pickle.load(fp)
       print keyword_string_list

   def load_overview_json(self):
       fp = open('../data/overview.pkl', 'rb')
       overview_string_list = pickle.load(fp)
       print overview_string_list

if __name__=='__main__':
    extractor = MetadataExtractor()
    extractor.load_genre_json()
    extractor.load_keyword_json()
    extractor.load_overview_json()