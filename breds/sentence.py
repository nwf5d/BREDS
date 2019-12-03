import re
from nltk import word_tokenize
from nltk.corpus import stopwords

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

# tokens between entities which do not represent relationships
bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words('english')
not_valid = bad_tokens + stopwords


def tokenize_entity(entity):
    parts = word_tokenize(entity)
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    locations = []
    e_parts = tokenize_entity(entity_string)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return e_parts, locations


class EntitySimple:
    def __init__(self, _e_string, _e_parts, _e_type, _locations):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations

    def __hash__(self):
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other):
        return self.string == other.string and self.type == other.type


class EntityLinked:
    def __init__(self, _e_string, _e_parts, _e_type, _locations, _url=None):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations
        self.url = _url

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return self.url == other.url


class Relationship:
    def __init__(self, _sentence, _before, _between, _after, _ent1, _ent2,
                 e1_type, e2_type):
        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.e1 = _ent1
        self.e2 = _ent2
        self.e1_type = e1_type
        self.e2_type = e2_type

    def __eq__(self, other):
        if self.e1 == other.e1 and self.e2 == other.e2 and \
                self.before == other.before and self.between == other.between \
                and self.after == other.after:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2) ^ hash(self.before) ^ \
               hash(self.between) ^ hash(self.after)


class Sentence:

    def __init__(self, sentence, e1_type, e2_type, max_tokens, min_tokens,
                 window_size, pos_tagger=None, config=None):
        self.relationships = list()
        self.tagged_text = None
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.window_size = window_size
        self.pos_tagger = pos_tagger
        self.config = config
        self.e1_type = e1_type
        self.e2_type = e2_type

        # determine which type of regex to use according to
        # how named-entities are tagged
        entities_regexs = {
            "simple": config.regex_simple,
            "linked": config.regex_linked
            }
        clean_regexs = {
            "simple": config.regex_clean_simple,
            "clean": config.regex_clean_linked
            }
        entities_regex = entities_regexs.get(config.tag_type)
        clean_regex = clean_regexs.get(config.tag_type)

        # find named-entities
        entities = []
        for m in re.finditer(entities_regex, sentence):
            entities.append(m)

        if len(entities) < 2: return

        # clean tags from text
        sentence_no_tags = re.sub(clean_regex, "", sentence)
        text_tokens = word_tokenize(sentence_no_tags)

        locations = self.extract_entities(entities, text_tokens, config)
        self.find_pair_entities(locations, text_tokens)


    def extract_entities(self, entities, text_tokens, config):
        # extract information about the entity, create an Entity instance
        # and store in a structure to hold information collected about
        # all the entities in the sentence
        entities_info = set()
        type_regex = '<([A-Z]+)'
        url_regex = 'url=([^>]+)'
        string_regexs = {"simple": '<[A-Z]+>([^<]+)</[A-Z]+>',
                         "linked": '<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>'}

        string_regex = string_regexs.get(config.tag_type)
        for x in range(0, len(entities)):
            entity = entities[x].group()
            e_type = re.findall(type_regex, entity)[0]
            e_string = re.findall(string_regex, entity)[0]
            e_parts, locations = find_locations(e_string, text_tokens)

            if config.tag_type == "simple":
                e = EntitySimple(e_string, e_parts, e_type, locations)
                entities_info.add(e)
            elif config.tag_type == "linked":
                e_url = re.findall(url_regex, entity)[0]
                e = EntityLinked(e_string, e_parts, e_type, locations, e_url)
                entities_info.add(e)

        # create an hash table:
        # - key is the starting index in the tokenized sentence of an entity
        # - value the corresponding Entity instance
        locations = dict()
        for e in entities_info:
            for start in e.locations:
                locations[start] = e
        return locations

    def find_pair_entities(self, locations, text_tokens):
        # look for pair of entities such that:
        # the distance between the two entities is less than 'max_tokens'
        # and greater than 'min_tokens'
        # the arguments match the seeds semantic types
        sorted_keys = list(sorted(locations))
        for i in range(len(sorted_keys)-1):
            e1_index = sorted_keys[i]
            e2_index = sorted_keys[i+1]
            distance = e2_index - e1_index
            e1 = locations[e1_index]
            e2 = locations[e2_index]
            if self.max_tokens < distance or distance < self.min_tokens or e1.type != self.e1_type \
                        or e2.type != self.e2_type:
                continue

            # ignore relationships between the same entity
            if self.config.tag_type == "simple" and e1.string == e2.string:
                continue
            elif self.config.tag_type == "linked" and e1.url == e2.url:
                continue

            # run PoS-tagger over the sentence only once
            if self.tagged_text is None:
                # split text into tokens and tag them using NLTK's
                # default English tagger
                # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/
                # english.pickle'
                self.tagged_text = self.pos_tagger.tag(text_tokens)

            self.add_relationship(self.sentence, text_tokens, e1_index, e2_index, e1, e2)

    def add_relationship(self, sentence, text_tokens, e1_index, e2_index, e1, e2):
        before = self.tagged_text[:e1_index]
        before = before[-self.window_size:]
        between = self.tagged_text[e1_index + len(e1.parts):e2_index]
        after = self.tagged_text[e2_index+len(e2.parts):]
        after = after[:self.window_size]

        # ignore relationships where BET context is only stopwords
        # or other invalid words
        if all(x in not_valid for x in
               text_tokens[e1_index + len(e1.parts):e2_index]):
            return

        if self.config.tag_type == "simple":
            r = Relationship(
                sentence, before, between, after, e1.string,
                e2.string, e1.type, e2.type
            )
            self.relationships.append(r)

        elif self.config.tag_type == "linked":
            r = Relationship(
                sentence, before, between, after, e1.url, e2.url,
                e1.type, e2.type
            )
            self.relationships.append(r)

