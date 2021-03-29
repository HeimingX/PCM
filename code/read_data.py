import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
import pickle
import utils
import collections
from random import random, randrange, randint, shuffle, choice
from ipdb import set_trace as breakpoint
import global_file

logger = utils.logger

TOKENS_SHOT = [
    [734, 449, 430, 341, 312, 306, 229, 213, 205, 190, 137, 137, 106, 105, 103, 101, 95, 95, 87, 80, 80, 74, 66, 64, 63, 57, 54, 53, 52, 51, 47, 46, 46, 45, 43, 42, 41, 39, 39, 37, 35, 34, 34, 29, 29, 28, 26, 26, 26, 26, 25, 25, 25],
    [360, 356, 263, 258, 258, 217, 198, 190, 181, 139, 138, 136, 136, 131, 131, 129, 128, 126, 121, 120, 120, 116, 113, 107, 101, 100, 98, 98, 94, 93, 81, 80, 80, 73, 73, 70, 70, 69, 68, 68, 67, 65, 65, 61, 60, 59, 58, 57, 54, 54, 54, 53, 53, 52, 52, 51, 51, 51, 50, 50, 49, 48, 47, 46, 46, 46, 45, 44, 44, 43, 43, 43],
    [340, 157, 143, 109, 105, 95, 87, 81, 80, 66, 66, 64, 56, 55, 51, 51, 49, 49, 49, 46, 42, 41, 41, 38, 37, 35, 35, 33, 33, 33, 32, 32, 31, 31, 29, 29, 28, 28, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 20, 19, 19],
    [151, 127, 126, 109, 103, 80, 80, 79, 52, 48, 48, 44, 43, 42, 41, 40, 40, 38, 38, 35, 35, 35, 33, 33, 33, 32, 31, 31, 30, 30, 30, 28, 28, 28, 27, 26, 25, 25, 24, 24, 24, 24, 22, 22, 22, 22, 21, 18],
    [1287, 409, 393, 377, 366, 362, 339, 336, 332, 307, 286, 284, 279, 261, 244, 239, 230, 223, 222, 219, 217, 213, 206, 202, 197, 191, 182, 180, 178, 174, 172, 164, 160, 160, 154, 147, 145, 142, 140, 129, 129, 129, 128, 126, 124, 123, 122, 122, 120, 118, 118, 117, 117, 116, 115, 115, 115, 114, 112, 107, 106, 105, 104, 102, 101],
    [366, 275, 215, 195, 183, 166, 165, 156, 155, 135, 131, 118, 117, 117, 113, 110, 109, 103, 97, 96, 90, 75, 69, 63, 62, 60, 57, 53, 52, 52, 51, 51, 50, 50, 49, 46, 42, 41, 36, 34, 31, 30, 30, 29, 28, 28, 28, 28, 27, 26, 25, 25, 25, 25, 23, 22, 22, 22],
    [226, 116, 112, 107, 100, 77, 70, 65, 63, 60, 55, 54, 52, 50, 49, 44, 42, 42, 42, 38, 37, 34, 31, 31, 31, 31, 30, 28, 27, 26, 26, 26, 25, 25, 25, 24, 24, 24, 24, 22, 21, 21, 20, 20, 19, 19, 19, 18, 18, 17, 17, 17, 17, 17, 16, 16, 15, 15, 14, 14, 13, 12, 12, 12, 11, 10, 10, 10, 10, 10, 10],
    [776, 435, 374, 199, 185, 169, 134, 127, 100, 94, 87, 85, 80, 78, 67, 67, 66, 64, 61, 57, 57, 57, 57, 55, 53, 49, 48, 45, 45, 44, 42, 41, 38, 36, 36, 36, 36, 36, 33, 31, 31, 30, 29, 29, 28, 28, 27, 26, 25, 25, 25, 24, 24, 24, 24, 23, 22, 21, 20, 20, 20],
    [880, 405, 325, 308, 299, 191, 154, 148, 145, 134, 105, 84, 80, 79, 70, 64, 54, 41, 40, 38, 37, 36, 34, 32, 31, 29, 29, 29, 27, 27, 26, 25, 25, 24, 24, 24, 22, 21, 20, 19, 18, 18, 18, 17, 17, 17, 16, 15, 14],
    [386, 252, 203, 179, 154, 147, 139, 131, 127, 122, 103, 102, 101, 101, 99, 96, 88, 85, 84, 81, 80, 75, 72, 69, 62, 60, 58, 58, 58, 56, 55, 55, 55, 54, 53, 53, 52, 52, 52, 50, 48, 46, 46, 45, 44, 44, 42, 42, 42, 41, 41, 41, 38, 37, 37, 37, 35, 35, 34, 32, 32, 32, 32, 31, 31, 30, 30, 29, 28, 28, 28, 27, 26, 25]
]


def get_class_names(data_path):
    if 'yahoo_answers' in data_path:
        class_names = [['society', 'culture'],
                       ['science', 'mathematics'],
                       ['health'],
                       ['education', 'reference'],
                       ['computers', 'internet'],
                       ['sports'],
                       ['business', 'finance'],
                       ['entertainment', 'music'],
                       ['family', 'relationships'],
                       ['politics', 'government']]
        # YAHOOANSWERS_SIM_NAMES = [['community', 'civilization', 'customs'],
        #                           ['discipline', 'knowledge', 'technique'],
        #                           ['fitness', 'wellbeing', 'vigor'],
        #                           ['schooling', 'discipline', 'recommendation'],
        #                           ['laptop', 'website', 'network'],
        #                           ['game', 'exercise', 'play'],
        #                           ['trade', 'commerce', 'economy'],
        #                           ['fun', 'songs', 'folk'],
        #                           ['household', 'children', 'parents'],
        #                           ['civics', 'authority', 'administration']]
        #
        # CLS_WISE_ATTENTED_TOKENS = [
        #     ['god', 'bible', 'religion', 'gay', 'church', 'jesus', 'religious', 'life', 'christian', 'christians',
        #      'faith',
        #      'islam', 'muslims', 'christ', 'heaven', 'muslim', 'religions', 'christianity', 'holy', 'catholic', 'lord',
        #      'pray',
        #      'spiritual', 'prophet', 'jewish', 'worship', 'atheist', 'soul', 'spirit', 'hell', 'jews', 'mormon',
        #      'devil',
        #      'testament', 'lesbian', 'belief', 'live', 'quran', 'black', 'allah', 'churches', 'judaism', 'gospel',
        #      'islamic',
        #      'prayer', 'meditation', 'homosexual', 'sexual', 'satan', 'pastor', 'adam', 'priest', 'hebrew'],
        #     ['2', '-', 'water', '(', ')', '=', 'question', 'number', '3', 'x', 'energy', 'equation', 'speed', 'science',
        #      'gas',
        #      'mass', 'reaction', '+', 'answer', '1', 'math', 'earth', 'power', 'force', 'numbers', 'moon', 'sun',
        #      'pressure',
        #      'function', 'physics', 'theory', 'formula', '4', '/', 'universe', '0', 'star', 'oil', '5', 'heat', 'air',
        #      'acid',
        #      'two', 'color', 'distance', 'factor', 'metal', 'temperature', 'chemistry', 'element', 'planet', 'weather',
        #      'velocity', '7', '&', 'god', 'stars', 'evolution', '6', 'value', 'density', 'fire', 'ball', 'area',
        #      'motion',
        #      'cell', 'oxygen', 'chemical', 'species', 'surface', 'gravity', 'system'],
        #     ['sex', 'weight', 'body', 'hair', 'heart', 'eye', 'blood', 'penis', 'tooth', 'brain', 'teeth', 'breast',
        #      'drug',
        #      'diet', 'cancer', 'skin', 'stomach', 'pain', 'back', 'smoking', 'eyes', 'medicine', 'face', 'fat',
        #      'running',
        #      'mouth', 'muscle', 'smoke', 'food', 'birth', 'pregnant', 'ear', 'life', 'sleep', 'hiv', 'drugs', 'aids',
        #      'nose',
        #      'disease', 'foot', 'alcohol', 'dental', 'eating', 'leg', 'pill', 'dentist', 'shoulder', 'knee', 'exercise',
        #      'diabetes', 'drinking', 'chest', 'run', 'doctor', 'feet', 'autism'],
        #     ['language', 'college', 'english', 'university', 'spanish', 'law', 'school', 'medical', 'degree',
        #      'computer',
        #      'teaching', 'spelling', 'math', 'mba', 'business', 'writing', 'job', 'spell', 'languages', 'science',
        #      'french',
        #      'japanese', 'colleges', 'engineering', 'universities', 'accent', 'write', 'latin', 'nursing', 'study',
        #      'teacher',
        #      'student', 'read', 'speak', 'practice', 'speech', 'physics', 'history', 'studying', 'students', 'book',
        #      'teach',
        #      'class', 'nurse', 'radio', 'teachers', 'chemistry', 'course'],
        #     ['computer', 'download', 'software', 'drive', 'free', 'pc', 'mail', 'internet', 'windows', 'file', 'video',
        #      'laptop', 'yahoo', 'program', 'website', 'email', 'system', 'myspace', '##ware', 'page', 'code', 'virus',
        #      'cd',
        #      'files', 'search', 'address', 'music', 'dvd', 'messenger', 'com', 'network', 'password', 'account', 'game',
        #      'player', 'screen', 'disk', 'connection', 'run', 'wireless', 'computers', 'web', 'version', 'server',
        #      'games',
        #      'sound', 'play', 'format', 'data', 'phone', 'memory', 'history', 'sites', 'open', 'text', 'copy',
        #      'install',
        #      'message', 'xp', 'mp3', 'printer', 'ram', 'mac', 'speed', 'link'],
        #     ['football', 'cup', 'team', 'baseball', 'player', 'game', 'soccer', 'nfl', 'basketball', 'nba', 'wwe',
        #      'golf',
        #      'win', 'fight', 'league', 'ball', 'match', 'wrestling', 'cricket', 'players', 'tennis', 'race', 'nascar',
        #      'play',
        #      'hockey', 'games', 'coach', 'teams', 'fifa', 'boxing', 'bowl', 'goal', 'sports', 'season', 'wrestler',
        #      'mlb',
        #      'nhl', 'sport', 'championship', 'racing', 'skate', 'wrestlers', 'running', 'champions', 'bike', 'clubs',
        #      'strike',
        #      'shot', 'run', 'champion', 'fighting', 'horse', 'softball', 'ufc', 'tournament', 'volleyball', 'tour',
        #      'tna'],
        #     ['credit', 'bank', 'tax', 'job', 'company', 'work', 'business', 'card', 'money', 'loan', 'mortgage',
        #      'taxes',
        #      'market', 'stock', 'gas', 'currency', 'insurance', 'rate', 'debt', 'banks', 'jobs', 'banking', 'product',
        #      'lend',
        #      'account', 'buy', 'prices', 'loans', 'management', 'service', 'rates', 'cards', 'trading', 'mart',
        #      'grants',
        #      'marketing', 'sell', 'bill', 'oil', 'finance', 'trade', 'income', 'bills', 'financial', 'invest', 'price',
        #      'sales',
        #      'people', '$', 'pay', 'stocks', 'financing', 'property', 'customer', 'store', 'accounting', 'investing',
        #      'security', 'bankruptcy', 'markets', 'funds', 'investment', 'payment', 'deposit', 'dollar', 'fund',
        #      'salary',
        #      'fraud', 'coins', 'ticket', 'purchase'],
        #     ['song', 'music', 'movie', 'show', 'songs', 'band', 'movies', 'lyrics', 'download', 'video', 'watch', 'rap',
        #      'guitar', 'film', 'episode', 'cd', 'tv', 'sings', 'play', 'sing', 'singer', 'idol', 'group', 'series',
        #      'dvd',
        #      'rock', 'played', 'bands', 'season', 'album', 'anime', 'chris', 'shows', 'watching', 'episodes', 'game',
        #      'artist',
        #      'films', 'jackson', 'singing', 'radio', 'cartoon', 'metal', 'story', 'michael', 'playing', 'rapper',
        #      'concert',
        #      'sang', 'fans', 'britney', 'fan', 'star', 'character', 'party', 'videos', 'comic', 'dance', 'mp3', 'piano',
        #      'manga'],
        #     ['love', 'boyfriend', 'friend', 'sex', 'relationship', 'friends', 'dating', 'husband', 'married', 'date',
        #      'ex',
        #      'wife', 'marriage', 'girlfriend', 'bf', 'kiss', 'cheating', 'feelings', 'friendship', 'marry', 'single',
        #      'dad',
        #      'family', 'father', 'relationships', 'dated', 'sister', 'mom', 'cheated', 'divorce', 'brother', 'mother',
        #      'daughter', 'parents', 'boss', 'partner', 'kids', 'lesbian', 'baby', 'flirt', 'wedding', 'virgin',
        #      'loving',
        #      'flirting', 'miss', 'sexual', 'children', 'home', 'child'],
        #     ['war', 'court', 'iraq', 'military', 'bush', 'country', 'immigration', 'police', 'law', 'president', 'usa',
        #      'government', 'army', 'illegal', 'iran', 'state', 'legal', 'mexican', 'americans', 'immigrants', 'judge',
        #      'prison',
        #      'america', 'crime', 'israel', 'fight', 'jail', 'civil', 'service', 'united', 'troops', 'nuclear',
        #      'soldiers',
        #      'visa', 'english', 'american', 'criminal', 'terrorists', 'abortion', 'george', 'liberal', 'liberals',
        #      'clinton',
        #      'terrorism', 'vietnam', 'laws', 'mexico', 'states', 'tax', 'terrorist', 'marines', 'religious', 'fire',
        #      'racism',
        #      'leader', 'violence', 'guilty', 'jews', 'iraqi', 'crimes', 'nation', 'jury', 'navy', 'fighting', 'uk',
        #      'political',
        #      'racist', 'republicans', 'countries', 'jewish', 'muslims', 'weapons', 'fbi', 'immigrant']]
    elif 'ag_news' in data_path:
        class_names = [['world'],
                       ['sports'],
                       ['business'],
                       ['science', 'technology']]
    elif 'imdb' in data_path:
        class_names = [['negative', 'bad'], ['positive', 'good']]
    elif 'dbpedia' in data_path:
        class_names = [['company'],
                       ['educational', 'institution'],
                       ['artist'],
                       ['athlete'],
                       ['office', 'holder'],
                       ['transportation'],
                       ['building'],
                       ['natural', 'place'],
                       ['village'],
                       ['animal'],
                       ['plant'],
                       ['album'],
                       ['film'],
                       ['written', 'work']]
    else:
        raise LookupError
    return class_names


class Translator:
    """Backtranslation. Here to save time, we pre-processing and save all the translated data into pickle files.
    """

    def __init__(self, path, transform_type='BackTranslation'):
        # Pre-processed German data
        with open(path + 'de_1.pkl', 'rb') as f:
            self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + 'ru_1.pkl', 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        out1 = self.de[idx]
        out2 = self.ru[idx]
        return out1, out2, ori


def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased',
             train_aug=False, add_cls_sep=False, mixText_origin=True, splited=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(
        './cache/pytorch_transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084')

    test_df = pd.read_csv(data_path + 'test.csv', header=None)
    test_labels = np.array([u - 1 for u in test_df[0]])
    test_text = np.array([v for v in test_df[2]])
    n_labels = max(test_labels) + 1

    if splited:
        if global_file.args.seed_l != 0:
            _path = data_path + 'labeled_data_nlab' + str(n_labeled_per_class)
            _path += '_seed' + str(global_file.args.seed_l)
            _path += '.csv'
        else:
            _path = data_path + 'labeled_data_nlab' + str(n_labeled_per_class) + '.csv'
        logger.info('load labeled data from: {}'.format(_path))
        l_df = pd.read_csv(_path, header=None)
        ul_df = pd.read_csv(data_path + 'ul_data_nlab' + str(n_labeled_per_class) + '.csv', header=None)
        val_df = pd.read_csv(data_path + 'val_data_nlab' + str(n_labeled_per_class) + '.csv', header=None)

        labeled_text, labeled_l = np.array([v for v in l_df[0]]), np.array([v for v in l_df[1]])
        ul_text, ul_l, train_unlabeled_idxs = np.array([v for v in ul_df[0]]), np.array(
            [v for v in ul_df[1]]), np.array([v for v in ul_df[2]])
        val_text, val_l = np.array([v for v in val_df[0]]), np.array([v for v in val_df[1]])
    else:
        train_df = pd.read_csv(data_path + 'train.csv', header=None)
        # Here we only use the bodies and removed titles to do the classifications
        train_labels = np.array([v - 1 for v in train_df[0]])
        train_text = np.array([v for v in train_df[2]])

        # Split the labeled training set, unlabeled training set, development set
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
            train_labels, n_labeled_per_class, unlabeled_per_class, n_labels)

        labeled_text, labeled_l = train_text[train_labeled_idxs], train_labels[train_labeled_idxs]
        ul_text, ul_l = train_text[train_unlabeled_idxs], train_labels[train_unlabeled_idxs]
        val_text, val_l = train_text[val_idxs], train_labels[val_idxs]

        # save data
        # l_data = pd.DataFrame({'text': labeled_text, 'label': labeled_l})
        # ul_data = pd.DataFrame({'text': ul_text, 'label': ul_l, 'idxs': train_unlabeled_idxs})
        # val_data = pd.DataFrame({'text': val_text, 'label': val_l})
        # if global_file.args.seed_l != 0:
        #     _path = data_path + 'labeled_data_nlab' + str(n_labeled_per_class)
        #     _path += '_seed' + str(global_file.args.seed_l)
        #     _path += '.csv'
        #     l_data.to_csv(_path, index=False, header=False)
        # else:
        #     l_data.to_csv(data_path + 'labeled_data_nlab' + str(n_labeled_per_class) + '.csv', index=False,
        #                   header=False)
        #     ul_data.to_csv(data_path + 'ul_data_nlab' + str(n_labeled_per_class) + '.csv', index=False, header=False)
        #     val_data.to_csv(data_path + 'val_data_nlab' + str(n_labeled_per_class) + '.csv', index=False, header=False)
        # l_df = pd.read_csv(data_path + 'labeled_data_nlab' + str(n_labeled_per_class) + '.csv', header=None)
        # ul_df = pd.read_csv(data_path + 'ul_data_nlab' + str(n_labeled_per_class) + '.csv', header=None)
        # val_df = pd.read_csv(data_path + 'val_data_nlab' + str(n_labeled_per_class) + '.csv', header=None)
        # labeled_text1, labeled_l1 = np.array([v for v in l_df[0]]), np.array([v for v in l_df[1]])
        # ul_text1, ul_l1, train_unlabeled_idxs1 = np.array([v for v in ul_df[0]]), np.array(
        #     [v for v in ul_df[1]]), np.array([v for v in ul_df[2]])
        # val_text1, val_l1 = np.array([v for v in val_df[0]]), np.array([v for v in val_df[1]])
        # breakpoint()

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(labeled_text, labeled_l, tokenizer, max_seq_len, train_aug,
                                           add_cls_sep=add_cls_sep, mixText_origin=mixText_origin)
    train_unlabeled_dataset = loader_unlabeled(ul_text, ul_l, train_unlabeled_idxs, tokenizer, max_seq_len,
                                               Translator(data_path), add_cls_sep=add_cls_sep)
    val_dataset = loader_labeled(val_text, val_l, tokenizer, max_seq_len, add_cls_sep=add_cls_sep,
                                 mixText_origin=mixText_origin)
    test_dataset = loader_labeled(test_text, test_labels, tokenizer, max_seq_len, add_cls_sep=add_cls_sep,
                                  mixText_origin=mixText_origin)

    logger.info(
        "#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(labeled_l), len(ul_l), len(val_l), len(test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels, tokenizer


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0):
    """Split the original training set into labeled training set, unlabeled training set, development set

    Arguments:
        labels {list} -- List of labeles for original training set
        n_labeled_per_class {int} -- Number of labeled data per class
        unlabeled_per_class {int} -- Number of unlabeled data per class
        n_labels {int} -- The number of classes

    Keyword Arguments:
        seed {int} -- [random seed of np.shuffle] (default: {0})

    Returns:
        [list] -- idx for labeled training set, unlabeled training set, development set
    """
    # np.random.seed(seed)
    seed = global_file.args.seed_l
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    def pseudo_shuffle(seed_l, train_pool):
        """In order to re-use the generated back-translations of ul data which is really time-consuming,
        we use this pseudo-shuffle code to test the performance of various labeled data"""
        if seed_l == 1:
            return train_pool[::-1]
        elif seed_l == 2:
            pool_size = len(train_pool)
            return np.concatenate((train_pool[pool_size//2:], train_pool[:pool_size//2]))
        else:
            return train_pool

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if n_labels == 2:
            # IMDB
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            # train_pool = pseudo_shuffle(global_file.args.seed_l, train_pool)
            train_labeled_idxs.extend(train_pool[n_labeled_per_class*seed:n_labeled_per_class*(seed+1)])
            train_unlabeled_idxs.extend(idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
        elif n_labels == 10:
            # DBPedia
            train_pool = np.concatenate((idxs[:500], idxs[10500:-2000]))
            # train_pool = pseudo_shuffle(global_file.args.seed_l, train_pool)
            train_labeled_idxs.extend(train_pool[n_labeled_per_class*seed:n_labeled_per_class*(seed+1)])
            train_unlabeled_idxs.extend(idxs[500: 500 + unlabeled_per_class])
            val_idxs.extend(idxs[-2000:])
        else:
            # Yahoo/AG News
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            # train_pool = pseudo_shuffle(global_file.args.seed_l, train_pool)
            train_labeled_idxs.extend(train_pool[n_labeled_per_class*seed:n_labeled_per_class*(seed+1)])
            train_unlabeled_idxs.extend(idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False, add_cls_sep=False, mixText_origin=True):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.add_cls_sep = add_cls_sep
        self.mixText_origin = mixText_origin
        self.trans_dist = {}

        self.train_tokens = list()

        if aug:
            logger.info('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def update_token_list(self):
        self.train_tokens = list()

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            cls_sep_token_len = 0
            if self.add_cls_sep:
                cls_sep_token_len = 2
            max_seq_len = self.max_seq_len - cls_sep_token_len
            if len(tokens) > max_seq_len:
                if 'imdb' in global_file.args.data_path or 'yelp' in global_file.args.data_path:
                    tokens = tokens[-max_seq_len:]  # text[-1024:]
                else:
                    tokens = tokens[:max_seq_len]

            segment_ids = [0 for _ in range(len(tokens))]
            if self.add_cls_sep:
                segment_ids = [0 for _ in range(len(tokens) + 2)]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
            length = len(tokens)
            self.train_tokens.append(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            mask_array = np.zeros(self.max_seq_len, dtype=np.bool)
            mask_array[:length] = 1
            segment_array = np.zeros(self.max_seq_len, dtype=np.bool)
            segment_array[:len(segment_ids)] = segment_ids
            if self.mixText_origin:
                return torch.tensor(encode_result), self.labels[idx], length, idx
            else:
                return torch.tensor(encode_result), self.labels[idx], length, torch.tensor(
                    mask_array.astype(np.int64)), torch.tensor(segment_array.astype(np.int64)), idx


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, dataset_label, unlabeled_idxs, tokenizer, max_seq_len, aug=None,
                 add_cls_sep=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len
        self.add_cls_sep = add_cls_sep

        self.train_tokens = list()

    def update_token_list(self):
        self.train_tokens = list()

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text, origin_flag=False):
        tokens = self.tokenizer.tokenize(text)
        if self.add_cls_sep:
            cls_sep_token_len = 2
        else:
            cls_sep_token_len = 0
        max_seq_len = self.max_seq_len - cls_sep_token_len
        if len(tokens) > max_seq_len:
            if 'imdb' in global_file.args.data_path or 'yelp' in global_file.args.data_path:
                tokens = tokens[-max_seq_len:]  # text[-1024:]
            else:
                tokens = tokens[:max_seq_len]
        if self.add_cls_sep:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        length = len(tokens)
        if origin_flag:
            self.train_tokens.append(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        text = self.text[idx]
        # if 'imdb' in global_file.args.data_path:
        #     text = text[-1024:]
        if self.aug is not None:
            u, v, ori = self.aug(text, self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori, origin_flag=True)
            return (torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), \
                   (length_u, length_v, length_ori), idx, self.labels[idx]
        else:
            encode_result, length = self.get_tokenized(text, origin_flag=True)
            return torch.tensor(encode_result), length, idx, self.labels[idx]


# generate MLM training data
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list,
                                 attention=None):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if whole_word_mask and len(cand_indices) >= 1 and token.startswith("##"):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    if attention is None:
        shuffle(cand_indices)
    else:
        cand_indices = torch.sort(attention, descending=True)[1].unsqueeze(1)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


class loader_mlm(Dataset):
    # Data loader for MLM
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, attention, add_cls_sep=False,
                 masked_lm_prob=0.15, max_predictions_per_seq=20, do_whole_word_mask=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        self.attention = attention
        self.vocab_list = list(tokenizer.vocab.keys())

        self.add_cls_sep = add_cls_sep
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.do_whole_word_mask = do_whole_word_mask

    def update_attention(self, new_attention):
        self.attention = new_attention

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text[idx]
        tokens = self.tokenizer.tokenize(text)
        cls_sep_token_len = 2 if self.add_cls_sep else 0
        max_seq_len = self.max_seq_len - cls_sep_token_len
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        if self.add_cls_sep:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            # tokens = ["CLS"] + tokens + ["SEP"]

        if self.attention is not None:
            if self.attention[idx].max() > 0:
                attention = self.attention[idx][:len(tokens)]
            else:
                attention = None
        else:
            attention = self.attention
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, self.masked_lm_prob,
                                                                                     self.max_predictions_per_seq,
                                                                                     self.do_whole_word_mask,
                                                                                     self.vocab_list, attention)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        mask_array = np.zeros(self.max_seq_len, dtype=np.bool)
        mask_array[:len(encode_result)] = 1

        masked_label_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        lm_label_array = np.full(self.max_seq_len, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        return torch.tensor(encode_result), torch.tensor(mask_array.astype(np.int64)), \
               torch.tensor(lm_label_array.astype(np.int64)), idx
