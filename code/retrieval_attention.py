import torch
import csv
import numpy as np
import json
import ipdb
import global_file

# copy from http://www.lextek.com/manuals/onix/stopwords1.html
stopping_words_list = [
    ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also',
     'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are',
     'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'am', 'answer', 'answers', 'al'],
    ['b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been',
     'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by'],
    ['c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'comes',
     'could'],
    ['d', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing',
     'downs', 'during'],
    ['e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every',
     'everybody', 'everyone', 'everything', 'everywhere', 'etc', 'et'],
    ['f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from',
     'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', '1st', 'friday'],
    ['g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'goes', 'going', 'good',
     'goods',
     'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups'],
    ['h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest',
     'him', 'himself', 'his', 'how', 'however', 'help'],
    ['i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its',
     'itself', 'im'],
    ['j', 'just'],
    ['k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows'],
    ['l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long',
     'longer', 'longest', 'look', 'looking'],
    ['m', 'made', 'make', 'making', 'man', 'many', 'may', 'maybe', 'me', 'member', 'members', 'men', 'might', 'more',
     'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'monday'],
    ['n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'no', 'nobody',
     'non', 'none', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'nor', 'name'],
    ['o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening',
     'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over'],
    ['p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing',
     'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts',
     'please'],
    ['q', 'quite', 'question', 'questions'],
    ['r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 'read', 'reuters'],
    ['s', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems',
     'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since',
     'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states',
     'still', 'still', 'such', 'sure', 'sometimes', 'start', 'saturday', 'sunday'],
    ['t', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they',
     'thing', 'things', 'think', 'thinks', 'talk', 'talks', 'this', 'those', 'though', 'thought', 'thoughts', 'three',
     'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two',
     'try', 'today', 'tuesday', 'thursday'],
    ['u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'using', 'usual', 'usually', 'unless'],
    ['v', 'very'],
    ['w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what',
     'whatever', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within',
     'without', 'work', 'worked', 'working', 'works', 'would', 'wednesday'],
    ['x'],
    ['y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'yes', 'yesterday'],
    ['z'],
    ['!', '"', '#', '$', '%', '&', '\\', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
     '@', '[', '\\\\', ']', '^', '_', '`', '{', '|', '}', '~', '“', '”', '’', '♥', '·', '—', '¡', 'χ']]

imdb_stopping_words_list = [
    ['actor', 'actors', 'people', 'acting'],
    ['boy', 'book'],
    ['character', 'characters', 'cast', 'comedy'],
    ['director', 'dvd', 'drama'],
    ['film', 'films', 'father', 'family'],
    ['guy', 'girl'],
    ['human'],
    ['life'],
    ['movie', 'movies', 'mather', 'music', 'money'],
    ['play', 'played', 'plays', 'performance', 'performances', 'plot'],
    ['story', 'scene', 'scenes', 'song', 'songs', 'seen', 'series', 'script'],
    ['time', 'tv'],
    ['role'],
    ['watch', 'watching']
]


def transform_shape(data, batchsize):
    new_data = list()
    for idx in range(len(data)):
        new_data.append(data[idx].unsqueeze(0).repeat(batchsize, 1))
    return new_data


def get_attention(model, input_data, length, require_pl=False, probing_words_list=None):
    with torch.no_grad():
        if probing_words_list is not None:
            # p, _, tmp_attention = model(input_data, length, output_attention=True, probing_words_list=probing_words_list)
            if global_file.args.cos_clsifer:
                p1, p2, tmp_attention = model(input_data, length, output_attention=True,
                                              cos_dist=global_file.args.cos_clsifer,
                                              probing_words_list=transform_shape(probing_words_list,
                                                                                 input_data.size()[0]))
            else:
                p1, p2, tmp_attention = model(input_data, length, output_attention=True,
                                              probing_words_list=transform_shape(probing_words_list,
                                                                                 input_data.size()[0]))
            tmp_attention = tmp_attention[-1].detach().cpu()
            tmp_attention = torch.mean(tmp_attention, 1)
            tmp_attention = torch.mean(tmp_attention, 1)
            if require_pl:
                return tmp_attention, (p1.cpu(), p2.cpu())
            else:
                return tmp_attention
        else:
            p, tmp_attention = model(input_data, length, output_attention=True)
            tmp_attention = tmp_attention[-1].detach().cpu()
            tmp_attention = torch.mean(tmp_attention, 1)
            tmp_attention = torch.mean(tmp_attention, 1)
            if require_pl:
                return tmp_attention, p.cpu()
            else:
                return tmp_attention


def get_attention_all(valloader, model, data_num, ul_flag=False, split_tf=False, probing_words_list=None, nlabel=10):
    model.eval()
    with torch.no_grad():
        if probing_words_list is not None:
            attention = torch.zeros((data_num, 256 + len(probing_words_list)))
        else:
            attention = torch.zeros((data_num, 256))
        pl = torch.zeros(data_num, nlabel)
        pl_prob = torch.zeros(data_num, nlabel)
        tf_flag = torch.zeros(data_num)
        if ul_flag:
            for batch_idx, ((_, _, inputs), (_, _, length_ori), idx, targets) in enumerate(valloader):
                cur_attention, cur_p = get_attention(model, inputs.cuda(), length_ori.cuda(), require_pl=True,
                                                     probing_words_list=probing_words_list)
                attention[idx] = cur_attention

                if probing_words_list is not None:
                    pl[idx] = cur_p[0]
                    pl_prob[idx] = cur_p[1]
                    cur_pl = cur_p[0].argmax(1).float()
                    tf_flag[idx] += (cur_pl[0] == targets).float()
                else:
                    pl[idx] = cur_p
                    cur_pl = cur_p.argmax(1).float()
                    tf_flag[idx] += (cur_pl == targets).float()
        else:
            for batch_idx, (inputs, targets, length, _, _, idx) in enumerate(valloader):
                cur_attention, cur_p = get_attention(model, inputs.cuda(), length.cuda(), require_pl=True,
                                                     probing_words_list=probing_words_list)
                attention[idx] = cur_attention
                if probing_words_list is not None:
                    pl[idx] = cur_p[0]
                    pl_prob[idx] = cur_p[1]
                    cur_pl = cur_p[0].argmax(1).float()
                    tf_flag[idx] += (cur_pl[0] == targets).float()
                else:
                    pl[idx] = cur_p
                    cur_pl = cur_p.argmax(1).float()
                    tf_flag[idx] += (cur_pl == targets).float()
    if probing_words_list is not None:
        pl_final = (pl, pl_prob)
    else:
        pl_final = pl
    if split_tf:
        true_flag = tf_flag == torch.ones(data_num)
        false_flag = tf_flag == torch.zeros(data_num)
        return attention, pl_final, true_flag, false_flag
    else:
        return attention, pl_final


def construct_attention_file(texts, tokens, attention, y_gt, y_pred, file_name, label_names):
    data_num = texts.shape[0]
    with open(file_name, 'w') as csvfile:
        row_name = ["ground truth", "predicted label", "text", "tokens", "attention"]
        writer = csv.DictWriter(csvfile, fieldnames=row_name)
        writer.writeheader()
        for data_idx in range(data_num):
            cur_pl = int(y_pred[data_idx].item())
            cur_pl_name = label_names[cur_pl][0]
            cur_gt_name = label_names[y_gt[data_idx]][0]
            cur_text = texts[data_idx]
            cur_token = tokens[data_idx]
            cur_attention = attention[data_idx][:len(cur_token)]
            cur_token_array = np.array(cur_token)
            cur_attention_descend, descend_idx = torch.sort(cur_attention, descending=True)
            cur_token_descend = cur_token_array[descend_idx]
            cur_token_descend = [tmp_token.encode('utf-8') for tmp_token in list(cur_token_descend)]
            try:
                writer.writerow({"ground truth": cur_gt_name,
                                 "predicted label": cur_pl_name,
                                 "text": cur_text,
                                 "tokens": cur_token_descend,
                                 "attention": list(cur_attention_descend.numpy())})
            except UnicodeEncodeError:
                print("encode error, skip data: {}".format(data_idx))


def satisfy_rule(prob, relu_type='uda'):
    T = global_file.args.T
    confid = global_file.args.confid

    if relu_type == 'uda':
        assert not isinstance(prob, tuple)
        orig_y_pred = prob / T
        targets_u = torch.softmax(orig_y_pred, dim=0)
        pl_u, pl_u_idx = torch.max(targets_u, 0)
        loss_u_mask = (pl_u >= confid).float()
        if loss_u_mask >= 1:
            return True
        else:
            return False
    else:
        assert isinstance(prob, tuple)
        prob, prob_match = prob

        orig_y_pred = prob / T
        targets_u = torch.softmax(orig_y_pred, dim=0)
        pl_u, pl_u_idx = torch.max(targets_u, 0)
        loss_u_mask = (pl_u >= confid).float()

        if global_file.args.cos_clsifer:
            prob_proba_orig = torch.clamp(prob_match, min=0.0)
        else:
            prob_proba_orig = torch.sigmoid(prob_match)
        _, pl_match_idx = torch.max(prob_proba_orig, 0)
        confid_prob = global_file.args.confid_prob
        prob_mask = ((prob_proba_orig > confid_prob).float().sum() == 1).float()

        same_mask = (pl_u_idx == pl_match_idx).float()

        if prob_mask * loss_u_mask * same_mask >= 1:
            return True
        else:
            return False


def extract_class_wise_attened_tokens(texts_set, tokens_set, attention_set, y_pred_set, num_class, epoch, savename,
                                      topk, relu_type='uda'):
    l_text, u_text = texts_set
    l_token, u_token = tokens_set
    l_attention, u_attention = attention_set
    l_pred, u_pred = y_pred_set
    if isinstance(u_pred, tuple):
        u_pred, u_pred_prob = u_pred
        multiple_pred = True
    else:
        multiple_pred = False

    cls_attended_tokens = dict()
    for idx in range(num_class):
        cls_attended_tokens[idx] = dict()

    all_attended_tokens = dict()

    l_num = l_text.shape[0]
    u_num = u_text.shape[0]

    topk /= 100.0
    for l_idx in range(l_num):
        # cur_pl = int(l_pred[l_idx].item())
        # if not satisfy_rule(l_pred[l_idx], relu_type):
        #     continue
        # else:
        cur_pl = l_pred[l_idx].item()
        cur_token = l_token[l_idx]
        cur_attention = l_attention[l_idx][:len(cur_token)]
        cur_topk = int(len(cur_token) * topk)
        topk_idx = torch.topk(cur_attention, cur_topk)[1]
        for top_idx in topk_idx:
            # top_token = cur_token[top_idx].encode('utf-8')
            top_token = cur_token[top_idx]
            if '#' in top_token or '[CLS]' in top_token or '[SEP]' in top_token or top_token.isdigit():
                continue
            else:
                if top_token in cls_attended_tokens[cur_pl].keys():
                    cls_attended_tokens[cur_pl][top_token] += 1
                else:
                    cls_attended_tokens[cur_pl][top_token] = 1

                if top_token in all_attended_tokens.keys():
                    all_attended_tokens[top_token].add(cur_pl)
                else:
                    all_attended_tokens[top_token] = set()
                    all_attended_tokens[top_token].add(cur_pl)

    for u_idx in range(u_num):
        # cur_pl = int(u_pred[u_idx].item())
        if not satisfy_rule((u_pred[u_idx], u_pred_prob[u_idx]) if multiple_pred else u_pred[u_idx], relu_type):
            continue
        else:
            cur_pl = u_pred[u_idx].argmax().item()
            cur_token = u_token[u_idx]
            cur_attention = u_attention[u_idx][:len(cur_token)]
            cur_topk = int(len(cur_token) * topk)
            topk_idx = torch.topk(cur_attention, cur_topk)[1]
            for top_idx in topk_idx:
                top_token = cur_token[top_idx]
                if '#' in top_token or '[CLS]' in top_token or '[SEP]' in top_token or top_token.isdigit():
                    continue
                else:
                    if top_token in cls_attended_tokens[cur_pl].keys():
                        cls_attended_tokens[cur_pl][top_token] += 1
                    else:
                        cls_attended_tokens[cur_pl][top_token] = 1

                    if top_token in all_attended_tokens.keys():
                        all_attended_tokens[top_token].add(cur_pl)
                    else:
                        all_attended_tokens[top_token] = set()
                        all_attended_tokens[top_token].add(cur_pl)

    # filter stopping words
    stopping_words = [x for j in stopping_words_list for x in j]
    if 'imdb' in savename:
        imdb_stopping_words = [x for j in imdb_stopping_words_list for x in j]
        stopping_words += imdb_stopping_words
    for cur_token in stopping_words:
        for idx in range(num_class):
            if cur_token in cls_attended_tokens[idx].keys():
                del cls_attended_tokens[idx][cur_token]

    cls_attended_tokens_list = list()
    for cls_id in range(num_class):
        cur_cls_tokens = np.array(list(cls_attended_tokens[cls_id].keys()))
        cur_cls_tokens_shot = np.array(list(cls_attended_tokens[cls_id].values()))
        sorted_idx = np.argsort(-1 * cur_cls_tokens_shot)
        cls_attended_tokens_list.append(list(cur_cls_tokens[sorted_idx]))
    # np.save(path + 'uda_top5_noStopWords_probWrods_ep' + str(epoch) + '.npy', cls_attended_tokens_list)
    # np.save(path + 'mixtext_top5_noStopWords_probWrods_ep' + str(epoch) + '.npy', cls_attended_tokens_list)
    # ipdb.set_trace()
    # np.save(path + 'dynamic_top5_noStopWords_probWrods_ep' + str(epoch) + '.npy', cls_attended_tokens_list)
    # np.save(path + 'uda_top5_noStopWords_probWrods_ep' + str(epoch) + '.npy', cls_attended_tokens_list)
    np.save(savename, cls_attended_tokens_list)

    return cls_attended_tokens_list


def extract_class_wise_attened_tokens_ft(texts_set, tokens_set, attention_set, y_pred_set, num_class, epoch, savename,
                                         topk):
    l_text = texts_set
    l_token = tokens_set
    l_attention = attention_set
    l_pred = y_pred_set

    cls_attended_tokens = dict()
    for idx in range(num_class):
        cls_attended_tokens[idx] = dict()

    all_attended_tokens = dict()

    l_num = l_text.shape[0]
    topk /= 100.0

    for l_idx in range(l_num):
        # cur_pl = l_pred[l_idx].argmax().item()
        cur_pl = l_pred[l_idx].item()
        cur_token = l_token[l_idx]
        cur_attention = l_attention[l_idx][:len(cur_token)]
        cur_topk = int(len(cur_token) * topk)
        topk_idx = torch.topk(cur_attention, cur_topk)[1]
        for top_idx in topk_idx:
            # top_token = cur_token[top_idx].encode('utf-8')
            top_token = cur_token[top_idx]
            if '#' in top_token or '[CLS]' in top_token or '[SEP]' in top_token or top_token.isdigit():
                continue
            else:
                if top_token in cls_attended_tokens[cur_pl].keys():
                    cls_attended_tokens[cur_pl][top_token] += 1
                else:
                    cls_attended_tokens[cur_pl][top_token] = 1

                if top_token in all_attended_tokens.keys():
                    all_attended_tokens[top_token].add(cur_pl)
                else:
                    all_attended_tokens[top_token] = set()
                    all_attended_tokens[top_token].add(cur_pl)

    # filter stopping words
    stopping_words = [x for j in stopping_words_list for x in j]
    if 'imdb' in savename:
        imdb_stopping_words = [x for j in imdb_stopping_words_list for x in j]
        stopping_words += imdb_stopping_words
    for cur_token in stopping_words:
        for idx in range(num_class):
            if cur_token in cls_attended_tokens[idx].keys():
                del cls_attended_tokens[idx][cur_token]

    cls_attended_tokens_list = list()
    for cls_id in range(num_class):
        cur_cls_tokens = np.array(list(cls_attended_tokens[cls_id].keys()))
        cur_cls_tokens_shot = np.array(list(cls_attended_tokens[cls_id].values()))
        sorted_idx = np.argsort(-1 * cur_cls_tokens_shot)
        cls_attended_tokens_list.append(list(cur_cls_tokens[sorted_idx]))
    np.save(savename, cls_attended_tokens_list)

    return cls_attended_tokens_list
