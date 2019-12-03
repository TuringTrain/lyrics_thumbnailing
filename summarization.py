import summa
from scipy.stats import wasserstein_distance, hmean
from sklearn.cluster import DBSCAN
import joblib
import listtools
from lyrics_reader import get_sentences, text_from, get_tree_segments_lines_token
import numpy as np
import warnings
from utils import string_similarity, self_similarity_matrix

############################
##### TEXTRANK SUMMARY #####
############################

# SETUP NOTE:
# in order for summa to accept lists of sentences (instead of raw text), we
# changed summarizer.py and textcleaner.py in the summa code (./summa/*)

def sorted_summary(full_summary, sorted_lyric, summary_length):
    smr_to_index = []
    for i in range(min(summary_length, len(full_summary))):
        smr_line = full_summary[i][0]
        smr_to_index.append((smr_line, sorted_lyric.index(smr_line)))
    smr_sorted = sorted(smr_to_index, key=lambda x: x[1])
    smr = [x for (x, _) in smr_sorted]
    return smr


def textrank_summary(lyric_lines, summary_length):
    lyric = listtools.make_strict(lyric_lines)
    linebased_textrank = summa.summarizer.summarize(lyric, scores=True, ratio=1)
    textrank_summary = sorted(linebased_textrank, key=lambda x: x[1], reverse=True)
    return sorted_summary(textrank_summary, lyric, summary_length)


##############################
##### TOPICRANK SUMMARY ######
##############################
def topic_distribution(text, tfidf_vectorizer, nmf_model):
    disti = nmf_model.transform(tfidf_vectorizer.transform([text])).reshape(1, -1)[0]
    return disti


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def display_topic(model, feature_names, no_top_words, topic_idx):
    topic = model.components_[topic_idx]
    print("Topic %d:" % (topic_idx), " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def topicrank_summary(lyric_lines, summary_length):
    warnings.filterwarnings('ignore')
    tfidf_vectorizer, _ = joblib.load('nmf_topic_model/tfidf_mpd.jl')
    nmf_model, _ = joblib.load('nmf_topic_model/nmf_mpd.jl')

    lyric_lines = listtools.make_strict(lyric_lines)
    lyric_str = get_sentences(lyric_lines)

    original_topics = topic_distribution(lyric_str, tfidf_vectorizer, nmf_model)
    original_topic_indices_sorted = original_topics.argsort().tolist()
    orig_topics = 3
    original_topic_indices_top3 = [original_topic_indices_sorted[-i] for i in range(1, orig_topics + 1)]
    original_topics_arr = original_topics.tolist()
    original_topics_top3 = [original_topics_arr[i] for i in original_topic_indices_top3]

    # for topic_idx in original_topic_indices_top3:
    #     display_topic(nmf_model, tfidf_vectorizer.get_feature_names(), no_top_words=10, topic_idx=topic_idx)
    # print('Original topics-3:', list(map(lambda x: round(x*100, 1), original_topics_top3)))

    # build summary as list of its line indices
    summary = ''
    summary_lines = []
    # assuming unique lines
    unused_lines = set(lyric_lines)
    for i in range(summary_length):
        line_to_distance = []
        line_to_topics_top3 = dict()
        for line in unused_lines:
            inc_summary = summary + '.' + line
            line_topics = topic_distribution(inc_summary, tfidf_vectorizer, nmf_model)
            line_topics_arr = line_topics.tolist()
            line_topics_top3 = [line_topics_arr[i] for i in original_topic_indices_top3]
            line_to_topics_top3[line] = line_topics_top3
            line_distance = wasserstein_distance(original_topics_top3, line_topics_top3)
            line_to_distance.append((line, line_distance))
        line_to_distance = [x for (x, s) in sorted(line_to_distance, key=lambda x: x[1])]
        best_line = line_to_distance[0]
        summary += best_line + '.'
        summary_lines.append(best_line)
        # print(list(map(lambda x: round(x*100, 1), line_to_topics_top3[best_line])), round(100 * wasserstein_distance(original_topics_top3, line_to_topics_top3[best_line]), 3))
        unused_lines.remove(best_line)
    summary_lines = sorted(summary_lines, key=lambda x: lyric_lines.index(x))
    return summary_lines


###########################
##### LYRIC THUMBNAIL #####
###########################
def cluster_coverage(clusters, lyric_length):
    coverage = dict()
    for cluster_id in clusters:
        coverage_of_elem = 0
        for elem in clusters[cluster_id]:
            coverage_of_elem += len(elem)
        coverage[cluster_id] = coverage_of_elem / lyric_length
    return coverage


def cluster_precision(clusters, similarity_measure):
    precisions = dict()
    for cluster_id in clusters:
        cluster_for_id = clusters[cluster_id]
        precisions_for_id = []
        for (i, j) in cross_indices_symmetric(len(cluster_for_id)):
            precisions_for_id.append(similarity_measure(text_from(cluster_for_id[i]),
                                                        text_from(cluster_for_id[j])))
        precisions[cluster_id] = precisions_for_id
    precisions_avg = dict()
    for pid in precisions:
        precisions_avg[pid] = np.average(precisions[pid])
    return precisions_avg


def cross_indices_symmetric(rng):
    res = []
    for i in range(rng):
        for j in range(rng):
            res.append((i, j))
    res_strict = []
    for (i, j) in res:
        if (j, i) in res_strict or i == j:
            continue
        res_strict.append((i, j))
    return res_strict


def cluster_fitness(precision, coverage):
    assert precision.keys() == coverage.keys()
    fitness = dict()
    for cluster_id in precision:
        fitness[cluster_id] = hmean([precision[cluster_id], coverage[cluster_id]])
    return fitness


def cluster_lyric_segments(segments, similarity_measure=string_similarity, min_similarity=0.7):
    X = self_similarity_matrix(segments, lambda x, y: 1 - similarity_measure(x, y))
    db = DBSCAN(eps=1 - min_similarity, min_samples=2, metric='precomputed').fit(X)
    labels = db.labels_
    # write cluster dictionary from cluster result labels
    clusters = dict()
    for segment_index in range(len(segments)):
        label = labels[segment_index]
        if label < 0:
            continue
        if not label in clusters:
            cluster_of_label = []
        else:
            cluster_of_label = clusters[label]
        cluster_of_label.append(segments[segment_index])
        clusters[label] = cluster_of_label
    return clusters


def lyric_thumbnail(raw_lyric, size=2):
    tree, segments, lines, token = get_tree_segments_lines_token(raw_lyric)
    thumbnail = []
    lyric_clusters = cluster_lyric_segments(segments)
    lyric_fitness = cluster_fitness(cluster_precision(lyric_clusters, similarity_measure=string_similarity),
                                    cluster_coverage(lyric_clusters, len(lines)))
    fitness = [(cid, fit) for (cid, fit) in lyric_fitness.items()]
    fitness = sorted(fitness, key=lambda x: x[1], reverse=True)
    size = min(len(lyric_clusters), size)
    if size == 0:
        return []
    fittest_index = fitness[0][0]
    fittest_segment = lyric_clusters[fittest_index][0]
    if size == 1:
        return fittest_segment
    else:
        second_fittest_index = fitness[1][0]
        second_fittest_segment = lyric_clusters[second_fittest_index][0]
        if segments.index(fittest_segment) < segments.index(second_fittest_segment):
            thumbnail = fittest_segment + second_fittest_segment
        else:
            thumbnail = second_fittest_segment + fittest_segment
    return thumbnail


def transition_thumbnail(thumbnail, allowed_thumbnail_linecount):
    thumbnail_linecount = len(thumbnail)
    if thumbnail_linecount <= allowed_thumbnail_linecount:
        return thumbnail
    cut_start = int(np.ceil((thumbnail_linecount - allowed_thumbnail_linecount) / 2))
    cut_end = int(np.floor((thumbnail_linecount - allowed_thumbnail_linecount) / 2))
    transition_thumbnail = thumbnail[cut_start: len(thumbnail) - cut_end]
    return transition_thumbnail


def lines_fitness(raw_lyric):
    tree, segments, lines, token = get_tree_segments_lines_token(raw_lyric)
    lyric_clusters = cluster_lyric_segments(segments)
    lyric_fitness = cluster_fitness(cluster_precision(lyric_clusters, similarity_measure=string_similarity),
                                    cluster_coverage(lyric_clusters, len(lines)))
    line_to_fitness = dict()
    for cluster_id in lyric_clusters:
        for segment in lyric_clusters[cluster_id]:
            for line in lines:
                fitness_for_line = line_to_fitness[line] if line in line_to_fitness else []
                fitness_for_line.append(lyric_fitness[cluster_id])
                line_to_fitness[line] = fitness_for_line
    for line in lines:
        fitness_for_line = line_to_fitness[line] if line in line_to_fitness else []
        for _ in range(lines.count(line) - len(fitness_for_line)):
            fitness_for_line.append(0)
        line_to_fitness[line] = fitness_for_line
    line_to_fitness = [(k, np.average(v)) for (k, v) in line_to_fitness.items()]
    return line_to_fitness


############################
##### COMBINED SUMMARY #####
############################
def normalized_ranks_list(elems_scores, higherScoreIsBetter):
    sorted_elems_scores = sorted(elems_scores, key=lambda x: x[1], reverse=higherScoreIsBetter)
    score_to_rank = listtools.make_strict([s for (_, s) in sorted_elems_scores])
    sorted_elems_ranks = list(map(lambda x: (x[0], score_to_rank.index(x[1])), sorted_elems_scores))
    max_rank = sorted_elems_ranks[-1][1]
    if max_rank == 0:
        return list(map(lambda x: (x[0], 0), sorted_elems_ranks))
    return list(map(lambda x: (x[0], x[1] / max_rank), sorted_elems_ranks))


def dict_from_pairlist(pair_list):
    return {k: v for (k, v) in pair_list}


def normalized_ranks(elems_scores, higherScoreIsBetter):
    return dict_from_pairlist(normalized_ranks_list(elems_scores, higherScoreIsBetter))


def textrank_topicrank_summary(raw_lyric, summary_length):
    warnings.filterwarnings('ignore')
    tfidf_vectorizer, _ = joblib.load('nmf_topic_model/tfidf_mpd.jl')
    nmf_model, _ = joblib.load('nmf_topic_model/nmf_mpd.jl')

    tree, segments, lyric_lines, token = get_tree_segments_lines_token(raw_lyric)
    lyric_lines_strict = listtools.make_strict(lyric_lines)
    lyric_str = text_from(lyric_lines_strict)

    # static TEXTRANK per line
    segmentwise_textrank = summa.summarizer.summarize(lyric_lines_strict, scores=True, ratio=1)
    assert len(segmentwise_textrank) == len(lyric_lines_strict), 'Gutt'
    # line_to_textrank = [line for (line,_) in sorted(segmentwise_textrank, key=lambda x: x[1], reverse=True)]
    line_to_textrank = normalized_ranks(segmentwise_textrank, higherScoreIsBetter=True)

    original_topics = topic_distribution(lyric_str, tfidf_vectorizer, nmf_model)
    original_topic_indices_sorted = original_topics.argsort().tolist()
    orig_topics = 3
    original_topic_indices_top3 = [original_topic_indices_sorted[-i] for i in range(1, orig_topics + 1)]
    original_topics_arr = original_topics.tolist()
    original_topics_top3 = [original_topics_arr[i] for i in original_topic_indices_top3]
    #     for topic_idx in original_topic_indices_top3:
    #         display_topic(nmf_model, tfidf_vectorizer.get_feature_names(), no_top_words=10, topic_idx=topic_idx)
    #     print('Original topics-3:', list(map(lambda x: round(x*100, 1), original_topics_top3)))

    # build summary as list of its line indices
    summary = ''
    summary_lines = []
    summary_scores = []
    # assuming unique lines
    unused_lines = set(lyric_lines)
    for i in range(summary_length):
        line_to_distance = []
        line_to_topics_top3 = dict()
        for line in unused_lines:
            inc_summary = summary + '.' + line
            line_topics = topic_distribution(inc_summary, tfidf_vectorizer, nmf_model)
            line_topics_arr = line_topics.tolist()
            line_topics_top3 = [line_topics_arr[i] for i in original_topic_indices_top3]
            line_to_topics_top3[line] = line_topics_top3
            line_distance = wasserstein_distance(original_topics_top3, line_topics_top3)
            line_to_distance.append((line, line_distance))
        remaining_lines = [x for (x, s) in sorted(line_to_distance, key=lambda x: x[1])]
        line_to_topicrank = normalized_ranks(line_to_distance, higherScoreIsBetter=False)

        #         print('XXX',line_to_textrank)
        #         print('ZZZ', line_to_topicrank)
        # EVALUATION CRITERIA here
        min_rank_line_index = np.argmin(
            list(map(lambda line: line_to_textrank[line] + line_to_topicrank[line], remaining_lines)))
        best_line = remaining_lines[min_rank_line_index]
        summary += best_line + '.'
        summary_lines.append(best_line)
        #         print(list(map(lambda x: round(x*100, 1), line_to_topics_top3[best_line])), round(100 * wasserstein_distance(original_topics_top3, line_to_topics_top3[best_line]), 3))
        unused_lines.remove(best_line)
        summary_scores.append((line_to_textrank[best_line], line_to_topicrank[best_line]))
    summary_lines = sorted(summary_lines, key=lambda x: lyric_lines.index(x))
    #     cumulative_scores = reduce(lambda x, elem: (x[0] + elem[0], x[1] + elem[1]), summary_scores, (0,0))
    #     print(cumulative_scores)
    return summary_lines


def textrank_topicrank_fitness_summary(raw_lyric, summary_length):
    warnings.filterwarnings('ignore')
    tfidf_vectorizer, _ = joblib.load('nmf_topic_model/tfidf_mpd.jl')
    nmf_model, _ = joblib.load('nmf_topic_model/nmf_mpd.jl')

    tree, segments, lyric_lines, token = get_tree_segments_lines_token(raw_lyric)
    lyric_lines_strict = listtools.make_strict(lyric_lines)
    lyric_str = text_from(lyric_lines_strict)

    # static TEXTRANK per line
    segmentwise_textrank = summa.summarizer.summarize(lyric_lines_strict, scores=True, ratio=1)
    assert len(segmentwise_textrank) == len(lyric_lines_strict), 'Gutt'
    # line_to_textrank = [line for (line,_) in sorted(segmentwise_textrank, key=lambda x: x[1], reverse=True)]
    line_to_textrank = normalized_ranks(segmentwise_textrank, higherScoreIsBetter=True)

    # static line fitness
    line_to_fitness = normalized_ranks(lines_fitness(raw_lyric), higherScoreIsBetter=True)

    original_topics = topic_distribution(lyric_str, tfidf_vectorizer, nmf_model)
    original_topic_indices_sorted = original_topics.argsort().tolist()
    orig_topics = 3
    original_topic_indices_top3 = [original_topic_indices_sorted[-i] for i in range(1, orig_topics + 1)]
    original_topics_arr = original_topics.tolist()
    original_topics_top3 = [original_topics_arr[i] for i in original_topic_indices_top3]
    #     for topic_idx in original_topic_indices_top3:
    #         display_topic(nmf_model, tfidf_vectorizer.get_feature_names(), no_top_words=10, topic_idx=topic_idx)
    #     print('Original topics-3:', list(map(lambda x: round(x*100, 1), original_topics_top3)))

    # build summary as list of its line indices
    summary = ''
    summary_lines = []
    summary_scores = []
    # assuming unique lines
    unused_lines = set(lyric_lines)
    for i in range(summary_length):
        line_to_distance = []
        line_to_topics_top3 = dict()
        for line in unused_lines:
            inc_summary = summary + '.' + line
            line_topics = topic_distribution(inc_summary, tfidf_vectorizer, nmf_model)
            line_topics_arr = line_topics.tolist()
            line_topics_top3 = [line_topics_arr[i] for i in original_topic_indices_top3]
            line_to_topics_top3[line] = line_topics_top3
            line_distance = wasserstein_distance(original_topics_top3, line_topics_top3)
            line_to_distance.append((line, line_distance))
        remaining_lines = [x for (x, s) in sorted(line_to_distance, key=lambda x: x[1])]
        line_to_topicrank = normalized_ranks(line_to_distance, higherScoreIsBetter=False)

        # EVALUATION CRITERIA here
        min_rank_line_index = np.argmin(list(
            map(lambda line: line_to_textrank[line] + line_to_topicrank[line] + line_to_fitness[line],
                remaining_lines)))
        best_line = remaining_lines[min_rank_line_index]
        summary += best_line + '.'
        summary_lines.append(best_line)
        #         print(list(map(lambda x: round(x*100, 1), line_to_topics_top3[best_line])), round(100 * wasserstein_distance(original_topics_top3, line_to_topics_top3[best_line]), 3))
        unused_lines.remove(best_line)
        summary_scores.append((line_to_textrank[best_line], line_to_topicrank[best_line], line_to_fitness[best_line]))
    summary_lines = sorted(summary_lines, key=lambda x: lyric_lines.index(x))
    #     cumulative_scores = reduce(lambda x, elem: (x[0] + elem[0], x[1] + elem[1], x[2] + elem[2]), summary_scores, (0,0,0))
    #     print(cumulative_scores)
    return summary_lines
