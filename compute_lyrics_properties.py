from lyrics_reader import EXAMPLE_LYRIC, get_tree_segments_lines_token, get_sentences
from ssm_structure import self_similarity_matrix, draw_ssm, string_similarity
from summarization import textrank_summary, topicrank_summary, lines_fitness, lyric_thumbnail, textrank_topicrank_summary, textrank_topicrank_fitness_summary
from explicit_lyrics_detection import SWEARWORDS_DICTIONARY, predict_explicitness
from utils import headline_from

# parse example lyric
tree, segments, lines, token = get_tree_segments_lines_token(EXAMPLE_LYRIC)

def showcase_structure():
    """Shows self-similarity matrices (SSM) where entry (i,j) is the similarity between elements i and j."""
    # Structure of segments
    ssm_segments = self_similarity_matrix(segments, string_similarity)
    print(ssm_segments)
    draw_ssm(ssm_segments)
    # Structure of lines
    ssm_lines = self_similarity_matrix(lines, string_similarity)
    draw_ssm(ssm_lines)

def showcase_summaries():
    def print_summary(method: str, lines):
        print(headline_from(method + ' summary of length ' + str(len(lines))))
        for line in lines:
            print(line)
        print()
    SUMMARY_LENGTH = 7
    print_summary(method='TextRank', lines=textrank_summary(lines, SUMMARY_LENGTH))
    print_summary(method='TopSum', lines=topicrank_summary(lines, SUMMARY_LENGTH))
    print_summary(method='TextRank + TopSum', lines=textrank_topicrank_summary(EXAMPLE_LYRIC, SUMMARY_LENGTH))
    print_summary(method='TextRank + TopSum + Lyric Thumbnail', lines=textrank_topicrank_fitness_summary(EXAMPLE_LYRIC, SUMMARY_LENGTH))

def showcase_explicitness():
    print(headline_from('Predicting explicitness of lyric based on swear word dictionary:'))
    print('Example lyric is', predict_explicitness(EXAMPLE_LYRIC))

def showcase():
    showcase_structure()
    showcase_summaries()
    showcase_explicitness()

showcase()