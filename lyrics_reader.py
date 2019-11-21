import html
from functools import reduce

EXAMPLE_LYRIC = 'The whispers in the morning<br>Of lovers sleeping tight<br>Are rolling by like thunder now<br>As I look in your eyes<br><br>I hold on to your body<br>And feel each move you make<br>Your voice is warm and tender<br>A love that I could not forsake<br><br>&apos;Cause I&apos;m your lady<br>And you are my man<br>Whenever you reach for me<br>I&apos;ll do all that I can<br><br>Lost is how I&apos;m feeling<br>Lying in your arms<br>When the world outside&apos;s too much to take<br>That all ends when I&apos;m with you<br><br>Even though there may be times<br>It seems I&apos;m far away<br>Never wonder where I am<br>&apos;Cause I am always by your side<br><br>&apos;Cause I&apos;m your lady<br>And you are my man<br>Whenever you reach for me<br>I&apos;ll do all that I can<br><br>We&apos;re heading for something<br>Somewhere I&apos;ve never been<br>Sometimes I am frightened but I&apos;m ready to learn<br>Of the power of love<br><br>The sound of your heart beating<br>Made it clear suddenly<br>The feeling that I can&apos;t go on<br>Is light years away<br><br>&apos;Cause I&apos;m your lady<br>And you are my man<br>Whenever you reach for me<br>I&apos;m gonna do all that I can<br><br>We&apos;re heading for something<br>Somewhere I&apos;ve never been<br>Sometimes I am frightened but I&apos;m ready to learn<br>Of the power of love<br><br>The power of love<br>The power of love<br>Sometimes I am frightened but I&apos;m ready to learn<br>Of the power of love<br><br>The power of love<br>Ooh, ooh, ooh<br>&apos;Cause I&apos;m ready to learn<br>Of the power of love'
BLOCK_SEPARATOR, LINE_SEPARATOR, TOKEN_SEPARATOR = '<br><br>', '<br>', lambda x: x.split()


def normalize_text(text, lowercase=True):
    text = html.unescape(text)
    if lowercase:
        text = text.lower()
    return text


# Read hierarchical blocks -> lines -> words structure from lyrics
def get_tree_segments_lines_token(text, block_separator=BLOCK_SEPARATOR,
                             line_separator=LINE_SEPARATOR, word_separator=TOKEN_SEPARATOR):
    text = normalize_text(text)
    tree = []
    all_segments = []
    all_lines = []
    all_token = []
    for block in text.split(block_separator):
        block = block.strip()
        sub_tree = []
        lines_in_segment = []
        for line in block.split(line_separator):
            line = line.strip()
            all_lines.append(line)
            lines_in_segment.append(line)
            words_from_line = word_separator(line)
            all_token.extend(words_from_line)
            sub_tree.append(words_from_line)
        tree.append(sub_tree)
        all_segments.append(' '.join(lines_in_segment))
    return tree, all_segments, all_lines, all_token

def text_from(tokens, separator=' '):
    return reduce(lambda x, elem: x + separator + elem, tokens, '').strip()

def get_sentences(lines):
    return (lambda t: text_from(t, separator='.'))(lines)
