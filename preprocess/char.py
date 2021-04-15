CHAR_DEFAULT = [
	# letter
	'a', 'æ', 'á', 'à', 'ả', 'ã', 'ạ', 'ă', 'ǎ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'b',
	'c', 'd', 'đ', 'e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'f', 'g', 'h', 'i', 'í', 'ï',
	'ỉ', 'î', 'ĩ', 'ị', 'ٱ', 'ì', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ',
	'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'p', 'q', 'r', 's', 't', 'µ', 'u', 'ú', 'ù', 'ü', 'ủ', 'ũ',
	'ụ', 'ư', 'ứ', 'ừ', 'ử' 'ữ', 'ự', 'v', 'w', 'x', 'y', 'ý', 'ỳ', 'ỹ', 'ỷ', 'ỵ', 'z',
	# special character
	'0', '+', '-', '^', '*', '/', '\\', '~', '=', "'", '`', '"', '(', ')', '[', ']', '{', '}', '|', '<', '>',
	'‘', '’', ',', '.', '–', '_', '#', '%', '&', ':', ';', '?', '!', '@', '©', '®', '$', '°', '•', '℃', 'ð', '¼', '‰', '́'
]

PAD_TOKEN = '<pad>'
PAD_INDEX = 0

UNK_TOKEN = '<unk>'
UNK_INDEX = 1

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '_root_'
ROOT_INDEX = 2

# 1 contain number, 2 lowercase, 3 uppercase
def word_format(word, position):
	# upper is normal at start of sentence
	if word[0].isupper() and position > 1 and word != 'LBKT' and word != 'RBKT':
		return 3
	for char in word:
		if char.isdigit():
			return 1
	return 2
