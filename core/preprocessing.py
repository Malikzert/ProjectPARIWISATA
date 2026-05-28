import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

CUSTOM_STOPWORDS = {
    'yg', 'nyg', 'nya', 'gk', 'nggak', 'ga', 'gak', 'tdk', 'tak',
    'klo', 'kl', 'kalo', 'kalau', 'aja', 'aj', 'dgn', 'dg', 'dng',
    'utk', 'buat', 'sih', 'deh', 'dong', 'kok', 'loh', 'lho',
    'sya', 'sy', 'gw', 'gue', 'lu', 'lo', 'kmu', 'km',
    'udh', 'udah', 'sdh', 'suda', 'sodah', 'ud',
    'krn', 'karna', 'karena', 'dr', 'dar', 'dari',
    'tp', 'tpi', 'tap', 'tapi', 'tpt', 'tmpat',
    'jgn', 'jngn', 'jangan',
    'spt', 'sprti', 'seperti',
    '#', 'pada', 'untuk', 'dan', 'di', 'ke', 'dengan',
    'yang', 'ini', 'itu', 'dan', 'di', 'ke', 'dari',
    'saya', 'kami', 'kita', 'anda', 'mereka',
    'adalah', 'ialah', 'yaitu', 'yakni',
    'telah', 'sudah', 'sedang', 'akan', 'belum',
    'dapat', 'bisa', 'mampu', 'dapat',
    'sangat', 'amat', 'sekali', 'paling',
    'lebih', 'kurang', 'cukup', 'agak',
    'juga', 'pula', 'lagi', 'masih',
    'hanya', 'saja', 'sebagai', 'secara',
    'oleh', 'dalam', 'tentang', 'antara',
    'setelah', 'sebelum', 'sejak', 'hingga',
    'sambil', 'tetapi', 'namun', 'melainkan',
    'atau', 'maupun', 'baik',
    'seperti', 'bagaikan', 'ibarat',
    'walaupun', 'meski', 'meskipun', 'biarpun',
    'jika', 'kalau', 'apabila', 'bila',
    'maka', 'maka dari itu', 'sehingga',
    'bahwa', 'apakah', 'siapa', 'mengapa', 'bagaimana',
    'ada', 'tidak', 'bukan', 'tiada',
    'ya', 'oh', 'wah', 'ah', 'hai',
    'pak', 'bu', 'mas', 'mbak',
}

def preprocess_indonesian(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in CUSTOM_STOPWORDS]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def clean_text(text):
    return preprocess_indonesian(text)
