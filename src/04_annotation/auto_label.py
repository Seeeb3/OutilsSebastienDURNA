import pandas as pd

def detect_regret(statement: str) -> bool:
    keywords = ['sorry', 'apologize', 'regret', 'forgive']
    return any(word in statement.lower() for word in keywords)

def detect_faith(statement: str) -> bool:
    keywords = ['god', 'jesus', 'lord', 'pray', 'faith']
    return any(word in statement.lower() for word in keywords)

def detect_family(statement: str) -> bool:
    keywords = ['mom', 'dad', 'sister', 'brother', 'family', 'wife', 'son', 'daughter']
    return any(word in statement.lower() for word in keywords)

def detect_justice(statement: str) -> bool:
    keywords = ['innocent', 'justice', 'truth', 'law', 'evidence', 'trial', 'judge', 'court']
    return any(word in statement.lower() for word in keywords)

def detect_peace(statement: str) -> bool:
    keywords = ['peace', 'at peace', 'calm', 'rest', 'free', 'ready']
    return any(word in statement.lower() for word in keywords)

def detect_forgiveness(statement: str) -> bool:
    keywords = ['forgive me', 'forgiveness', 'i forgive you']
    return any(word in statement.lower() for word in keywords)

def detect_reform(statement: str) -> bool:
    keywords = ['change', 'rehabilitate', 'different man', 'grew', 'evolved']
    return any(word in statement.lower() for word in keywords)

def label_statement(statement: str) -> list:
    labels = []
    if detect_regret(statement):
        labels.append('regret')
    if detect_faith(statement):
        labels.append('faith')
    if detect_family(statement):
        labels.append('family')
    if detect_justice(statement):
        labels.append('justice')
    if detect_peace(statement):
        labels.append('peace')
    if detect_forgiveness(statement):
        labels.append('forgiveness')
    return labels if labels else ['none']

def main():
    df = pd.read_csv('data/processed/death_row_clean.csv')
    df['auto_labels'] = df['last_statement'].fillna('').apply(label_statement)
    df.to_csv('data/processed/death_row_labeled.csv', index=False)
    print("Annotations automatiques sauvegardées dans data/processed/death_row_labeled.csv")

if __name__ == "__main__":
    main()