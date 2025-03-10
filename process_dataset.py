import pandas as pd
import re

def clear_spaces_inside(text):
    words = text.split()
    words = list(map(lambda x: x.strip(), words))
    text_clear = ' '.join(words)
    
    return text_clear

if __name__ == "__main__":
    PATH_TO_EXCEL = ""
    qa_df = pd.read_excel(PATH_TO_EXCEL)

    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    link_pattern = r'(https?://[^\s]+|www\.[^\s]+)'

    qa_df['content_clear'] = qa_df['content'].apply(lambda x: x.lower().strip())
    qa_df['content_clear'] = qa_df['content_clear'].str.replace(email_pattern, 'MAIL', regex=True)
    qa_df['content_clear'] = qa_df['content_clear'].str.replace(link_pattern, 'LINK', regex=True)
    qa_df['content_clear'] = qa_df['content_clear'].str.replace('+7 (xxx) xxx xx xx', 'PHONE', regex=False)
    qa_df['content_clear'] = qa_df['content_clear'].apply(clear_spaces_inside)

    qa_df.to_excel(qa_df, index=False)