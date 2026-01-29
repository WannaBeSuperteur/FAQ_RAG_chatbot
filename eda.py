import pickle
import numpy as np


if __name__ == '__main__':
    with open('final_result.pkl', 'rb') as f:
        documents = pickle.load(f)

    doc_lengths = []

    for doc in documents:
        doc_lengths.append(len(str(doc)))

    print('DOCUMENT LENGTH STATISTICS :')
    print(f'count : {len(documents)}')
    print(f'max   : {np.max(doc_lengths)}')
    print(f'min   : {np.min(doc_lengths)}')
    print(f'avg   : {np.mean(doc_lengths)}')
    print(f'std   : {np.std(doc_lengths)}')

    print('\nFIRST 10 DOCUMENTS (PREVIEW):')
    for idx, doc in enumerate(documents):
        if idx >= 10:
            break
        print(f'{idx+1}. {doc}')
