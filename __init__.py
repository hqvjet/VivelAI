from feature_extract import useFeatureExtractor
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('USING GPU')
        device = torch.device('cuda')
    else:
        print('USING CPU')
        device = torch.device('cpu')

    useFeatureExtractor(device)
