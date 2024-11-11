import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('USING GPU')
        device = torch.device('cuda')
    else:
        print('USING CPU')
        device = torch.device('cpu')

    key = input('What do you want to do ?:\n1. Extract Feature\n2. Train Model\n3. Run server\nYour Input: ')

    if key == '1':
        from feature_extract import useFeatureExtractor
        useFeatureExtractor(device)
    elif key == '2':
        from models import startTraining
        startTraining(device)
    elif key == '3':
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=4567,
            reload=True
        )
    else:
        print('Key error!')
    print('End Session !')
