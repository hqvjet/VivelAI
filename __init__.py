import torch
from constant import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('USING GPU')
        device = torch.device('cuda')
    else:
        print('USING CPU')
        device = torch.device('cpu')

    ds_key = input('Choose dataset:\n1. UIT-ViHSD\n2. UIT-VSFC\nYour Input:\n')
    act_key = input('What do you want to do ?:\n1. Extract Feature\n2. Train Model\nYour Input:\n')

    ext_model = E2T_PHOBERT
    if ds_key == '1':
        ds = UIT_VIHSD
    elif ds_key == '2':
        ds = UIT_VSFC
    else:
        print('Dataset Key error!')

    if act_key == '1':
        from feature_extract import useFeatureExtractor
        useFeatureExtractor(device, extract_model=ext_model, dataset=ds)

    elif act_key == '2':
        from models import startTraining

        model_key = input('Choose model:\n1. BiLSTM\n2. XGBoost\n3. LR\n4. GRU\n5. BiGRU\n6. CNN\n7. A-BiLSTM\n8. CNN Trans Enc\n9. BiGRU CNN Trans Enc\nYour Input:\n')

        if model_key == '1':
            model = BILSTM
        elif model_key == '2':
            model = XGBOOST
        elif model_key == '3':
            model = LR
        elif model_key == '4':
            model = GRU
        elif model_key == '5':
            model = BIGRU
        elif model_key == '6':
            model = CNN
        elif model_key == '7':
            model = ATTENTION_BILSTM
        elif model_key == '8':
            model = CNN_TRANS_ENC
        elif model_key == '9':
            model = BIGRU_CNN_TRANS_ENC
        else:
            print('Model Key error!')

        startTraining(device, model_name=model, dataset=ds, extract_model=ext_model)

    elif act_key == '3':
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
