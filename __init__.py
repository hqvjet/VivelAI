import torch
from constant import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('USING GPU')
        device = torch.device('cpu')
    else:
        print('USING CPU')
        device = torch.device('cpu')

    ext_model_key = input('Choose feature extraction model:\n1. E2T-PhoBERT\n2. PhoBERT\n3. VISOBERT\n4. E2V-PhoBERT\nYour Input:\n')
    ds_key = input('Choose dataset:\n1. AIVIVN\n2. UIT-ViHSD\n3. UIT-VSFC\nYour Input:\n')
    act_key = input('What do you want to do ?:\n1. Extract Feature\n2. Train Model\n3. Run server\nYour Input:\n')

    if ext_model_key == '1':
        ext_model = E2T_PHOBERT
    elif ext_model_key == '2':
        ext_model = PHOBERT
    elif ext_model_key == '3':
        ext_model = VISOBERT
    elif ext_model_key == '4':
        ext_model = E2V_PHOBERT

    if ds_key == '1':
        ds = AIVIVN
    elif ds_key == '2':
        ds = UIT_VIHSD
    else:
        ds = UIT_VSFC

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
        else:
            model = BIGRU_CNN_TRANS_ENC

        startTraining(device, model=model, dataset=ds, extract_model=ext_model)

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
