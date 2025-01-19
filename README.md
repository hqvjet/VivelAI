# Vietnamese Sentiment Analysis
## Step to run project:
### Downnload VnCoreNLP Tools:
- `mkdir -p tools/vncorenlp/models/wordsegmenter`
- `wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar`
- `wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab`
- `wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr`
- `mv VnCoreNLP-1.1.1.jar tools/vncorenlp/`
- `mv vi-vocab tools/vncorenlp/models/wordsegmenter/`
- `mv wordsegmenter.rdr tools/vncorenlp/models/wordsegmenter/`
### Setup Model Folders:
- `mkdir res/features`
- `mkdir res/models`
- `mkdir res/models/no_title`
- `mkdir res/models/no_title/phobert`
- `mkdir res/models/with_title`
- `mkdir res/models/with_title/phobert`
- `mkdir res/models/with_icon`
- `mkdir res/models/with_icon/phobert`
- `mkdir res/report/with_icon/phobert`
- `mkdir res/train_process/with_icon/phobert`
### Install Requirements Libs:
- `python3 -m venv venv` for creating virtual enviroment, you can active this env by `source venv/bin/activate` - Linux or `venv/script/activate` - Window
- `pip install -r res/requirements.py`
- run project by `python3 __init__.py` at your_path/project_name/
