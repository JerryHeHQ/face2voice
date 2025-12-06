# face2voice

## OpenVoice Setup

Follow these steps to set up **OpenVoice V2** for voice cloning:

### 1. Python Environment
Create a new conda environment and activate it (optional but recommended):

```bash
conda create -n openvoice python=3.9
conda activate openvoice
```

### 2. Repository Setup
Clone the OpenVoice repository into your project folder:

```bash
git clone https://github.com/myshell-ai/OpenVoice.git
```

The file structure should be:
```bash
face2voice
├─ OpenVoice/
```

Refer to the official Linux installation instructions for more details and model checkpoint downloads:
[OpenVoice V2 Usage Guide](https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md#openvoice-v2)



### 3. Install Dependencies
Install the required Python packages. Your requirements.txt should include:
```makefile
av==15.1.0
librosa==0.9.1
faster-whisper==1.2.1
pydub==0.25.1
wavmark==0.0.3
numpy==1.22.0
eng_to_ipa==0.0.2
inflect==7.0.0
unidecode==1.3.7
whisper-timestamped==1.14.2
openai
python-dotenv
pypinyin==0.50.0
cn2an==0.5.22
jieba==0.42.1
gradio==3.48.0
langid==1.1.6
Install them with:
```

```bash
cd OpenVoice
pip install -r requirements.txt
```

Note: If you encounter Cython build errors during installation, refer to this issue for solutions:
[OpenVoice Issue #462](https://github.com/myshell-ai/OpenVoice/issues/462)

### 4. Install FFmpeg
FFmpeg is required for audio processing. Install via conda:

```bash
conda install -c conda-forge ffmpeg
```

### 5. Verify Setup
Run the demo notebook to ensure everything works:

```bash
jupyter notebook OpenVoice/demo_part3.ipynb
```

If you encounter missing NLTK resources (for English G2P), add the following code at the top of the notebook:

```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')
```

### 6. Example
For a simplier example, see:
```bash
jupyter notebook openvoice_example.ipynb
```