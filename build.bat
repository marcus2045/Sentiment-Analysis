@echo off
REM Builds "Sentiment Analyzer.exe" into the dist folder.
REM Run this from the project folder (where gui.py, main.py and walrus.ico are).

echo Installing requirements...
python -m pip install -r requirements.txt pyinstaller || goto :error

echo Downloading the VADER word list to bundle inside the exe...
python -m nltk.downloader -d nltk_data vader_lexicon || goto :error

echo Building the exe...
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "Sentiment Analyzer" ^
  --icon walrus.ico ^
  --add-data "nltk_data;nltk_data" ^
  gui.py || goto :error

echo.
echo Done! Your app is at: dist\Sentiment Analyzer.exe
pause
exit /b 0

:error
echo.
echo Build failed - see the messages above.
pause
exit /b 1
