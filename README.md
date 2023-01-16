# voyageAnalysis

to run this on your own machine first clone the repo, in a terminal go into a folder where you want to save this repo and type (only need to do this once):

```git clone https://github.com/kylemath/voyageAnalysis```

Then enter the created folder

```cd voyageAnalysis```

Then create a virtual environment called "venv" to install dependencies in (only need to do this once)

```python3 -m venv venv```

Then activate that environment (do this each time):

```source venv/bin/activate```

Then install the dependencies (only need to do this once or if we add a dependency): 

```pip install -r requirements.txt```

Then run the two version of the code, either:

```python VoyageLoadParse.py```

which allows you to specify the filename if you edit the code

or you can run another version which allows you to also indicate the filename

```python parse_csv.py VoyageRecording.csv```

